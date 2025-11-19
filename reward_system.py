import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from typing import List, Tuple, Dict

class FactRewardModel:
    def __init__(self, model_name_or_path="/data1/shares/bge-large-zh-v1.5", device="cuda"):
        """
        初始化奖励模型。
        建议使用 BGE (BAAI General Embedding) 系列，在语义匹配任务上表现 SOTA。
        它比使用 Qwen 本身更轻量、更准确。
        """
        print(f"正在加载奖励模型 (Embedding): {model_name_or_path} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(model_name_or_path).to(device)
        self.device = device
        self.model.eval()
        
        # 阈值设定 (需要根据实验微调)
        self.GOLD_THRESHOLD = 0.85  # 视为命中黄金事实
        self.EVIDENCE_THRESHOLD = 0.75 # 视为被原文支持（非幻觉）

    def _get_embeddings(self, texts: List[str]):
        """批量计算文本向量"""
        # BGE 需要 query 指令 (对于检索任务)，但在做相似度对比时直接编码即可
        encoded_input = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt').to(self.device)
        with torch.no_grad():
            model_output = self.model(**encoded_input)
            # Perform pooling. In this case, cls pooling.
            sentence_embeddings = model_output.last_hidden_state[:, 0]
        # Normalize embeddings
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings

    def calculate_reward(self, 
                         agent_triple: Tuple[str, str, str], 
                         gold_triples: List[Tuple[str, str, str]], 
                         current_context: str) -> Dict:
        """
        计算单步 Assert 的奖励。
        
        Args:
            agent_triple: Agent 生成的三元组 (S, R, O)
            gold_triples: 当前样本的黄金三元组列表
            current_context: Agent 当前看到的 Search Result (用于判断是否是 Distractor)
            
        Returns:
            result: {
                "reward": float,
                "type": str ("hit", "noise", "hallucination"),
                "score": float
            }
        """
        # 1. 格式化文本
        triple_text = f"{agent_triple[0]}, {agent_triple[1]}, {agent_triple[2]}"
        gold_texts = [f"{t[0]}, {t[1]}, {t[2]}" for t in gold_triples]
        
        # 简单预处理：将 context 切分为句子，用于更精准的匹配
        # (实际使用可用 NLTK 或 Spacy 分句)
        context_sentences = [s.strip() for s in current_context.split('.') if len(s) > 10]

        # 2. 向量化
        # 将 agent triple, gold facts, context sentences 一起编码以节省开销
        all_texts = [triple_text] + gold_texts + context_sentences
        embeddings = self._get_embeddings(all_texts)
        
        agent_emb = embeddings[0].unsqueeze(0) # (1, dim)
        gold_embs = embeddings[1:1+len(gold_texts)] # (N_gold, dim)
        context_embs = embeddings[1+len(gold_texts):] # (N_context, dim)

        # 3. 计算与 Gold 的相似度
        if len(gold_embs) > 0:
            gold_scores = torch.mm(agent_emb, gold_embs.T).squeeze(0) # (N_gold, )
            max_gold_score = gold_scores.max().item()
        else:
            max_gold_score = 0.0

        # 4. 计算与 Context (Evidence) 的相似度
        if len(context_embs) > 0:
            evidence_scores = torch.mm(agent_emb, context_embs.T).squeeze(0)
            max_evidence_score = evidence_scores.max().item()
        else:
            max_evidence_score = 0.0

        # ================= 核心奖励逻辑 =================
        
        # 情况 1: 命中黄金事实 (Precision Bonus)
        if max_gold_score >= self.GOLD_THRESHOLD:
            return {
                "reward": 0.5,
                "type": "HIT_GOLD",
                "description": "成功提取关键事实",
                "score": max_gold_score
            }
            
        # 情况 2: 干扰项误导 (Noise Penalty)
        # 没命中 Gold，但是命中 Context，说明 Agent 读懂了，但是提取了无关信息
        elif max_evidence_score >= self.EVIDENCE_THRESHOLD:
            return {
                "reward": -0.5, 
                "type": "DISTRACTOR_NOISE",
                "description": "提取了真实但无关的信息 (Distractor)",
                "score": max_evidence_score
            }
            
        # 情况 3: 幻觉 (Hallucination Penalty)
        # 既不在 Gold 里，原文里也找不到对应证据
        else:
            return {
                "reward": -1.0,
                "type": "HALLUCINATION",
                "description": "事实在原文中不存在 (幻觉)",
                "score": max_evidence_score
            }

# ================== 单元测试演示 ==================
if __name__ == "__main__":
    # 模拟环境
    reward_model = FactRewardModel(device="cuda:7" if torch.cuda.is_available() else "cpu")
    
    # 场景设定：HotpotQA 关于 "Scott Derrickson"
    gold_facts = [
        ("Scott Derrickson", "director of", "Doctor Strange"),
        ("Doctor Strange", "released in", "2016")
    ]
    
    # 模拟检索到的文本 (包含无关信息/Distractor)
    retrieved_text = (
        "Scott Derrickson is an American director. He grew up in Denver, Colorado. "
        "He directed the movie Doctor Strange. "
        "Ed Wood was also an American director but he is famous for bad movies."
    )
    
    print(f"\nContext: {retrieved_text}\n")
    
    test_cases = [
        # Case A: 完美命中
        ("Scott Derrickson", "directed", "Doctor Strange"),
        # Case B: 真实但无关 (Distractor) -> 应该给 -0.5
        ("Scott Derrickson", "grew up in", "Denver"),
        # Case C: 真实但无关 (另一个实体的 Distractor) -> 应该给 -0.5
        ("Ed Wood", "occupation", "director"),
        # Case D: 幻觉 -> 应该给 -1.0
        ("Scott Derrickson", "directed", "Avatar") 
    ]
    
    for sub, rel, obj in test_cases:
        result = reward_model.calculate_reward(
            (sub, rel, obj), 
            gold_facts, 
            retrieved_text
        )
        print(f"Action: Assert({sub}, {rel}, {obj})")
        print(f"Result: {result['type']} | Reward: {result['reward']} | Sim: {result['score']:.4f}")
        print("-" * 50)