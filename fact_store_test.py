#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fact-Store Minimal Demo: 演示 "Search -> Assert -> Retrieve -> Answer" 的完整 Agent 流程。

核心特性：
1. 结构化内存 (Fact-Store): 代理显式断言三元组。
2. 证据回溯 (Evidence DB): 代理可按需查阅事实对应的原始文本。
3. 软匹配奖励 (Soft Match): 使用 BGE 向量计算断言与黄金事实的相似度，模拟中间奖励。

运行环境约定同上：GPU 5, 本地模型路径。
"""

import os
import re
import torch
import numpy as np
from typing import List, Dict, Tuple, Set

# 限制 GPU
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "7")

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel

# ================= 配置与语料 =================

# 模拟的“黄金标准事实库” (Gold Fact Store)
# 在实际训练中，这来自数据集的 Supporting Facts；这里我们根据 build_corpus 手动定义。
GOLD_FACT_STORE = [
    ("三体", "作者", "刘慈欣"),
    ("刘慈欣", "职业", "科幻作家"),
    ("刘慈欣", "出生地", "北京"),
    ("北京", "著名皇家园林", "颐和园"),
    ("颐和园", "始建朝代", "清朝"),
    ("颐和园", "建造者", "乾隆皇帝"),
]

def build_corpus() -> List[Dict[str, str]]:
    docs = [
        {
            "title": "三体 (小说)", 
            "text": "《三体》是刘慈欣创作的长篇科幻小说系列，由《三体》、《三体II·黑暗森林》、《三体III·死神永生》组成。"
        },
        {
            "title": "刘慈欣生平", 
            "text": "刘慈欣，1963年6月出生于北京，祖籍河南罗山。他是中国科幻小说的代表人物。虽然他在山西阳泉长大并工作，但他的出生地是北京。"
        },
        {
            "title": "阳泉市概况", 
            "text": "阳泉市位于山西省东部，是一座新兴工业城市。这里有著名的娘子关旅游景区。"
        },
        {
            "title": "北京市地标", 
            "text": "北京是中国的首都，拥有众多历史名胜，如故宫、天坛、颐和园等。其中颐和园是中国保存最完整的皇家行宫御苑。"
        },
        {
            "title": "颐和园历史", 
            "text": "颐和园，前身为清漪园，始建于清朝乾隆十五年（1750年）。它是以昆明湖、万寿山为基址，按照江南园林的设计手法建造的。"
        },
        {
            "title": "圆明园", 
            "text": "圆明园坐落在北京西北郊，由圆明、长春、绮春三园组成，也始建于清朝，但在战争中遭到严重破坏。"
        },
        {
            "title": "莫言", 
            "text": "莫言，中国当代作家，2012年获得诺贝尔文学奖，出生于山东高密。"
        }
    ]
    formatted_docs = []
    for d in docs:
        d["contents"] = f'Title: "{d["title"]}"\nContent: {d["text"]}'
        formatted_docs.append(d)
        
    return formatted_docs
# ================= 模型加载与基础工具 =================

class ModelEngine:
    def __init__(self):
        self.llm_path = "/data1/shares/Qwen2.5-7B-Instruct"
        self.bge_path = "/data1/shares/e5-base-v2   " 
        
        print(">>> Loading Models...")
        self.tokenizer_llm = AutoTokenizer.from_pretrained(self.llm_path, trust_remote_code=True)
        self.model_llm = AutoModelForCausalLM.from_pretrained(
            self.llm_path, trust_remote_code=True, torch_dtype=torch.float16, use_flash_attention_2=False
        ).eval().cuda()
        
        self.tokenizer_bge = AutoTokenizer.from_pretrained(self.bge_path, use_fast=True)
        self.model_bge = AutoModel.from_pretrained(self.bge_path).eval().cuda()
        
        self.corpus = build_corpus()
        self.doc_embs = self.encode_texts([d["contents"] for d in self.corpus], is_query=False)
        self.gold_embs = self.encode_texts([f"{s} {r} {o}" for s,r,o in GOLD_FACT_STORE], is_query=False)

    @torch.no_grad()
    def encode_texts(self, texts: List[str], is_query: bool) -> np.ndarray:
        if not texts: return np.array([])
        if is_query:
            texts = [f"为这个句子生成检索向量: {q}" for q in texts]
        inputs = self.tokenizer_bge(texts, max_length=256, padding=True, truncation=True, return_tensors="pt")
        inputs = {k: v.cuda() for k, v in inputs.items()}
        outputs = self.model_bge(**inputs, return_dict=True)
        cls_emb = outputs.last_hidden_state[:, 0]
        cls_emb = torch.nn.functional.normalize(cls_emb, dim=-1)
        return cls_emb.detach().cpu().numpy().astype(np.float32)

    def generate(self, messages: List[Dict]) -> str:
        input_ids = self.tokenizer_llm.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors='pt'
        ).cuda()
        out = self.model_llm.generate(
            input_ids=input_ids, max_new_tokens=512, do_sample=True, temperature=0.3, top_p=0.9,
            pad_token_id=self.tokenizer_llm.pad_token_id
        )
        gen_ids = out[0][input_ids.shape[-1]:]
        return self.tokenizer_llm.decode(gen_ids, skip_special_tokens=True)

    def search(self, query: str, top_k=2) -> str:
        """
        Search 返回纯文本字符串
        """
        q_emb = self.encode_texts([query], is_query=True)
        scores = (q_emb @ self.doc_embs.T)[0]
        top_idxs = scores.argsort()[::-1][:top_k]
        
        result_strs = []
        for i, idx in enumerate(top_idxs):
            doc = self.corpus[idx]
            result_strs.append(f"[Doc {i+1}] {doc['contents']}")
            
        return "\n\n".join(result_strs)

    def calculate_reward(self, triple: Tuple[str,str,str], current_context: str) -> Dict:
        """
        V2.9 核心逻辑: 简化的阈值分级奖励
        """
        t_text = f"{triple[0]} {triple[1]} {triple[2]}"
        t_emb = self.encode_texts([t_text], is_query=False)
        
        # 1. 计算与 Gold Facts 的相似度 (Ground Truth Matching)
        scores_gold = (t_emb @ self.gold_embs.T)[0]
        max_gold_idx = np.argmax(scores_gold) if len(scores_gold) > 0 else -1
        max_gold_score = scores_gold[max_gold_idx] if len(scores_gold) > 0 else 0.0
        
        # 2. 计算与当前 Context 的相似度 (用于区分 Distractor 和 Hallucination)
        max_ctx_score = 0.0
        if current_context:
            # 简单切句以提高匹配精度
            sentences = re.split(r'[。！？\n]', current_context)
            sentences = [s.strip() for s in sentences if len(s) > 5]
            if sentences:
                ctx_embs = self.encode_texts(sentences, is_query=False)
                scores_ctx = (t_emb @ ctx_embs.T)[0]
                max_ctx_score = np.max(scores_ctx)

        # 3. 分级奖励逻辑 (User Requested)
        
        # Case A: High Match (> 0.8) -> HIT
        if max_gold_score > 0.8:
            return {
                "reward": 0.5,
                "type": "HIT_GOLD",
                "gold_idx": max_gold_idx,
                "score": max_gold_score,
                "desc": "命中核心事实"
            }
            
        # Case B: Medium Match (0.5 - 0.8) -> WEAK MATCH
        elif 0.5 <= max_gold_score <= 0.8:
            return {
                "reward": 0.1,
                "type": "WEAK_MATCH",
                "gold_idx": -1,
                "score": max_gold_score,
                "desc": "语义相关/部分正确"
            }
            
        # Case C: Low Match (< 0.5) -> MISS / DISTRACTOR
        else:
            # 细分：如果是从 Context 里提取的有效信息 (但不是 Gold)，也给 -0.5
            if max_ctx_score > 0.8:
                return {
                    "reward": -0.5,
                    "type": "DISTRACTOR", # 提取了非目标事实
                    "gold_idx": -1,
                    "score": max_gold_score,
                    "desc": "干扰项(提取了非目标信息)"
                }
            else:
                return {
                    "reward": -1, # 或者 -1.0
                    "type": "HALLUCINATION",
                    "gold_idx": -1,
                    "score": max_gold_score,
                    "desc": "幻觉/完全错误"
                }

# ================= 核心 Agent 逻辑 (Update Here) =================
class FactStoreAgent:
    def __init__(self, engine: ModelEngine):
        self.engine = engine
        self.visible_facts = [] 
        self.evidence_db = {}   # dict: {fact_str: context_str}
        self.history_summary = [] 
        self.current_observation_str = None 
        self.searched_queries = set()
        self.covered_gold_indices = set() 

    def get_system_prompt(self) -> str:
        facts_str = "\n".join([f"{i+1}. <{f}>" for i, f in enumerate(self.visible_facts)])
        if not facts_str: facts_str = "(空)"
        
        return (
            "你是一个基于 Fact-Store 的推理 Agent。\n"
            "**核心原则**：\n"
            "1. 不要重复搜索。\n"
            "2. 必须从 Search 结果中提取事实，严禁臆造。\n\n"
            "**动作**：\n"
            "- <search>关键词</search>\n"
            "- <assert>主体, 关系, 客体</assert>\n"
            "- <retrieve>主体, 关系, 客体</retrieve>\n"
            "- <answer>答案</answer>\n\n"
            f"=== Fact-Store ===\n{facts_str}\n\n"
        )

    def build_context(self, query: str) -> List[Dict]:
        messages = [{"role": "system", "content": self.get_system_prompt()}]
        messages.append({"role": "user", "content": f"任务目标：{query}"})
        
        if self.history_summary:
            messages.append({"role": "user", "content": "历史操作：\n" + "\n".join(self.history_summary)})
        
        if self.current_observation_str:
            obs_prompt = (
                f"=== Observation ===\n"
                f"{self.current_observation_str}\n\n"
                f"指令：请从上述 Observation 中提取事实 (<assert>)。"
            )
            messages.append({"role": "user", "content": obs_prompt})
        else:
            msg = "指令：Fact-Store 为空，请 <search>。" if not self.visible_facts else "指令：请 <answer> 或补充 <search>。"
            messages.append({"role": "user", "content": msg})
                
        return messages
    def parse_and_execute(self, response: str) -> bool:
        # === Phase 1: Assert ===
        assert_matches = re.findall(r"<assert>(.*?)</assert>", response)
        assert_matches = [m.strip() for m in assert_matches if m.strip()]
        
        if assert_matches:
            print(f"   [Action] Processing {len(assert_matches)} Assertions...")
            step_reward = 0
            new_facts = []
            
            # 使用当前观测作为上下文进行判断
            context_for_reward = self.current_observation_str if self.current_observation_str else ""

            for content in assert_matches:
                parts = [p.strip() for p in content.split(',', 2)]
                if len(parts) == 3:
                    triple = (parts[0], parts[1], parts[2])
                    
                    reward_res = self.engine.calculate_reward(triple, context_for_reward)
                    r = reward_res["reward"]
                    
                    # Coverage Bonus (解锁新 Gold Fact 额外给分)
                    gold_idx = reward_res.get("gold_idx", -1)
                    bonus = 0.0
                    if reward_res["type"] == "HIT_GOLD":
                        if gold_idx not in self.covered_gold_indices:
                            self.covered_gold_indices.add(gold_idx)
                            bonus = 0.2
                            print(f"        [BONUS] New Gold Fact! (+0.2)")
                    
                    total_r = r + bonus
                    step_reward += total_r
                    
                    print(f"     -> Assert: {triple}")
                    print(f"        Type: [{reward_res['type']}] Sim: {reward_res['score']:.4f} Total: {total_r:+.1f}")
                    
                    # 只要是正向奖励 (Hit / Weak Match) 就存入 Fact-Store
                    # 即使是 Distractor/Hallucination (-0.5)，在 Demo 中我们选择不存入，以免污染 Context
                    if reward_res['type'] in ['HIT_GOLD', 'WEAK_MATCH']:
                        fact_str = f"{parts[0]}, {parts[1]}, {parts[2]}"
                        if fact_str not in self.visible_facts:
                            self.visible_facts.append(fact_str)
                            new_facts.append(fact_str)
                            self.evidence_db[fact_str] = context_for_reward
            
            if new_facts:
                self.history_summary.append(f"- Asserted {len(new_facts)} facts")
            elif assert_matches:
                # 记录失败尝试，提供反馈
                self.history_summary.append(f"- {len(assert_matches)} assertions rejected (low quality/distractor).")

            print(f"   [Step Total Reward] {step_reward:.2f}")
            self.current_observation_str = None

        # === Phase 2: Search ===
        search_match = re.search(r"<search>(.*?)</search>", response, re.DOTALL)
        if search_match:
            query = search_match.group(1).strip()
            if query in self.searched_queries:
                print(f"   [Blocked] 重复搜索: {query}")
            else:
                self.searched_queries.add(query)
                print(f"   [Action] Search: {query}")
                
                obs_str = self.engine.search(query)
                self.current_observation_str = obs_str
                
                self.history_summary.append(f"- Search: {query}")
                print(f"\n{'='*20} Retrieved Information {'='*20}")
                print(obs_str)
                print(f"{'='*62}\n")
                return True

        # === Phase 3: Retrieve ===
        retrieve_match = re.search(r"<retrieve>(.*?)</retrieve>", response, re.DOTALL)
        if retrieve_match:
            content = retrieve_match.group(1).strip()
            parts = [p.strip() for p in content.split(',', 2)]
            if len(parts) == 3:
                fact_key = f"{parts[0]}, {parts[1]}, {parts[2]}"
                print(f"   [Action] Retrieve: {fact_key}")
                
                source_text = self.evidence_db.get(fact_key, "未找到记录。")
                
                self.current_observation_str = f"Fact: <{fact_key}>\nSource:\n{source_text}"
                self.history_summary.append(f"- Retrieve: {fact_key}")
                print(f"\n{'='*20} Retrieved Evidence {'='*20}\n{source_text}\n{'='*62}\n")
                return True

        # === Phase 4: Answer ===
        answer_match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
        if answer_match:
            ans = answer_match.group(1).strip()
            print(f"\n>>> [Final Answer]: {ans}")
            return False 

        if assert_matches:
            return True

        print("   [Warning] 无法解析动作。")
        return True

    def run(self, query: str, max_steps=15):
        print(f"Task: {query}\n")
        for step in range(max_steps):
            print(f"--- Step {step + 1} ---")
            messages = self.build_context(query)
            print("\n[Model Input Messages]")
            for m in messages:
                content_preview = m['content']
                print(f"[{m['role'].upper()}]:\n{content_preview}")
                print("-" * 40)
            print("\n")
            response = self.engine.generate(messages)
            print(f"[LLM Output]: {response}")
            keep_going = self.parse_and_execute(response)
            if not keep_going:
                break
        print("\nTask Finished.")

if __name__ == "__main__":
    engine = ModelEngine()
    agent = FactStoreAgent(engine)
    query = "刘慈欣的出生地在哪里？该城市最著名的皇家园林是什么？它始建于哪个朝代？"
    agent.run(query)