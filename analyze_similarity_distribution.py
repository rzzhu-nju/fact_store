import numpy as np
import pandas as pd
import json
import os
import sys
import requests
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add project root to path
sys.path.append(os.getcwd())

# Import components from generation_no_memory
from environment.generation_no_memory import BatchFactStoreAgent, ModelEngineV1
from environment.reward_utils import calculate_final_answer_reward

# --- Configuration ---
DATASET_PATH = "/data1/rzzhu/my_fact_store_project/data/nq_hotpotqa_train_autorefine/train.parquet"
NUM_SAMPLES = 50 
RETRIEVER_URL = "http://210.28.135.113:8085/search"

# --- Components (保持不变) ---

class RealHTTPRetriever:
    def __init__(self, url):
        self.url = url
    def search(self, queries, num=3):
        try:
            # 即使是单条，接口通常也期望列表格式
            payload = {"queries": queries, "topk": num}
            session = requests.Session()
            session.trust_env = False
            resp = session.post(self.url, json=payload, timeout=30) 
            resp.raise_for_status()
            data = resp.json()
            batch_results = []
            for res_str in data['result']:
                 if isinstance(res_str, str):
                     batch_results.append([{'contents': res_str}])
                 else:
                     batch_results.append([{'contents': str(res_str)}])
            return batch_results, data.get('raw_scores', None)
        except Exception as e:
            print(f"Search failed: {e}")
            return [[{'contents': ''}] for _ in queries], None

def search_func(retriever, queries, top_k=3):
    if not queries:
        return []
    results_batch, _ = retriever.search(queries, num=top_k)
    
    batch_result_strs = []
    for results in results_batch:
        if results and 'contents' in results[0]:
             batch_result_strs.append(results[0]['contents'])
        else:
             batch_result_strs.append("")
    return batch_result_strs

# --- Main Logic (修改后：完全串行) ---

def analyze_real_model():
    print(">>> Starting Real Model Analysis (Sequential Mode)...")
    
    # 1. Load Data
    df = pd.read_parquet(DATASET_PATH)
    # df_sample = df.sample(NUM_SAMPLES, random_state=42)
    df_sample = df.iloc[:NUM_SAMPLES] # 取前N个
    print(f"Sampled {len(df_sample)} items.")

    # 2. Load Engine
    engine = ModelEngineV1()
    http_retriever = RealHTTPRetriever(RETRIEVER_URL)
    
    # 3. Prepare Agents
    agents = []
    for idx, row in df_sample.iterrows():
        meta = row['meta_info']
        question = meta['question']
        gold_answer = meta['answer']
        gold_facts = meta['gold_facts']
        if hasattr(gold_facts, 'tolist'): gold_facts = gold_facts.tolist()
        
        if not gold_facts: continue
        
        agent = BatchFactStoreAgent(
            generator=engine, 
            retriever=search_func, # 这里传入函数引用
            reward_encoder=engine.reward_encoder,
            question=question, 
            answer=gold_answer, 
            gold_facts=gold_facts
        )
        agents.append(agent)
        
    print(f"Initialized {len(agents)} agents.")
    
    # 4. Run Sequential Inference
    # 这里不再使用 while any(...) 的批处理逻辑
    # 而是使用双层循环：外层遍历Agent，内层遍历Step
    
    print("\nRunning Inference Loop...")
    
    for i, agent in enumerate(tqdm(agents, desc="Processing Agents")):
        
        # 防止死循环的步数限制
        max_steps = 15 
        step_cnt = 0
        
        while not agent.is_finished and step_cnt < max_steps:
            step_cnt += 1
            
            # --- 分支 A: 生成 (Generate) ---
            if agent.next_action == "generate":
                # 1. 准备 Context
                agent.step() 
                prompt = agent.action_data # 此时 action_data 是 prompt 字符串
                
                # 2. 调用模型 (Batch Size = 1)
                # 注意：虽然是单条，但通常 generate 接口期望输入是列表
                responses = engine.generate([prompt], max_tokens=512)
                response = responses[0] # 取出唯一的那个结果
                
                # 3. 解析结果并更新
                agent.next_action = "parse"
                agent.step(response)
            
            # --- 分支 B: 搜索 (Search) ---
            elif agent.next_action == "search":
                query = agent.action_data # 此时 action_data 是 query 字符串
                
                # 1. 调用搜索 (Batch Size = 1)
                search_results = search_func(http_retriever, [query])
                result_str = search_results[0]
                
                # 2. 更新 Agent 状态
                agent.current_observation_str = result_str
                
                # 记录日志
                agent.logs.append({
                    "step": agent.steps_taken,
                    "type": "search_result",
                    "query": query,
                    "result_snippet": result_str[:200] + "..."
                })
                agent.history_summary.append(f"- Search executed: {query}")
                
                # 搜完之后，下一步通常回到 generate 继续推理
                agent.next_action = "generate"
    
    # 5. Analyze Results (后续分析逻辑保持不变)
    correct_sims = []
    incorrect_sims = []
    
    print("\nAnalyzing Results...")
    for agent in agents:
        gold_answers = agent.answer if isinstance(agent.answer, list) else [agent.answer]
        final_ans = agent.final_answer if agent.final_answer else ""
        
        # 判断正误
        is_correct = False
        score = calculate_final_answer_reward(final_ans, gold_answers[0], engine.reward_encoder)
        if score > 0.8: 
            is_correct = True
            
        # 收集三元组相似度
        agent_sims = []
        for log in agent.logs:
            if log.get('type') == 'reward':
                sim = log['content'].get('max_similarity', 0.0)
                agent_sims.append(sim)
        
        if is_correct:
            correct_sims.extend(agent_sims)
        else:
            incorrect_sims.extend(agent_sims)

    # 6. Plot / Print Stats
    def print_stats(name, data):
        if not data:
            print(f"{name}: No data.")
            return
        d = np.array(data)
        print(f"{name}: Mean={np.mean(d):.4f}, Std={np.std(d):.4f}, Count={len(d)}")

    print_stats("Correct Traj Triplets", correct_sims)
    print_stats("Incorrect Traj Triplets", incorrect_sims)

    plt.figure(figsize=(10, 6))
    if correct_sims:
        plt.hist(correct_sims, bins=20, range=(0,1), alpha=0.5, label='Correct Trajectories', density=True)
    if incorrect_sims:
        plt.hist(incorrect_sims, bins=20, range=(0,1), alpha=0.5, label='Incorrect Trajectories', density=True)
    
    plt.title('Triplet Similarity Distribution (Sequential Run)')
    plt.xlabel('Cosine Similarity to Gold Facts')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    out_file = "triplet_sim_distribution_seq.png"
    plt.savefig(out_file)
    print(f"\nPlot saved to {out_file}")

if __name__ == "__main__":
    analyze_real_model()