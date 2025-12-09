#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fact-Store Agent: Batch Processing Version (Version 1: No Summary Memory)

Changes from Original:
- Removed 'Working Memory' feature.
- Search results are presented directly as Observation.
- Uses external reward_utils.py for easy reward modification.
"""

import os
import re
import torch
import numpy as np
from typing import List, Dict, Tuple, Any
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from datasets import load_dataset
from tqdm import tqdm
import argparse
import ray
from omegaconf import OmegaConf
# from verl.trainer.main_ppo import main_task # Moved to usage

# Import external reward functions
try:
    from reward_utils import calculate_reward, calculate_final_answer_reward, calculate_retrieve_penalty
except ImportError:
    # Fallback if running from a different directory context
    import sys
    sys.path.append(os.path.dirname(__file__))
    from reward_utils import calculate_reward, calculate_final_answer_reward, calculate_retrieve_penalty

# ================= Configuration =================
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

# --- Mock imports for simple_retrieval ---
try:
    from simple_retrieval import SimpleDenseRetriever, Encoder
except ImportError:
    print("[Warning] 'simple_retrieval' module not found. Using mock classes.")
    class SimpleDenseRetriever:
        def __init__(self, **kwargs): pass
        def search(self, queries, num=3):
            return [[{'contents': f'Title: Dummy Doc\nContent: result for {q}'} for _ in range(num)] for q in queries], None
            
    class Encoder:
        def __init__(self, **kwargs): pass
        def encode(self, texts, **kwargs):
            return np.random.rand(len(texts), 768)

def search(retriever, queries: List[str], top_k=3) -> List[str]:
    if not queries:
        return []
    results_batch, _ = retriever.search(queries, num=top_k)
    
    batch_result_strs = []
    for results in results_batch:
        result_strs = []
        for i, doc in enumerate(results):
            contents = doc.get('contents', '')
            lines = contents.split('\n')
            if lines:
                title = lines[0].strip('"')
                text = '\n'.join(lines[1:])
                length = len(contents)
                result_strs.append(f"[Doc {i+1}] Title: {title} (Length: {length} chars)\n{text}")
            else:
                result_strs.append(f"[Doc {i+1}] {contents}")
        batch_result_strs.append("\n\n".join(result_strs))
            
    return batch_result_strs

class RealHTTPRetriever:
    def __init__(self, url: str, timeout: int = 30):
        import requests
        self.url = url
        self.timeout = timeout
        self._session = requests.Session()
        self._session.trust_env = False
    def search(self, queries, num=3):
        try:
            payload = {"queries": queries, "topk": num}
            resp = self._session.post(self.url, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            batch_results = []
            for res_str in data.get('result', []):
                batch_results.append([{'contents': res_str}])
            return batch_results, data.get('raw_scores', None)
        except Exception as e:
            print(f"Search failed: {e}")
            return [[{'contents': ''}] for _ in queries], None

# ================= Model Engine =================

class ModelEngineV1:
    def __init__(self):
        self.llm_path = "/data1/shares/Qwen2.5-7B-Instruct"
        
        print(">>> Loading LLM...")
        self.tokenizer_llm = AutoTokenizer.from_pretrained(self.llm_path, trust_remote_code=True)
        self.model_llm = AutoModelForCausalLM.from_pretrained(
            self.llm_path, trust_remote_code=True, torch_dtype=torch.float16, use_flash_attention_2=False
        ).eval().cuda()
        
        self.tokenizer_llm.padding_side = "left"
        if self.tokenizer_llm.pad_token is None:
            self.tokenizer_llm.pad_token = self.tokenizer_llm.eos_token
        
        print(">>> Initializing Global Retriever...")
        self.retriever = SimpleDenseRetriever(
            model_name="e5",
            model_path="/data1/shares/e5-base-v2",
            pooling_method="mean",
            max_length=512,
            use_fp16=True,
            index_path="/data1/rzzhu/wiki-2018/e5_Flat.index",
            corpus_path="/data1/rzzhu/wiki-2018/wiki-18.jsonl",
            topk=3,
            faiss_gpu=False
        )
        
        print(">>> Loading Embedding Model...")
        self.reward_encoder = Encoder(
            model_name="e5",
            model_path="/data1/shares/e5-base-v2",
            pooling_method="mean",
            max_length=512,
            use_fp16=True
        )

    def generate(self, messages_batch: List[List[Dict]], max_tokens=512) -> List[str]:
        if not messages_batch:
            return []

        tokenized_inputs = [
            self.tokenizer_llm.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors='pt'
            ) for messages in messages_batch
        ]

        max_len = max(t.shape[-1] for t in tokenized_inputs)
        
        input_ids_list = []
        attention_mask_list = []
        
        for t in tokenized_inputs:
            t = t.squeeze(0)
            pad_len = max_len - t.shape[0]
            if pad_len > 0:
                pad_tensor = torch.full((pad_len,), self.tokenizer_llm.pad_token_id, dtype=t.dtype, device=t.device)
                padded_t = torch.cat([pad_tensor, t], dim=0)
                mask = torch.cat([torch.zeros(pad_len, dtype=torch.long, device=t.device), torch.ones(t.shape[0], dtype=torch.long, device=t.device)], dim=0)
            else:
                padded_t = t
                mask = torch.ones(t.shape[0], dtype=torch.long, device=t.device)
            
            input_ids_list.append(padded_t)
            attention_mask_list.append(mask)

        input_ids = torch.stack(input_ids_list).to('cuda')
        attention_mask = torch.stack(attention_mask_list).to('cuda')

        with torch.no_grad():
            out = self.model_llm.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.8,
                top_k=20,
                pad_token_id=self.tokenizer_llm.pad_token_id
            )
        
        decoded_outputs = []
        for i, full_output in enumerate(out):
            gen_ids = full_output[input_ids.shape[-1]:]
            decoded_outputs.append(self.tokenizer_llm.decode(gen_ids, skip_special_tokens=True))
            
        return decoded_outputs

# ================= Core Agent Logic =================

class BatchFactStoreAgent:
    def __init__(self, generator, retriever, reward_encoder, question: str, answer: str, gold_facts: List[str] = None):
        self.generator = generator
        self.retriever = retriever
        self.reward_encoder = reward_encoder
        self.question = question
        self.answer = answer 
        self.gold_facts = gold_facts # List of golden triples (strings)
        
        self.visible_facts = [] 
        self.evidence_db = {}
        self.history_summary = [] 
        self.current_observation_str = None 
        
        # === REMOVED: Memory for Summaries ===
        # self.working_memory = [] 
        
        self.searched_queries = set()
        self.search_history = []
        self.covered_gold_indices = set()
        self.logs = [] # For saving full traces
        
        self.is_finished = False
        self.final_answer = None
        self.next_action = "generate"
        self.action_data = None
        self.steps_taken = 0
        # === Config Update: Increased Max Steps ===
        self.MAX_STEPS = 12 

    def reconstruct_training_sample(self, tokenizer) -> Tuple[List[int], List[int], List[float], List[float]]:
        """
        Reconstructs the full training sequence with masking, dense rewards, and log probabilities.
        Returns: input_ids, loss_mask, reward_tensor, log_probs_tensor
        """
        # Start with the initial prompt
        initial_prompt = self.get_system_prompt() + f"\n\nTask: {self.question}\n\nInstruction: Fact-Store is empty. Please think and <search>."
        
        full_input_ids = tokenizer.encode(initial_prompt, add_special_tokens=True)
        full_loss_mask = [0] * len(full_input_ids)
        full_rewards = [0.0] * len(full_input_ids)
        # Log probs for prompt are 0 (masked anyway)
        full_log_probs = [0.0] * len(full_input_ids)
        
        for log in self.logs:
            if log.get('role') == 'assistant':
                # Model generation
                content = log['content']
                tokens = tokenizer.encode(content, add_special_tokens=False)
                full_input_ids.extend(tokens)
                full_loss_mask.extend([1] * len(tokens))
                
                step_rewards = [0.0] * len(tokens)
                full_rewards.extend(step_rewards)
                
                # Append Log Probs
                # If 'log_probs' is available in log, use it. Otherwise 0.0 (and hope for recompute or ignore)
                # log['log_probs'] should be a list of floats matching 'tokens' length
                if 'log_probs' in log and len(log['log_probs']) == len(tokens):
                    full_log_probs.extend(log['log_probs'])
                else:
                    # Fallback: fill with 0.0. 
                    # Note: If this happens in PPO, it might be problematic unless recomputed.
                    full_log_probs.extend([0.0] * len(tokens))
                
            elif log.get('type') == 'search_result':
                # Observation
                content = f"\n\n=== Observation ===\n{log['result_snippet']}\n"
                tokens = tokenizer.encode(content, add_special_tokens=False)
                full_input_ids.extend(tokens)
                full_loss_mask.extend([0] * len(tokens))
                full_rewards.extend([0.0] * len(tokens))
                full_log_probs.extend([0.0] * len(tokens))
                
            elif log.get('type') == 'reward':
                # Reward entry - attribute to last token
                reward_val = log['content']['reward']
                if full_rewards:
                    full_rewards[-1] += reward_val
            
        return full_input_ids, full_loss_mask, full_rewards, full_log_probs


    def get_system_prompt(self) -> str:
        facts_str = "\n".join([f"{i+1}. <{f}>" for i, f in enumerate(self.visible_facts)])
        if not facts_str: facts_str = "(Empty)"

        return (
            "You are a verifiable reasoning agent based on a **Fact-Store** architecture.\n\n"
            "**INPUT STRUCTURE**:\n"
            "The user will provide a message containing the following sections:\n"
            "1. **Task**: The question you need to answer.\n"
            "2. **History**: A summary of your past actions and their results.\n"
            "3. **Executed Searches**: Queries you have already performed (to avoid duplication).\n"
            "4. **Observation**: The text documents retrieved by your LAST <search> action. **NOTE**: If your last action was NOT <search>, this section will be None/Empty.\n"
            "5. **Current Fact-Store**: The list of all facts you have asserted so far.\n\n"
            "**CORE RULES**:\n"
            "1. **Reasoning First**: You must conduct reasoning inside <think>...</think> before generating any action.\n"
            "2. **No Hallucination**: You cannot answer from your internal training memory. You must <search>, read the Observation, and <assert> facts.\n"
            "3. **Fact-Dependency**: Your final <answer> must be strictly derived from the **Current Fact-Store**.\n"
            "4. **Format requirement**: You must enclose your action selection within <>.\n"
            "5. **Avoid Duplicate Searches**: Review the Executed Searches block. Do not repeat executed queries; craft new searches that are clearly distinct.\n\n"
            "**QUERY REFORMULATION STRATEGIES**:\n"
            "If your previous search failed (marked as 'NO INFO FOUND' in history), you MUST change your strategy:\n"
            "1. **Broaden**: Remove specific constraints (e.g., 'Director of movie X' -> 'Movie X cast').\n"
            "2. **Decompose**: Split complex questions into smaller entities.\n"
            "3. **Pivot**: Search for related entities mentioned in the question.\n"
            "4. **Do NOT** generate a query semantically identical to previous ones.\n\n"
            "**AVAILABLE ACTIONS**:\n"
            "- <search>keywords</search>: Acquire info.\n"
            "- <assert>Subject, Relation, Object</assert>: Extract facts from observation (e.g. <assert>Obama, born in, Hawaii</assert>). DO NOT include labels like 'Subject:'. Just use commas. You can assert multiple facts in one turn.\n"
            "- <retrieve>Subject, Relation, Object</retrieve>: Check sources.\n"
            "- <answer>Final Answer</answer>: Use Fact-Store to answer. **Make the answer extremely concise (e.g., a single word, entity name, or short phrase). Do NOT use full sentences.**\n\n"
            "=== Current Fact-Store ===\n"
            f"{facts_str}\n"
        )

    def build_context(self) -> List[Dict]:
        messages = [{"role": "system", "content": self.get_system_prompt()}]
        
        user_content_parts = []
        user_content_parts.append(f"Task: {self.question}")
        
        if self.history_summary:
            user_content_parts.append("History:\n" + "\n".join(self.history_summary))
        
        if self.search_history:
            hist = "\n".join([f"- {q}" for q in self.search_history])
            user_content_parts.append(f"=== Executed Searches ===\n{hist}")
        
        # Add Current Fact-Store explicitly in user message as well to be safe, or rely on system prompt update
        # Since get_system_prompt() is called every time build_context is called (it's dynamic), 
        # the system prompt already contains the latest facts.
        
        if self.current_observation_str:
            obs_block = f"=== Observation ===\n{self.current_observation_str}\n"
            user_content_parts.append(obs_block)
            user_content_parts.append("Instruction: Analyze the observation inside <think>, then extract facts using <assert>.")
        else:
            msg = "Instruction: Fact-Store is empty. Please think and <search>." if not self.visible_facts else "Instruction: Please think and verify if you can <answer> or need to <search> more."
            user_content_parts.append(msg)
            
        # Append "Let's think" or similar if we want to force CoT?
        # But we want to let the model do it.
            
        full_user_content = "\n\n".join(user_content_parts)
        messages.append({"role": "user", "content": full_user_content})
        
        return messages

    def step(self, response: str = None):
        if self.steps_taken >= self.MAX_STEPS:
            self.is_finished = True
            self.final_answer = "Max steps reached."
            return

        if self.next_action == "generate":
            context = self.build_context()
            self.action_data = context
            self.steps_taken += 1
            # Log the prompt (User message)
            # Only append if not already last message (to avoid dupes if re-running)
            # Actually, `build_context` creates new messages.
            # In RL, we usually append.
            self.logs.append({"step": self.steps_taken, "role": "user", "content": context[-1]['content']})
            return
            
        elif self.next_action == "parse":
            # Log the response
            if response is None: response = "" # Safety check
            self.logs.append({"step": self.steps_taken, "role": "assistant", "content": response})
            self.parse_and_execute(response)

    def parse_and_execute(self, response: str):
        def find_last_pos(pattern, text):
            matches = list(re.finditer(pattern, text, re.DOTALL))
            if not matches: return -1
            return matches[-1].end()

        actions = {
            'search': find_last_pos(r'</search>', response),
            'assert': find_last_pos(r'</assert>', response),
            'retrieve': find_last_pos(r'</retrieve>', response),
            'answer': find_last_pos(r'</answer>', response),
        }

        valid_actions = {k: v for k, v in actions.items() if v != -1}
        
        if not valid_actions:
            if re.search(r"<search>(.*)", response): valid_actions['search'] = 1
            elif re.search(r"<answer>(.*)", response): valid_actions['answer'] = 1
            
            if not valid_actions:
                self.history_summary.append("- Failed to parse action. Please use correct format.")
                self.next_action = "generate"
                return

        last_action = max(valid_actions, key=valid_actions.get)

        if last_action == 'search':
            search_matches = re.findall(r"[<=]?search>(.*?)</search>", response, re.DOTALL)
            if not search_matches: search_matches = re.findall(r"[<=]?search>(.*)", response, re.DOTALL)
                
            if search_matches:
                query = search_matches[-1].strip()
                if query in self.searched_queries:
                    self.history_summary.append(f"- Blocked duplicate search: {query}")
                    self.next_action = "generate"
                else:
                    self.searched_queries.add(query)
                    self.search_history.append(query)
                    self.next_action = "search"
                    self.action_data = query 
                return

        elif last_action == 'assert':
            assert_matches = re.findall(r"<assert>(.*?)</assert>", response, re.DOTALL)
            new_facts_count = 0
            context_for_reward = self.current_observation_str if self.current_observation_str else ""


            
            for content in assert_matches:
                parts = [p.strip() for p in content.split(',', 2)]
                if len(parts) == 3:
                    triple = (parts[0], parts[1], parts[2])
                    
                    # === USE EXTERNAL REWARD FUNCTION ===
                    reward_info = calculate_reward(
                        triple, 
                        context_for_reward, 
                        embedding_model=self.reward_encoder,
                        gold_facts=self.gold_facts
                    )
                    
                    fact_str = f"{parts[0]}, {parts[1]}, {parts[2]}"
                    if fact_str not in self.visible_facts:
                        self.visible_facts.append(fact_str)
                        new_facts_count += 1
                        self.evidence_db[fact_str] = context_for_reward
                        
                        # Log the reward
                        self.logs.append({
                            "step": self.steps_taken,
                            "type": "reward",
                            "content": reward_info
                        })
            
            if new_facts_count > 0:
                self.history_summary.append(f"- Asserted {new_facts_count} facts")
            else:
                self.history_summary.append("- No new facts asserted.")
            self.current_observation_str = None
            self.next_action = "generate"
            return

        elif last_action == 'retrieve':
            retrieve_matches = re.findall(r"<retrieve>(.*?)</retrieve>", response, re.DOTALL)
            if not retrieve_matches: 
                 self.history_summary.append("- Retrieve action failed to parse.")
                 self.next_action = "generate"
                 return
            
            fact_triple_str = retrieve_matches[-1].strip()
            # Clean up the triple string to match how it's stored in evidence_db
            # evidence_db stores keys as "Subject, Relation, Object"
            # We assume the model outputs exactly that, or we might need to normalize spaces.
            
            # Simple normalization: split by comma and rejoin
            parts = [p.strip() for p in fact_triple_str.split(',', 2)]
            if len(parts) == 3:
                normalized_key = f"{parts[0]}, {parts[1]}, {parts[2]}"
                source_doc = self.evidence_db.get(normalized_key, "No source document found for this fact in Evidence DB.")
                
                self.current_observation_str = f"Source Document for <{normalized_key}>:\n{source_doc}"
                self.history_summary.append(f"- Retrieved source for: {normalized_key}")
                penalty_info = calculate_retrieve_penalty()
                self.logs.append({
                    "step": self.steps_taken,
                    "type": "reward",
                    "content": penalty_info
                })
            else:
                 self.history_summary.append(f"- Retrieve failed: Invalid format '{fact_triple_str}'")
                 self.current_observation_str = None

            self.next_action = "generate"
            return

        elif last_action == 'answer':
            answer_matches = re.findall(r"[<=]?answer>(.*?)</answe", response, re.DOTALL)
            if not answer_matches: answer_matches = re.findall(r"[<=]?answer>(.*)", response, re.DOTALL)
            if answer_matches:
                ans = answer_matches[-1].strip()
                self.final_answer = ans
                self.is_finished = True
                self.next_action = "finished"
                return

        self.next_action = "generate"


# ================= Data Loading =================

def load_bamboogle_dataset():
    print(">>> Loading RUC-NLPIR/FlashRAG_datasets bamboogle subset...")
    try:
        dataset = load_dataset("RUC-NLPIR/FlashRAG_datasets", "bamboogle", split="test", trust_remote_code=True)
    except Exception as e:
        print(f"Error loading dataset: {e}. Creating dummy dataset for testing.")
        dataset = [
            {
                "question": "Who is the author of The Three-Body Problem?", 
                "golden_answers": ["Liu Cixin"],
                "gold_facts": ["The Three-Body Problem, author, Liu Cixin", "Liu Cixin, wrote, The Three-Body Problem"]
            },
            {
                "question": "What is the capital of France?", 
                "golden_answers": ["Paris"],
                "gold_facts": ["France, capital, Paris", "Paris, is capital of, France"]
            },
        ] * 5 
    return dataset

# ================= Main Orchestrator =================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["standalone", "verl_train"], default="standalone")
    parser.add_argument("--data_dir", default="data/nq_hotpotqa_train_autorefine")
    parser.add_argument("--base_model", default="Qwen/Qwen2.5-3B")
    parser.add_argument("--experiment_name", default="mygen-autorefine-qwen2.5-3b")
    parser.add_argument("--retriever_url", default="http://127.0.0.1:8085/search")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--n_gpus", type=int, default=8)
    parser.add_argument("--nnodes", type=int, default=1)
    parser.add_argument("--total_steps", type=int, default=300)
    args = parser.parse_args()

    if args.mode == "verl_train":
        from verl.trainer.main_ppo import main_task
        config = OmegaConf.load(os.path.join(os.path.dirname(__file__), "verl/trainer/config/grpo_trainer.yaml"))
        print("Verl training mode not fully adapted in this script. Please use standalone for testing logic.")
    else:
        dataset = load_bamboogle_dataset()
        engine = ModelEngineV1()
        if args.retriever_url:
            try:
                engine.retriever = RealHTTPRetriever(args.retriever_url)
                print(f"Using HTTP Retriever at {args.retriever_url}")
            except Exception as e:
                print(f"Failed to initialize HTTP Retriever: {e}. Falling back to local retriever.")
        BATCH_SIZE = 8
        all_results = []
        total_len = len(dataset)
        print(f"Total questions to process: {total_len}")
        with tqdm(total=total_len, desc="Processing Queries", unit="q") as pbar:
            for i in range(0, total_len, BATCH_SIZE):
                if isinstance(dataset, list):
                    batch_data = dataset[i : i + BATCH_SIZE]
                else:
                    batch_data = dataset.select(range(i, min(i + BATCH_SIZE, total_len)))
                current_agents = []
                for item in batch_data:
                    q = item["question"]
                    a = item.get("golden_answers", item.get("answer", []))
                    gf = item.get("gold_facts", [])
                    current_agents.append(BatchFactStoreAgent(engine, search, engine.reward_encoder, q, a, gold_facts=gf))
                active_agents = current_agents
                while any(not agent.is_finished for agent in active_agents):
                    generate_batch_indices = []
                    prompts = []
                    for idx, agent in enumerate(active_agents):
                        if not agent.is_finished and agent.next_action == "generate":
                            agent.step()
                            prompts.append(agent.action_data)
                            generate_batch_indices.append(idx)
                    if prompts:
                        responses = engine.generate(prompts)
                        for local_idx, response in zip(generate_batch_indices, responses):
                            agent = active_agents[local_idx]
                            agent.next_action = "parse"
                            agent.step(response)
                    search_batch_indices = []
                    queries = []
                    for idx, agent in enumerate(active_agents):
                        if not agent.is_finished and agent.next_action == "search":
                            queries.append(agent.action_data)
                            search_batch_indices.append(idx)
                    if queries:
                        search_results = search(engine.retriever, queries)
                        for local_idx, result_str in zip(search_batch_indices, search_results):
                            agent = active_agents[local_idx]
                            agent.current_observation_str = result_str
                            agent.logs.append({
                                "step": agent.steps_taken,
                                "type": "search_result",
                                "query": agent.action_data,
                                "result_snippet": result_str[:200] + "..."
                            })
                            agent.history_summary.append(f"- Search executed: {agent.action_data}")
                            agent.next_action = "generate"
                            
                    if all(agent.is_finished for agent in active_agents):
                        break
                for agent in current_agents:
                    final_reward = calculate_final_answer_reward(
                        agent.final_answer or "",
                        agent.answer[0] if isinstance(agent.answer, list) else agent.answer,
                        embedding_model=engine.reward_encoder
                    )
                    input_ids, loss_mask, _, _ = agent.reconstruct_training_sample(engine.tokenizer_llm)
                    result_entry = {
                        "question": agent.question,
                        "final_answer": agent.final_answer,
                        "golden": agent.answer,
                        "em": final_reward,
                        "trace": agent.logs,
                        "input_ids": input_ids,
                        "loss_mask": loss_mask
                    }
                    all_results.append(result_entry)
                pbar.update(len(batch_data))
                with open("batch_results_no_memory.json", "w") as f:
                    json.dump(all_results, f, indent=2)
        total_em = sum(r["em"] for r in all_results)
        avg_em = total_em / len(all_results) if all_results else 0
        print(f"\nProcessing Complete. Overall EM/Reward: {avg_em:.4f}")
