#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fact-Store Agent: Batch Processing Version (Optimized + Memory Experiment)

Updates:
1. Max Steps increased to 12.
2. Added detailed process logging (saved in 'trace').
3. Added 'Working Memory' feature: Automatically summarizes search results and adds to context.
4. Added Progress Bar (tqdm) for query tracking.
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
from verl.trainer.main_ppo import main_task

# ================= Configuration =================
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

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

    def calculate_reward(self, triple: Tuple[str,str,str], current_context: str) -> Dict:
        return {"reward": 0.5, "type": "HIT_GOLD", "gold_idx": 1, "score": 1, "desc": "Core Fact Hit"}

# ================= Core Agent Logic =================

class BatchFactStoreAgent:
    def __init__(self, engine: ModelEngineV1, search_fn, question: str, answer: str):
        self.engine = engine
        self.search_fn = search_fn
        self.question = question
        self.answer = answer 
        
        self.visible_facts = [] 
        self.evidence_db = {}
        self.history_summary = [] 
        self.current_observation_str = None 
        
        # === New: Memory for Summaries ===
        self.working_memory = [] 
        
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

    def get_system_prompt(self) -> str:
        facts_str = "\n".join([f"{i+1}. <{f}>" for i, f in enumerate(self.visible_facts)])
        if not facts_str: facts_str = "(Empty)"
        
        memory_str = "\n".join([f"- {m}" for m in self.working_memory])
        if not memory_str: memory_str = "(Empty)"
        
        return (
            "You are a verifiable reasoning agent based on a **Fact-Store** architecture.\n\n"
            
            "**INPUT STRUCTURE**:\n"
            "1. **Task**: The question you need to answer.\n"
            "2. **History**: A summary of your past actions.\n"
            "3. **Working Memory**: Summarized key information from previous searches. Use this to guide your reasoning.\n"
            "4. **Observation**: Raw documents from your LAST <search>.\n\n"
            
            "**CORE RULES**:\n"
            "1. **Reasoning First**: Conduct reasoning inside <think>...</think>.\n"
            "2. **Fact-Dependency**: Final <answer> must be derived from Fact-Store and Working Memory.\n"
            "3. **Format**: Use <> for actions.\n\n"

            "**AVAILABLE ACTIONS**:\n"
            "- <search>keywords</search>: Acquire info.\n"
            "- <assert>Subject, Relation, Object</assert>: Extract facts from Observation.\n"
            "- <retrieve>Subject, Relation, Object</retrieve>: Check sources.\n"
            "- <answer>Final Answer</answer>: Answer concisely.\n\n"
            
            f"=== Current Fact-Store ===\n{facts_str}\n\n"
            f"=== Working Memory ===\n{memory_str}\n\n"
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
        
        if self.current_observation_str:
            obs_block = f"=== Observation ===\n{self.current_observation_str}\n"
            user_content_parts.append(obs_block)
            user_content_parts.append("Instruction: Analyze the observation inside <think>, then extract facts using <assert>.")
        else:
            msg = "Instruction: Fact-Store is empty. Please think and <search>." if not self.visible_facts else "Instruction: Please think and verify if you can <answer> or need to <search> more."
            user_content_parts.append(msg)
            
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
            self.logs.append({"step": self.steps_taken, "role": "user", "content": context[-1]['content']})
            return
            
        elif self.next_action == "parse":
            # Log the response
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
                    _ = self.engine.calculate_reward(triple, context_for_reward)
                    fact_str = f"{parts[0]}, {parts[1]}, {parts[2]}"
                    if fact_str not in self.visible_facts:
                        self.visible_facts.append(fact_str)
                        new_facts_count += 1
                        self.evidence_db[fact_str] = context_for_reward
            
            if new_facts_count > 0:
                self.history_summary.append(f"- Asserted {new_facts_count} facts")
            else:
                self.history_summary.append("- No new facts asserted.")
            self.current_observation_str = None
            self.next_action = "generate"
            return

        elif last_action == 'retrieve':
            self.history_summary.append("- Retrieve action executed.")
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
            {"question": "Who is the author of The Three-Body Problem?", "golden_answers": ["Liu Cixin"]},
            {"question": "What is the capital of France?", "golden_answers": ["Paris"]},
        ] * 5 
    return dataset

# ================= Main Orchestrator =================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["standalone", "verl_train"], default="standalone")
    parser.add_argument("--data_dir", default="data/nq_hotpotqa_train_autorefine")
    parser.add_argument("--base_model", default="Qwen/Qwen2.5-3B")
    parser.add_argument("--experiment_name", default="mygen-autorefine-qwen2.5-3b")
    parser.add_argument("--retriever_url", default="http://127.0.0.1:8000/retrieve")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--n_gpus", type=int, default=8)
    parser.add_argument("--nnodes", type=int, default=1)
    parser.add_argument("--total_steps", type=int, default=300)
    args = parser.parse_args()

    if args.mode == "verl_train":
        config = OmegaConf.load(os.path.join(os.path.dirname(__file__), "verl/trainer/config/grpo_trainer.yaml"))
        config.reward_model.reward_style = "F1"
        config.data.train_files = os.path.join(args.data_dir, "train.parquet")
        config.data.val_files = os.path.join(args.data_dir, "valid_500.parquet")
        config.data.train_data_num = None
        config.data.val_data_num = None
        config.data.train_batch_size = 256
        config.data.val_batch_size = 256
        config.data.max_prompt_length = 6656
        config.data.max_response_length = 512
        config.data.max_start_length = 2048
        config.data.max_obs_length = 512
        config.max_turns = 5
        config.data.shuffle_train_dataloader = True
        config.algorithm.adv_estimator = "grpo"
        config.algorithm.filter_groups.enable = False
        config.actor_rollout_ref.model.path = args.base_model
        config.actor_rollout_ref.model.enable_gradient_checkpointing = True
        config.actor_rollout_ref.model.use_remove_padding = True
        config.actor_rollout_ref.actor.refine_lambda = -1
        config.actor_rollout_ref.actor.refine_score = 0.1
        config.actor_rollout_ref.actor.format_score = 0.0
        config.actor_rollout_ref.actor.optim.lr = 1e-6
        config.actor_rollout_ref.actor.use_kl_loss = True
        config.actor_rollout_ref.actor.ppo_mini_batch_size = 256
        config.actor_rollout_ref.actor.ppo_micro_batch_size = 64
        config.actor_rollout_ref.actor.fsdp_config.param_offload = True
        config.actor_rollout_ref.actor.fsdp_config.grad_offload = True
        config.actor_rollout_ref.actor.fsdp_config.optimizer_offload = True
        config.actor_rollout_ref.rollout.log_prob_micro_batch_size = 128
        config.actor_rollout_ref.rollout.tensor_model_parallel_size = 1
        config.actor_rollout_ref.rollout.name = "vllm"
        config.actor_rollout_ref.rollout.gpu_memory_utilization = 0.6
        config.actor_rollout_ref.ref.log_prob_micro_batch_size = 128
        config.actor_rollout_ref.ref.fsdp_config.param_offload = True
        config.actor_rollout_ref.actor.kl_loss_coef = 0.001
        config.actor_rollout_ref.actor.kl_loss_type = "low_var_kl"
        config.algorithm.no_think_rl = False
        config.actor_rollout_ref.rollout.n_agent = 5
        config.actor_rollout_ref.rollout.temperature = 1
        config.actor_rollout_ref.actor.state_masking = True
        config.trainer.logger = ["wandb"]
        config.trainer.val_only = False
        config.trainer.val_before_train = True
        config.trainer.default_hdfs_dir = None
        config.trainer.n_gpus_per_node = args.n_gpus
        config.trainer.nnodes = args.nnodes
        config.trainer.save_freq = 300
        config.trainer.test_freq = 20
        config.trainer.project_name = "AutoRefine"
        config.trainer.experiment_name = args.experiment_name
        config.trainer.total_epochs = 15
        config.trainer.total_training_steps = args.total_steps
        config.trainer.default_local_dir = os.path.join("verl_checkpoints", args.experiment_name)
        config.retriever.url = args.retriever_url
        config.retriever.topk = args.topk

        OmegaConf.resolve(config)
        if not ray.is_initialized():
            ray.init(runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN"}})
        ray.get(main_task.remote(config))
    else:
        dataset = load_bamboogle_dataset()
        engine = ModelEngineV1()
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
                    current_agents.append(BatchFactStoreAgent(engine, search, q, a))
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
                        summ_prompts = []
                        for local_idx, res_str in zip(search_batch_indices, search_results):
                            agent = active_agents[local_idx]
                            p = [
                                {"role": "system", "content": "You are a helpful assistant. Summarize the relevant information from the documents for the query."},
                                {"role": "user", "content": f"Query: {agent.action_data}\n\nDocuments:\n{res_str}\n\nInstruction: Provide a concise summary of facts relevant to the query. If no relevant info, say 'No relevant info'."}
                            ]
                            summ_prompts.append(p)
                        summaries = engine.generate(summ_prompts, max_tokens=256)
                        for local_idx, result_str, summary in zip(search_batch_indices, search_results, summaries):
                            agent = active_agents[local_idx]
                            agent.current_observation_str = result_str
                            clean_summary = summary.strip()
                            agent.working_memory.append(f"Query: {agent.action_data} -> {clean_summary}")
                            agent.logs.append({
                                "step": agent.steps_taken,
                                "type": "system_summary",
                                "query": agent.action_data,
                                "summary": clean_summary
                            })
                            agent.history_summary.append(f"- Search executed: {agent.action_data}")
                            agent.next_action = "generate"
                    if all(agent.is_finished for agent in active_agents):
                        break
                for agent in current_agents:
                    is_correct = False
                    golden = agent.answer
                    if isinstance(golden, list):
                        is_correct = any(g.lower() in (agent.final_answer or "").lower() for g in golden)
                    else:
                        is_correct = (golden.lower() in (agent.final_answer or "").lower())
                    result_entry = {
                        "question": agent.question,
                        "final_answer": agent.final_answer,
                        "golden": agent.answer,
                        "em": 1.0 if is_correct else 0.0,
                        "trace": agent.logs
                    }
                    all_results.append(result_entry)
                pbar.update(len(batch_data))
                with open("batch_results.json", "w") as f:
                    json.dump(all_results, f, indent=2)
        total_em = sum(r["em"] for r in all_results)
        avg_em = total_em / len(all_results) if all_results else 0
        print(f"\nProcessing Complete. Overall EM: {avg_em:.4f}")