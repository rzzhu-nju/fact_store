#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fact-Store Agent V3.0: HotpotQA Benchmark (English)

Updates:
1. Dataset: Loads real data from HuggingFace 'hotpot_qa' dataset.
2. Dynamic Environment: 'setup_task' allows switching contexts per question.
3. Language: All prompts and interactions are now in English.
4. Logic: Retains the V2.9 Embedding-based Threshold Reward logic.
"""

import os
import re
import torch
import numpy as np
from typing import List, Dict, Tuple, Any
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from datasets import load_dataset

# ================= Configuration =================
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "7")

# ================= Model Engine =================

class ModelEngine:
    def __init__(self):
        # Paths (Adjust to your local environment)
        self.llm_path = "/data1/shares/Qwen2.5-7B-Instruct"
        self.embedding_path = "/data1/shares/e5-base-v2" # Updated to E5
        
        print(">>> Loading Models...")
        self.tokenizer_llm = AutoTokenizer.from_pretrained(self.llm_path, trust_remote_code=True)
        self.model_llm = AutoModelForCausalLM.from_pretrained(
            self.llm_path, trust_remote_code=True, torch_dtype=torch.float16, use_flash_attention_2=False
        ).eval().cuda()
        
        # E5 Tokenizer and Model
        self.tokenizer_emb = AutoTokenizer.from_pretrained(self.embedding_path)
        self.model_emb = AutoModel.from_pretrained(self.embedding_path).eval().cuda()
        
        # Task specific data (initialized in setup_task)
        self.corpus = []
        self.doc_embs = None
        self.gold_embs = None
        
    def setup_task(self, corpus: List[Dict], gold_facts: List[str]):
        """
        Initialize environment for a specific HotpotQA question.
        """
        self.corpus = corpus
        print(f">>> Encoding Corpus ({len(corpus)} docs)...")
        # For E5, documents need 'passage: ' prefix
        self.doc_embs = self.encode_texts([d["contents"] for d in self.corpus], is_query=False)
        
        print(f">>> Encoding Gold Facts ({len(gold_facts)} sentences)...")
        # Gold facts are treated as passages for similarity comparison
        self.gold_embs = self.encode_texts(gold_facts, is_query=False)

    @torch.no_grad()
    def encode_texts(self, texts: List[str], is_query: bool) -> np.ndarray:
        if not texts: return np.array([])
        
        # E5 specific prefixes
        if is_query:
            texts = [f"query: {q}" for q in texts]
        else:
            texts = [f"passage: {t}" for t in texts]
            
        inputs = self.tokenizer_emb(texts, max_length=512, padding=True, truncation=True, return_tensors="pt")
        inputs = {k: v.cuda() for k, v in inputs.items()}
        outputs = self.model_emb(**inputs, return_dict=True)
        
        # E5 uses mean pooling or cls pooling depending on version. 
        # V2 usually works well with mean pooling, but let's check standard usage.
        # intfloat/e5-base-v2 recommends average pooling.
        last_hidden = outputs.last_hidden_state.masked_fill(~inputs['attention_mask'][..., None].bool(), 0.0)
        embeddings = last_hidden.sum(dim=1) / inputs['attention_mask'].sum(dim=1)[..., None]
        embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
        
        return embeddings.detach().cpu().numpy().astype(np.float32)

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
        Returns formatted string of top-k documents.
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
        V2.9 Logic: Threshold-based Reward with E5 Embeddings
        """
        t_text = f"{triple[0]} {triple[1]} {triple[2]}"
        # Triple is treated as a query/hypothesis to check against gold/context passages
        t_emb = self.encode_texts([t_text], is_query=True) 
        
        # 1. Similarity with Gold Facts (Supporting Sentences - encoded as passages)
        scores_gold = (t_emb @ self.gold_embs.T)[0]
        max_gold_idx = np.argmax(scores_gold) if len(scores_gold) > 0 else -1
        max_gold_score = scores_gold[max_gold_idx] if len(scores_gold) > 0 else 0.0
        
        # 2. Similarity with Current Context
        max_ctx_score = 0.0
        if current_context:
            # Simple sentence splitting
            sentences = re.split(r'[.!?\n]', current_context)
            sentences = [s.strip() for s in sentences if len(s) > 10]
            if sentences:
                # Context sentences encoded as passages
                ctx_embs = self.encode_texts(sentences, is_query=False)
                scores_ctx = (t_emb @ ctx_embs.T)[0]
                max_ctx_score = np.max(scores_ctx)

        # 3. Reward Logic (Thresholds might need tuning for E5 compared to BGE)
        # E5 scores are cosine similarities. 
        
        # Case A: High Match (> 0.82) -> HIT (Slightly adjusted for E5)
        if max_gold_score > 0.82:
            return {
                "reward": 0.5,
                "type": "HIT_GOLD",
                "gold_idx": max_gold_idx,
                "score": max_gold_score,
                "desc": "Core Fact Hit"
            }
            
        # Case B: Medium Match (0.75 - 0.82) -> WEAK MATCH
        elif 0.75 <= max_gold_score <= 0.82:
            return {
                "reward": 0.1,
                "type": "WEAK_MATCH",
                "gold_idx": -1,
                "score": max_gold_score,
                "desc": "Weak Match / Related"
            }
            
        # Case C: Low Match (< 0.75) -> MISS / DISTRACTOR
        else:
            if max_ctx_score > 0.82:
                return {
                    "reward": -0.5,
                    "type": "DISTRACTOR",
                    "gold_idx": -1,
                    "score": max_gold_score,
                    "desc": "Distractor (Irrelevant info)"
                }
            else:
                return {
                    "reward": -1,
                    "type": "HALLUCINATION",
                    "gold_idx": -1,
                    "score": max_gold_score,
                    "desc": "Hallucination"
                }

# ================= Core Agent Logic =================

class FactStoreAgent:
    def __init__(self, engine: ModelEngine):
        self.engine = engine
        self.visible_facts = [] 
        self.evidence_db = {}
        self.history_summary = [] 
        self.current_observation_str = None 
        self.searched_queries = set()
        self.search_history = []
        self.covered_gold_indices = set() 

    def get_system_prompt(self) -> str:
        facts_str = "\n".join([f"{i+1}. <{f}>" for i, f in enumerate(self.visible_facts)])
        if not facts_str: facts_str = "(Empty)"
        
        # Updated Search-R1 style Prompt with Fact-Store constraints
        return (
            "You are a verifiable reasoning agent based on a **Fact-Store** architecture.\n\n"
            
            "**INPUT STRUCTURE**:\n"
            "The user will provide a message containing the following sections:\n"
            "1. **Task**: The question you need to answer.\n"
            "2. **History**: A summary of your past actions and their results.\n"
            "3. **Executed Searches**: Queries you have already performed (to avoid duplication).\n"
            "4. **Observation**: The text documents retrieved by your LAST <search> action. "
            "**NOTE**: If your last action was NOT <search>, this section will be None/Empty.\n\n"
            
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
            "- <assert>Subject, Relation, Object</assert>: Extract facts from observation.\n"
            "- <retrieve>Subject, Relation, Object</retrieve>: Check sources.\n"
            "<answer>Final Answer</answer>: Use Fact-Store to answer. **Your answer MUST be a concise phrase or single word** to facilitate Exact Match (EM) evaluation.\n\n"
            
            f"=== Current Fact-Store ===\n{facts_str}\n\n"
        )

    def build_context(self, query: str) -> List[Dict]:
        """
        Constructs the context messages.
        Consolidates all user context (Task, History, Searches, Observation) into a SINGLE user message.
        """
        messages = [{"role": "system", "content": self.get_system_prompt()}]
        
        user_content_parts = []
        
        # 1. Task
        user_content_parts.append(f"Task: {query}")
        
        # 2. History
        if self.history_summary:
            user_content_parts.append("History:\n" + "\n".join(self.history_summary))
        
        # 3. Executed Searches
        if self.search_history:
            hist = "\n".join([f"- {q}" for q in self.search_history])
            sh_block = (
                f"=== Executed Searches ===\n{hist}\n"
                f"(Instruction: Do not repeat the above searches. Craft a new query if needed.)"
            )
            user_content_parts.append(sh_block)
        
        # 4. Observation & Instruction
        if self.current_observation_str:
            obs_block = (
                f"=== Observation ===\n"
                f"{self.current_observation_str}\n"
            )
            user_content_parts.append(obs_block)
            user_content_parts.append("Instruction: Analyze the observation inside <think>, then extract facts using <assert>.")
        else:
            # Observation is None (implicitly handled by omission or we could explicitly say "Observation: None")
            # Given instructions, we append the next step guidance.
            msg = "Instruction: Fact-Store is empty. Please think and <search>." if not self.visible_facts else "Instruction: Please think and verify if you can <answer> or need to <search> more."
            user_content_parts.append(msg)
            
        # Combine into a single user message
        full_user_content = "\n\n".join(user_content_parts)
        messages.append({"role": "user", "content": full_user_content})
        
        return messages

                
    def parse_and_execute(self, response: str) -> bool:
        # Print Thought Process for debugging
        # think_match = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
        # if think_match:
        #     print(f"\n[Thought Process]:\n{think_match.group(1).strip()}\n")

        # More robustly find the last occurrence of each action's end tag.
        def find_last_pos(pattern, text):
            matches = list(re.finditer(pattern, text, re.DOTALL))
            if not matches:
                return -1
            return matches[-1].end()

        actions = {
            'search': find_last_pos(r'</search>', response),
            'assert': find_last_pos(r'</assert>', response), # Asserts are expected to be well-formed.
            'retrieve': find_last_pos(r'</retrieve>', response),
            'answer': find_last_pos(r'</answer>', response),
        }

        valid_actions = {k: v for k, v in actions.items() if v != -1}
        if not valid_actions:
            print("   [Warning] Unable to parse action.")
            return True

        last_action = max(valid_actions, key=valid_actions.get)

        # Execute the last action
        if last_action == 'search':
            # Robustly find all matches and process the last one.
            search_matches = re.findall(r"[<=]?search>(.*?)</search>", response, re.DOTALL)
            if search_matches:
                query = search_matches[-1].strip()
                if query in self.searched_queries:
                    print(f"   [Blocked] Duplicate Search: {query}")
                else:
                    self.searched_queries.add(query)
                    self.search_history.append(query)
                    print(f"   [Action] Search: {query}")
                    
                    obs_str = self.engine.search(query)
                    self.current_observation_str = obs_str
                    
                    self.history_summary.append(f"- Search: {query}")
                    print(f"\n{'='*20} Retrieved Information {'='*20}")
                    print(obs_str)
                    print(f"{'='*62}\n")
                return True

        elif last_action == 'assert':
            # This part is already robust for multiple asserts.
            assert_matches = re.findall(r"<assert>(.*?)</assert>", response, re.DOTALL)
            assert_matches = [m.strip() for m in assert_matches if m.strip()]
            
            if assert_matches:
                print(f"   [Action] Processing {len(assert_matches)} Assertions...")
                step_reward = 0
                new_facts = []
                
                context_for_reward = self.current_observation_str if self.current_observation_str else ""

                for content in assert_matches:
                    parts = [p.strip() for p in content.split(',', 2)]
                    if len(parts) == 3:
                        triple = (parts[0], parts[1], parts[2])
                        
                        reward_res = self.engine.calculate_reward(triple, context_for_reward)
                        r = reward_res["reward"]
                        
                        # Coverage Bonus
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
                        
                        if reward_res['type'] in ['HIT_GOLD', 'WEAK_MATCH']:
                            fact_str = f"{parts[0]}, {parts[1]}, {parts[2]}"
                            if fact_str not in self.visible_facts:
                                self.visible_facts.append(fact_str)
                                new_facts.append(fact_str)
                                self.evidence_db[fact_str] = context_for_reward
                
                if new_facts:
                    self.history_summary.append(f"- Asserted {len(new_facts)} facts")
                elif assert_matches:
                    self.history_summary.append(f"- {len(assert_matches)} assertions rejected (low quality/distractor).")

                print(f"   [Step Total Reward] {step_reward:.2f}")
                self.current_observation_str = None
            return True

        elif last_action == 'retrieve':
            retrieve_matches = re.findall(r"[<=]?retrieve>(.*?)</retriev", response, re.DOTALL)
            if retrieve_matches:
                content = retrieve_matches[-1].strip()
                parts = [p.strip() for p in content.split(',', 2)]
                if len(parts) == 3:
                    fact_key = f"{parts[0]}, {parts[1]}, {parts[2]}"
                    print(f"   [Action] Retrieve: {fact_key}")
                    
                    source_text = self.evidence_db.get(fact_key)
                    
                    if source_text is None:
                        print(f"        -> Exact fact not in Evidence DB. Finding most similar fact...")
                        if self.evidence_db:
                            query_emb = self.engine.encode_texts([fact_key], is_query=True)
                            db_keys = list(self.evidence_db.keys())
                            db_keys_embs = self.engine.encode_texts(db_keys, is_query=False)
                            scores = (query_emb @ db_keys_embs.T)[0]
                            best_match_idx = np.argmax(scores)
                            best_match_key = db_keys[best_match_idx]
                            best_match_score = scores[best_match_idx]
                            
                            print(f"        -> Best match (Score: {best_match_score:.4f}): {best_match_key}")
                            source_text = self.evidence_db[best_match_key]
                        else:
                            source_text = "No record found in an empty Evidence DB."
                    self.current_observation_str = f"Fact: <{fact_key}>\nSource:\n{source_text}"
                    self.history_summary.append(f"- Retrieve: {fact_key}")
                    print(f"\n{'='*20} Retrieved Evidence {'='*20}\n{source_text}\n{'='*62}\n")
                return True

        elif last_action == 'answer':
            answer_matches = re.findall(r"[<=]?answer>(.*?)</answe", response, re.DOTALL)
            if answer_matches:
                ans = answer_matches[-1].strip()
                print(f"\n>>> [Final Answer]: {ans}")
                return False 

        print("   [Warning] Unable to parse action (last action was identified but content extraction failed).")
        return True

    def run(self, query: str, max_steps=10):
        print(f"Task: {query}\n")
        debug_logs = []
        log_file = "test.txt"
        for step in range(max_steps):
            print(f"--- Step {step + 1} ---")
            messages = self.build_context(query)
            # debug 输出input
            # print("\n[Model Input Messages]")
            # for m in messages:
            #     content_preview = m['content']
            #     print(f"[{m['role'].upper()}]:\n{content_preview}")
            #     print("-" * 40)
            response = self.engine.generate(messages)
            # log_entry = {
            #     "step": step + 1,
            #     "messages": messages,
            #     "response": response
            # }
            # debug_logs.append(log_entry)
            # try:
            #     with open("test.json", "w", encoding="utf-8") as f:
            #         json.dump(debug_logs, f, ensure_ascii=False, indent=2)
            # except Exception as e:
            #     print(f"[Warning] Logging failed: {e}")
            try:
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(f"--- Step {step + 1} ---\n")
                    f.write("[MESSAGES]:\n")
                    for msg in messages:
                        role = msg['role'].upper()
                        content = msg['content']
                        f.write(f"[{role}]:\n{content}\n\n")
                    
                    f.write("-" * 40 + "\n")
                    f.write("[RESPONSE]:\n")
                    f.write(response)
                    f.write("\n\n" + "="*60 + "\n\n")
            except Exception as e:
                print(f"[Warning] Logging failed: {e}")
            print(f"[LLM Output]: {response}")
            keep_going = self.parse_and_execute(response)
            if not keep_going:
                break
        print("\nTask Finished.")

# ================= Data Loading Helper =================

def load_hotpot_sample(index=0):
    """
    Load a single sample from HotpotQA validation set and format it for the agent.
    """
    print(">>> Loading HotpotQA Dataset (distractor/validation)...")
    dataset = load_dataset("hotpot_qa", "fullwiki", split="train", trust_remote_code=True)
    print(len(dataset))
    data = dataset[index]
    question = data["question"]
    print(data["answer"])
    # 1. Build Corpus from Context
    corpus = []
    for title, sentences in zip(data["context"]["title"], data["context"]["sentences"]):
        full_text = "".join(sentences)
        corpus.append({
            "title": title, 
            "contents": f'Title: "{title}"\nContent: {full_text}'
        })
    
    # 2. Extract Gold Supporting Facts (Sentences)
    gold_facts = []
    context_dict = {t: s for t, s in zip(data["context"]["title"], data["context"]["sentences"])}
    
    for title, sent_id in zip(data["supporting_facts"]["title"], data["supporting_facts"]["sent_id"]):
        if title in context_dict and sent_id < len(context_dict[title]):
            sent = context_dict[title][sent_id]
            gold_facts.append(sent)
            
    print(f"   Question: {question}")
    print(f"   Gold Facts Count: {len(gold_facts)}")
    
    return question, corpus, gold_facts

# ================= Main =================

if __name__ == "__main__":
    # 1. Load Data
    query, corpus, gold_facts = load_hotpot_sample(index=0)
    
    # 2. Initialize Engine
    engine = ModelEngine()
    engine.setup_task(corpus, gold_facts)
    
    # 3. Initialize Agent & Run
    agent = FactStoreAgent(engine)
    agent.run(query)
