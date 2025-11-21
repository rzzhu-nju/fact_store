#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fact-Store Agent V3.1: Global Retrieval Integration

Updates:
1.  Integrated `simple_retrieval` for global, large-scale search.
2.  `ModelEngine` now uses `SimpleDenseRetriever` instead of in-memory search.
3.  Search scope is now the entire Wikipedia corpus, not just HotpotQA context.
"""

import os
import re
import torch
import numpy as np
from typing import List, Dict, Tuple

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

# --- Import from our new retrieval script ---
from simple_retrieval import SimpleDenseRetriever, Encoder

# ================= Configuration =================
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "7")

# ================= Model Engine =================

class ModelEngineV1:
    def __init__(self):
        self.llm_path = "/data1/shares/Qwen2.5-7B-Instruct"
        
        print(">>> Loading LLM...")
        self.tokenizer_llm = AutoTokenizer.from_pretrained(self.llm_path, trust_remote_code=True)
        self.model_llm = AutoModelForCausalLM.from_pretrained(
            self.llm_path, trust_remote_code=True, torch_dtype=torch.float16, use_flash_attention_2=False
        ).eval().cuda()
        
        print(">>> Initializing Global Retriever...")
        # --- Use SimpleDenseRetriever for all search operations ---
        self.retriever = SimpleDenseRetriever(
            model_name="e5",
            model_path="/data1/shares/e5-base-v2",
            pooling_method="mean",
            max_length=512,
            use_fp16=True,
            index_path="/data1/rzzhu/wiki-2018/e5_Flat.index",
            corpus_path="/data1/rzzhu/wiki-2018/wiki-18.jsonl",
            topk=2, # Default top_k for searches
            faiss_gpu=False # As requested, use CPU for Faiss
        )
        
        # The reward calculation still needs its own encoder, separate from the retriever's
        print(">>> Loading Embedding Model for Reward Calculation...")
        self.reward_encoder = Encoder(
            model_name="e5",
            model_path="/data1/shares/e5-base-v2",
            pooling_method="mean",
            max_length=512,
            use_fp16=True
        )

        self.gold_embs = None

    def setup_task(self, gold_facts: List[str]):
        """
        Simplified setup: only encodes gold facts for reward calculation.
        The corpus is now globally managed by the retriever.
        """
        print(f">>> Encoding Gold Facts ({len(gold_facts)} sentences)...")
        # self.gold_embs = self.reward_encoder.encode(gold_facts, is_query=False)

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
        Returns formatted string of top-k documents by calling the global retriever.
        """
        print(f"    (Forwarding to SimpleDenseRetriever: '{query}')")
        results, _ = self.retriever.search(query, num=top_k)
        
        result_strs = []
        for i, doc in enumerate(results):
            # Extract content and length as we did in simple_retrieval.py
            contents = doc.get('contents', '')
            title = contents.split('\n')[0].strip('"')
            text = '\n'.join(contents.split('\n')[1:])
            length = len(contents)
            
            # Format for the agent
            result_strs.append(f"[Doc {i+1}] Title: {title} (Length: {length} chars)\n{text}")
            
        return "\n\n".join(result_strs)

    def calculate_reward(self, triple: Tuple[str,str,str], current_context: str) -> Dict:
        return {"reward": 0.5, "type": "HIT_GOLD", "gold_idx": max_gold_idx, "score": max_gold_score, "desc": "Core Fact Hit"}
        t_text = f"{triple[0]} {triple[1]} {triple[2]}"
        t_emb = self.reward_encoder.encode([t_text], is_query=True)
        
        scores_gold = (t_emb @ self.gold_embs.T)[0]
        max_gold_idx = np.argmax(scores_gold) if len(scores_gold) > 0 else -1
        max_gold_score = scores_gold[max_gold_idx] if len(scores_gold) > 0 else 0.0
        
        max_ctx_score = 0.0
        if current_context:
            sentences = re.split(r'[.!?\n]', current_context)
            sentences = [s.strip() for s in sentences if len(s) > 10]
            if sentences:
                ctx_embs = self.reward_encoder.encode(sentences, is_query=False)
                scores_ctx = (t_emb @ ctx_embs.T)[0]
                max_ctx_score = np.max(scores_ctx)

        if max_gold_score > 0.8:
            return {"reward": 0.5, "type": "HIT_GOLD", "gold_idx": max_gold_idx, "score": max_gold_score, "desc": "Core Fact Hit"}
        elif 0.5 <= max_gold_score <= 0.8:
            return {"reward": 0.1, "type": "WEAK_MATCH", "gold_idx": -1, "score": max_gold_score, "desc": "Weak Match / Related"}
        else:
            if max_ctx_score > 0.82:
                return {"reward": -0.5, "type": "DISTRACTOR", "gold_idx": -1, "score": max_gold_score, "desc": "Distractor (Irrelevant info)"}
            else:
                return {"reward": -1, "type": "HALLUCINATION", "gold_idx": -1, "score": max_gold_score, "desc": "Hallucination"}

# Keep the original Agent and Data Loading logic, as it remains compatible.
# The agent just calls engine.search, it doesn't care how it's implemented.
from fact_store import FactStoreAgent, load_hotpot_sample

# ================= Main =================

if __name__ == "__main__":
    # 1. Load Data (We still load this to get the question and gold facts for reward)
    query, _, gold_facts = load_hotpot_sample(index=0)
    
    # 2. Initialize NEW Engine
    engine = ModelEngineV1()
    engine.setup_task(gold_facts) # Setup only needs gold facts now
    
    # 3. Initialize Agent & Run
    agent = FactStoreAgent(engine)
    agent.run(query)