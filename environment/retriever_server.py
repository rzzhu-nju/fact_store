import os
import json
import argparse
import multiprocessing
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
import sys

# Server dependencies
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# ML/Retrieval dependencies
import torch
import numpy as np
import faiss

# --- 1. 更加健壮的 transformers 导入 ---
try:
    from transformers import AutoConfig, AutoTokenizer, AutoModel
except ImportError as e:
    if "flash_attn" in str(e):
        print("\n" + "!"*50)
        print("CRITICAL ERROR: 'flash-attn' is causing compatibility issues.")
        print("SOLUTION: Please run `pip uninstall flash-attn -y` in your terminal.")
        print("!"*50 + "\n")
        sys.exit(1)
    raise e

import datasets

# --- 配置: 限制 CPU 使用核心数 ---
CPU_LIMIT = 8

# Set Faiss to use limited CPU cores for index searching
try:
    # 限制 Faiss 使用的线程数
    faiss.omp_set_num_threads(CPU_LIMIT)
    print(f"Setting Faiss OMP threads to {CPU_LIMIT}")
except Exception as e:
    print(f"Warning: Could not set faiss threads: {e}")

# ================= 检索器核心逻辑 =================

def load_corpus(corpus_path: str):
    """Loads a corpus from a JSONL file."""
    if not os.path.exists(corpus_path):
        raise FileNotFoundError(f"Corpus file not found at: {corpus_path}")
    print(f"Loading corpus from {corpus_path}...")
    corpus = datasets.load_dataset(
        'json',
        data_files=corpus_path,
        split="train",
    )
    return corpus

def load_docs(corpus, doc_idxs):
    """Loads documents from the corpus based on their indices."""
    results = [corpus[int(idx)] for idx in doc_idxs]
    return results

def load_model(model_path: str, use_fp16: bool = False):
    """Loads a Hugging Face model and tokenizer."""
    if not os.path.exists(model_path):
         raise FileNotFoundError(f"Model path not found at: {model_path}")
         
    model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    model.eval()
    if torch.cuda.is_available():
        model.cuda()
    if use_fp16:
        model = model.half()
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
    return model, tokenizer

def pooling(pooler_output, last_hidden_state, attention_mask=None, pooling_method="mean"):
    """Performs pooling on the model's output."""
    if pooling_method == "mean":
        last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    elif pooling_method == "cls":
        return last_hidden_state[:, 0]
    elif pooling_method == "pooler":
        return pooler_output
    else:
        raise NotImplementedError("Pooling method not implemented!")

class Encoder:
    """Encodes text into embeddings."""
    def __init__(self, model_name, model_path, pooling_method, max_length, use_fp16):
        self.model_name = model_name
        self.model_path = model_path
        self.pooling_method = pooling_method
        self.max_length = max_length
        self.use_fp16 = use_fp16
        self.model, self.tokenizer = load_model(model_path=model_path, use_fp16=use_fp16)

    @torch.no_grad()
    def encode(self, query_list: List[str], is_query=True):
        if isinstance(query_list, str):
            query_list = [query_list]

        if "e5" in self.model_name.lower():
            if is_query:
                query_list = [f"query: {query}" for query in query_list]
            else:
                query_list = [f"passage: {query}" for query in query_list]

        inputs = self.tokenizer(query_list, max_length=self.max_length, padding=True, truncation=True, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        output = self.model(**inputs, return_dict=True)
        query_emb = pooling(output.pooler_output, output.last_hidden_state, inputs['attention_mask'], self.pooling_method)
        query_emb = torch.nn.functional.normalize(query_emb, dim=-1)
        query_emb = query_emb.detach().cpu().numpy().astype(np.float32, order="C")
        
        del inputs, output
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return query_emb

class SimpleDenseRetriever:
    def __init__(self, model_name, model_path, pooling_method, max_length, use_fp16, index_path, corpus_path, topk, faiss_gpu=False):
        self.encoder = Encoder(model_name, model_path, pooling_method, max_length, use_fp16)
        
        if not os.path.exists(index_path):
             raise FileNotFoundError(f"Index file not found at: {index_path}")

        print(f"Loading index from {index_path}...")
        self.index = faiss.read_index(index_path)
        if faiss_gpu and torch.cuda.is_available():
            print("Using GPU for Faiss.")
            co = faiss.GpuMultipleClonerOptions()
            co.useFloat16 = True
            co.shard = True
            self.index = faiss.index_cpu_to_all_gpus(self.index, co=co)
        else:
            print(f"Using CPU for Faiss (Limited to {CPU_LIMIT} threads).")
            faiss.omp_set_num_threads(CPU_LIMIT)
            
        self.corpus = load_corpus(corpus_path)
        self.topk = topk
        self.num_threads = CPU_LIMIT

    def search(self, queries: List[str] or str, num: int = None):
        if isinstance(queries, str): queries = [queries]
        if num is None: num = self.topk
            
        query_embs = self.encoder.encode(queries)
        scores, idxs = self.index.search(query_embs, k=num)
        results = [None] * len(queries)
        
        def fetch_doc(i):
            return load_docs(self.corpus, idxs[i])

        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = {executor.submit(fetch_doc, i): i for i in range(len(queries))}
            for future in futures:
                results[futures[future]] = future.result()

        return results, scores.tolist()

# ================= FastAPI 服务逻辑 (Lifespan Update) =================

# Global State
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    print(">>> Initializing Retriever Server...")
    
    # 硬编码路径，确保无误
    MODEL_PATH = "/data1/shares/e5-base-v2"
    INDEX_PATH = "/data1/rzzhu/wiki-2018/e5_Flat.index"
    CORPUS_PATH = "/data1/rzzhu/wiki-2018/wiki-18.jsonl"
    
    try:
        ml_models["retriever"] = SimpleDenseRetriever(
            model_name="e5",
            model_path=MODEL_PATH,
            pooling_method="mean",
            max_length=512,
            use_fp16=True,
            index_path=INDEX_PATH,
            corpus_path=CORPUS_PATH,
            topk=3,
            faiss_gpu=False
        )
        print(">>> Retriever Initialized Successfully!")
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to load retriever: {e}")
        # In a real app, you might want to exit here, but we'll let it run to show the error
        pass
        
    yield
    
    # Shutdown logic
    ml_models.clear()
    print(">>> Retriever Server Shutting Down...")

app = FastAPI(title="Fact-Store Retriever Service", lifespan=lifespan)

class SearchRequest(BaseModel):
    queries: List[str]
    topk: int = 3

class SearchResponse(BaseModel):
    result: List[str]
    raw_scores: Optional[List[List[float]]] = None

@app.post("/search", response_model=SearchResponse)
def search_api(req: SearchRequest):
    retriever = ml_models.get("retriever")
    if retriever is None:
        raise HTTPException(status_code=503, detail="Retriever not initialized or failed to load.")

    try:
        results, scores = retriever.search(req.queries, num=req.topk)
        
        batch_result_strs = []
        for doc_list in results:
            res_str = ""
            for i, doc in enumerate(doc_list):
                content = doc.get('contents', '')
                if not content:
                    title = doc.get('title', '')
                    text = doc.get('text', '')
                    content = f"{title}\n{text}"
                res_str += f"[Doc {i+1}] {content[:500]}...\n" 
            batch_result_strs.append(res_str)
            
        return {"result": batch_result_strs, "raw_scores": scores}
        
    except Exception as e:
        print(f"Search Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8085, help="Port to run the server on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the server on")
    args = parser.parse_args()
    
    print(f"Starting server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")