import json
import os
import warnings
from typing import List, Dict, Optional
import argparse

import faiss
import torch
import numpy as np
from transformers import AutoConfig, AutoTokenizer, AutoModel
from tqdm import tqdm
import datasets

def load_corpus(corpus_path: str):
    """Loads a corpus from a JSONL file."""
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
    model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    model.eval()
    if torch.cuda.is_available():
        model.cuda()
    if use_fp16:
        model = model.half()
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
    return model, tokenizer

def pooling(
    pooler_output,
    last_hidden_state,
    attention_mask = None,
    pooling_method = "mean"
):
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
        self.model.eval()

    @torch.no_grad()
    def encode(self, query_list: List[str], is_query=True):
        """Encodes a list of queries."""
        if isinstance(query_list, str):
            query_list = [query_list]

        if "e5" in self.model_name.lower():
            if is_query:
                query_list = [f"query: {query}" for query in query_list]
            else:
                query_list = [f"passage: {query}" for query in query_list]

        inputs = self.tokenizer(query_list,
                                max_length=self.max_length,
                                padding=True,
                                truncation=True,
                                return_tensors="pt"
                                )
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        output = self.model(**inputs, return_dict=True)
        query_emb = pooling(output.pooler_output,
                            output.last_hidden_state,
                            inputs['attention_mask'],
                            self.pooling_method)
        query_emb = torch.nn.functional.normalize(query_emb, dim=-1)

        query_emb = query_emb.detach().cpu().numpy()
        query_emb = query_emb.astype(np.float32, order="C")
        
        del inputs, output
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return query_emb

class SimpleDenseRetriever:
    """A simplified dense retriever for demonstration."""
    def __init__(self, model_name, model_path, pooling_method, max_length, use_fp16,
                 index_path, corpus_path, topk, faiss_gpu=False):
        self.encoder = Encoder(
            model_name=model_name,
            model_path=model_path,
            pooling_method=pooling_method,
            max_length=max_length,
            use_fp16=use_fp16
        )
        self.index = faiss.read_index(index_path)
        if faiss_gpu and torch.cuda.is_available():
            print("Using GPU for Faiss.")
            co = faiss.GpuMultipleClonerOptions()
            co.useFloat16 = True
            co.shard = True
            self.index = faiss.index_cpu_to_all_gpus(self.index, co=co)
            
        self.corpus = load_corpus(corpus_path)
        self.topk = topk

    def search(self, queries: List[str] or str, num: int = None):
        """Performs a search for a batch of queries."""
        if isinstance(queries, str):
            queries = [queries]
        if num is None:
            num = self.topk
        query_embs = self.encoder.encode(queries)
        scores, idxs = self.index.search(query_embs, k=num)
        
        results = []
        for i in range(len(queries)):
            results.append(load_docs(self.corpus, idxs[i]))
        return results, scores.tolist()

if __name__ == "__main__":
    # --- Configuration ---
    INDEX_PATH = "/data1/rzzhu/wiki-2018/e5_Flat.index"
    CORPUS_PATH = "/data1/rzzhu/wiki-2018/wiki-18.jsonl"
    MODEL_PATH = "/data1/shares/e5-base-v2"
    MODEL_NAME = "e5"
    QUERY = "Which magazine was started first Arthur's Magazine or First for Women?"

    # --- Instantiate Retriever ---
    retriever = SimpleDenseRetriever(
        model_name=MODEL_NAME,
        model_path=MODEL_PATH,
        pooling_method="mean",
        max_length=256,
        use_fp16=True,
        index_path=INDEX_PATH,
        corpus_path=CORPUS_PATH,
        topk=5,
        faiss_gpu=False 
    )

    # --- Perform Search ---
    print(f"Searching for: '{QUERY}'")
    results, scores = retriever.search(QUERY)

    # --- Print Results ---
    print("\n--- Top Results ---")
    for i, (doc, score) in enumerate(zip(results, scores)):
        # The corpus from wiki-18.jsonl seems to have 'title' and 'text' fields.
        title = doc.get('contents', '').split('\n')[0].strip('"')
        text = '\n'.join(doc.get('contents', '').split('\n')[1:])
        print(f"Rank {i+1} (Score: {score:.4f}):")
        print(f"  Title: {title}")
        print(f"  Length: {len(doc.get('contents', ''))} characters")
        print(f"  Text: {text[:300]}...")
        print("-" * 20)
