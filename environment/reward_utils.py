import torch
import numpy as np
from typing import Tuple, Dict, Any, List

def compute_cosine_similarity(emb1, emb2):
    """
    Compute cosine similarity between two embeddings.
    Assumes embeddings are already normalized.
    """
    if isinstance(emb1, torch.Tensor):
        emb1 = emb1.cpu().numpy()
    if isinstance(emb2, torch.Tensor):
        emb2 = emb2.cpu().numpy()
        
    # Flatten if necessary
    if len(emb1.shape) > 1: emb1 = emb1.flatten()
    if len(emb2.shape) > 1: emb2 = emb2.flatten()
    
    return np.dot(emb1, emb2)

def calculate_reward(
    triple: Tuple[str, str, str], 
    current_context: str, 
    embedding_model: Any = None,
    gold_facts: List[str] = None
) -> Dict[str, Any]:
    """
    Calculates reward for an extracted triple based on similarity with Golden Facts.
    """
    
    subject, relation, obj = triple
    triple_str = f"{subject}, {relation}, {obj}"
    
    # Base reward for valid format
    reward = 0.0
    max_sim = 0.0
    matched_gold = None
    
    if gold_facts and embedding_model:
        try:
            triple_emb = embedding_model.encode([triple_str], is_query=False)[0]
            gold_embs = embedding_model.encode(gold_facts, is_query=False)
            
            sims = []
            for g_emb in gold_embs:
                sim = compute_cosine_similarity(triple_emb, g_emb)
                sims.append(sim)
            
            if sims:
                max_sim = max(sims)
                best_idx = sims.index(max_sim)
                matched_gold = gold_facts[best_idx]
                
            # === Reward Logic (Relaxed V3.3) ===
            # Based on user feedback: 0.9, 0.75, 0.6 thresholds
            # Additive/Subtractive logic
            
            HIGH_THRESHOLD = 0.95
            MEDIUM_THRESHOLD = 0.9
            LOW_THRESHOLD = 0.8
            
            if max_sim >= HIGH_THRESHOLD:
                reward = 0.2  # Strong match
            elif max_sim >= MEDIUM_THRESHOLD:
                reward = 0.1  # Good match
            elif max_sim < LOW_THRESHOLD:
                reward = -0.1 # Mild Penalty for noise
            else:
                reward = 0.0   # Neutral zone (0.6 - 0.75)
                
        except Exception as e:
            print(f"Error in reward calculation: {e}")
            reward = 0.0
            
    else:
        if current_context and (subject in current_context) and (obj in current_context):
            reward = 0.05
        else:
            reward = 0.0

    return {
        "reward": reward,
        "type": "fact_extraction",
        "max_similarity": float(max_sim),
        "matched_gold": matched_gold,
        "desc": f"Extracted: {triple_str} | Max Sim: {max_sim:.3f}"
    }

def precomputed_reward_fn(data_proto: Any) -> torch.Tensor:
    """
    Reward function that expects rewards to be already computed in the batch.
    Used when the Rollout Worker calculates rewards (e.g. process rewards).
    """
    if 'precomputed_rewards' in data_proto.batch:
        return data_proto.batch['precomputed_rewards']
    
    # Fallback: Return zeros if not found
    batch_size = data_proto.batch['input_ids'].shape[0]
    seq_len = data_proto.batch['input_ids'].shape[1]
    return torch.zeros((batch_size, seq_len), device=data_proto.batch['input_ids'].device)

def calculate_final_answer_reward(
    student_answer: str,
    gold_answer: str,
    embedding_model: Any = None
) -> float:
    """
    Calculate final answer reward.
    """
    if not gold_answer: return 0.0
    
    # Normalize input
    if isinstance(gold_answer, list):
        gold_answer = gold_answer[0] # Take first if list
        
    # 1. Exact Match (Case Insensitive)
    if gold_answer.lower() in student_answer.lower():
        return 1.0
        
            
    return 0.0

def calculate_retrieve_penalty() -> Dict[str, Any]:
    return {
        "reward": -0.5,
        "type": "retrieve_penalty",
        "desc": "Retrieve tool used"
    }

def compute_score(solution_str, ground_truth, **kwargs):
    """
    Adapter for Verl custom reward function.
    Verl calls this to compute the outcome reward.
    """
    # Note: 'embedding_model' is not passed by Verl by default.
    # If you strictly need embedding reward here, you'd need to load it globally or pass it via kwargs if Verl supports it.
    # For now, we rely on Exact Match which is robust and fast.
    return calculate_final_answer_reward(solution_str, ground_truth, embedding_model=None)
