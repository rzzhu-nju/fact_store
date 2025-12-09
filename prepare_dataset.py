import json
import random
import pandas as pd
import os
from datasets import load_dataset, concatenate_datasets
import pyarrow as pa
import pyarrow.parquet as pq

# --- Configuration ---
HOTPOT_FACT_PATH = "hotpotqa_fact.jsonl"
OUTPUT_DIR = "/data1/rzzhu/my_fact_store_project/data/nq_hotpotqa_train_autorefine"
os.makedirs(OUTPUT_DIR, exist_ok=True)
TRAIN_OUTPUT_FILE = os.path.join(OUTPUT_DIR, "train.parquet")
VAL_OUTPUT_FILE = os.path.join(OUTPUT_DIR, "test.parquet")

# --- Helper Functions ---

def load_jsonl(file_path):
    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data

def format_verl_sample(example, gold_facts=None):
    """
    Formats a single sample into the structure expected by Verl.
    Verl expects:
    - prompt: List of dicts (messages)
    - response: String (gold answer)
    - label: String (gold answer)
    - meta_info: Dict (extra info for reward/logging)
    """
    
    question = example['question']
    answer = example['answer']
    
    # Construct System and User Prompts
    # We use the same prompt template as in generation_no_memory.py
    system_prompt = (
        "You are a verifiable reasoning agent based on a **Fact-Store** architecture.\n\n"
        "**INPUT STRUCTURE**:\n"
        "The user will provide a message containing the following sections:\n"
        "1. **Task**: The question you need to answer.\n"
        "2. **History**: A summary of your past actions and their results.\n"
        "3. **Executed Searches**: Queries you have already performed (to avoid duplication).\n"
        "4. **Observation**: The text documents retrieved by your LAST <search> action. **NOTE**: If your last action was NOT <search>, this section will be None/Empty.\n\n"
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
        "- <assert>Subject, Relation, Object</assert>: Extract facts from observation (e.g. <assert>Obama, born in, Hawaii</assert>). DO NOT include labels like 'Subject:'. Just use commas.\n"
        "- <retrieve>Subject, Relation, Object</retrieve>: Check sources.\n"
        "- <answer>Final Answer</answer>: Use Fact-Store to answer.\n\n"
        "=== Current Fact-Store ===\n"
        "(Empty)\n\n"
    )
    
    user_prompt = (
        f"Task: {question}\n\n"
        "Instruction: Fact-Store is empty. Please think and <search>."
    )
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    # Meta Info for Reward Function
    meta_info = {
        "question": question,
        "answer": answer,
        "gold_facts": gold_facts if gold_facts else []
    }
    
    return {
        "prompt": messages,
        "response": answer,
        "label": answer,
        "meta_info": meta_info
    }

# --- 1. Prepare Training Set (HotpotQA with Facts) ---
print(">>> Preparing Training Set...")

# Load Fact Map
print("Loading hotpotqa_fact.jsonl...")
fact_data = load_jsonl(HOTPOT_FACT_PATH)
fact_map = {item['id']: item['fact'] for item in fact_data}
valid_ids = list(fact_map.keys())

# Sample 2000 IDs
print(f"Total available facts: {len(valid_ids)}")
if len(valid_ids) >= 2000:
    sampled_ids = random.sample(valid_ids, 2000)
else:
    print(f"Warning: Only {len(valid_ids)} facts available, using all.")
    sampled_ids = valid_ids
    
sampled_id_set = set(sampled_ids)

# Load HotpotQA Dataset to get Questions/Answers
print("Loading hotpot_qa dataset from HuggingFace...")
# We use 'distractor' split as it is commonly used, or 'fullwiki'. 
# The 'train' split usually contains the questions.
hotpot_dataset = load_dataset("hotpot_qa", "distractor", split="train", trust_remote_code=True)

train_samples = []
found_count = 0

print("Matching sampled IDs with HotpotQA dataset...")
for example in hotpot_dataset:
    ex_id = example['id']
    if ex_id in sampled_id_set:
        # Get Gold Facts
        # Note: fact_map[ex_id] is list of triples [[s,r,o], ...]
        # We format them as strings "s, r, o" for the reward function
        raw_facts = fact_map[ex_id]
        formatted_gold_facts = [f"{f[0]}, {f[1]}, {f[2]}" for f in raw_facts]
        
        sample = format_verl_sample(example, formatted_gold_facts)
        train_samples.append(sample)
        found_count += 1
        
        if found_count >= len(sampled_ids):
            break

print(f"Constructed {len(train_samples)} training samples.")

# Convert to DataFrame and Parquet
df_train = pd.DataFrame(train_samples)

# Helper to serialize complex objects for Parquet
def serialize_row(row):
    return {
        "prompt": json.dumps(row["prompt"]), # Verl often expects object or json string? 
        # Actually Verl expects `prompt` column to be list of dicts if using internal logic, 
        # BUT standard parquet doesn't support list of dicts well without schema.
        # Verl's `create_rl_dataset` handles different formats.
        # Let's look at `create_rl_dataset` logic.
        # It usually uses `apply_chat_template` on `prompt` column.
        # If `prompt` is list of dicts, it's fine for Arrow/Parquet if schema allows.
        # However, to be safe and consistent with common practices, we keep it as Python objects in DataFrame
        # and let PyArrow handle serialization.
        
        # Wait, PyArrow handles List[Struct] fine.
        "prompt": row["prompt"],
        "response": row["response"],
        "label": row["label"],
        "meta_info": row["meta_info"] # Dict, might need to be struct
    }

# Actually, let's keep it simple.
# PyArrow will infer schema.
# But meta_info is a dict with mixed types (list of strings).
# Let's ensure meta_info is stored as a struct or JSON string if Verl supports it.
# Verl usually looks for `meta_info` key in `non_tensor_batch`.

# Let's try direct conversion.
try:
    # We need to ensure types are consistent for Arrow
    # prompt: List[Dict[str, str]]
    # meta_info: Dict[str, Any]
    
    # Clean meta_info: ensure all values are arrow-compatible
    # gold_facts is List[str], answer is str, question is str.
    pass
except Exception as e:
    print(f"Error preparing dataframe: {e}")

# Save Training Data
table_train = pa.Table.from_pandas(df_train)
pq.write_table(table_train, TRAIN_OUTPUT_FILE)
print(f"Saved training data to {TRAIN_OUTPUT_FILE}")


# --- 2. Prepare Validation Set (NQ + HotpotQA Dev) ---
print("\n>>> Preparing Validation Set (NQ + HotpotQA)...")

# Load Dev Sets from FlashRAG
# Note: FlashRAG dataset structure needs inspection.
# We'll use streaming to find dev/test splits.
# Based on common knowledge:
# nq: test
# hotpotqa: validation/distractor

print("Loading FlashRAG NQ (test)...")
nq_dataset = load_dataset("RUC-NLPIR/FlashRAG_datasets", "nq", split="test", streaming=True)
print("Loading FlashRAG HotpotQA (dev)...")
hotpot_val_dataset = load_dataset("RUC-NLPIR/FlashRAG_datasets", "hotpotqa", split="dev", streaming=True) # FlashRAG uses 'dev' for HotpotQA

val_samples = []

# Sample 200 from NQ
print("Sampling 200 from NQ...")
count = 0
for example in nq_dataset:
    if count >= 200:
        break
    
    # NQ in FlashRAG might have 'question', 'golden_answers' (list)
    # Check format
    q = example.get('question')
    a_list = example.get('golden_answers', [])
    a = a_list[0] if a_list else ""
    
    # NQ doesn't have "Gold Facts" triples usually.
    # For validation, we might just check answer correctness (Outcome Reward).
    # Or we leave gold_facts empty.
    
    sample = format_verl_sample({'question': q, 'answer': a}, gold_facts=[])
    val_samples.append(sample)
    count += 1

# Sample 200 from HotpotQA
print("Sampling 200 from HotpotQA...")
count = 0
for example in hotpot_val_dataset:
    if count >= 200:
        break
    
    q = example.get('question')
    a_list = example.get('golden_answers', [])
    a = a_list[0] if a_list else ""
    
    # HotpotQA in FlashRAG might not have structured facts either, 
    # unless we cross-reference or parse 'supporting_facts'.
    # For now, we leave gold_facts empty for validation set too, 
    # focusing on Outcome Reward for validation metrics.
    
    sample = format_verl_sample({'question': q, 'answer': a}, gold_facts=[])
    val_samples.append(sample)
    count += 1

print(f"Constructed {len(val_samples)} validation samples.")

# Save Validation Data
df_val = pd.DataFrame(val_samples)
table_val = pa.Table.from_pandas(df_val)
pq.write_table(table_val, VAL_OUTPUT_FILE)
print(f"Saved validation data to {VAL_OUTPUT_FILE}")

print("\n>>> Dataset Preparation Complete!")
