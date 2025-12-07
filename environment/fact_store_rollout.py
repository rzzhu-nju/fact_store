from verl.workers.rollout.vllm_rollout import vLLMRollout
from verl import DataProto
from environment.generation_no_memory import BatchFactStoreAgent
from environment.reward_utils import calculate_final_answer_reward
import torch
import numpy as np
from vllm import SamplingParams

class FactStoreRollout(vLLMRollout):
    def __init__(self, config, model_config, device_mesh):
        super().__init__(config, model_config, device_mesh)
        
        # Initialize Retriever (Real or Mock)
        # In a real setup, ensure 'retriever_server.py' is running and accessible
        try:
            from environment.generation_no_memory import SimpleDenseRetriever, Encoder
            # We assume the retriever server is used via HTTP or local if possible
            # If SimpleDenseRetriever connects to a server, that's great.
            # If it loads local index, make sure we have memory.
            # For Online RL, usually we use a light HTTP client.
            # Let's assume we use the same class as in generation script.
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
            self.reward_encoder = Encoder(
                 model_name="e5",
                 model_path="/data1/shares/e5-base-v2",
                 pooling_method="mean",
                 max_length=512,
                 use_fp16=True
            )
        except Exception as e:
            print(f"Warning: Could not load Retriever/Encoder in Rollout ({e}). Using Mocks.")
            class MockRetriever:
                def search(self, queries, num=3):
                    return [[{'contents': f'Content for {q}'} for _ in range(num)] for q in queries], None
            
            class MockEncoder:
                def encode(self, texts, **kwargs):
                    return np.random.rand(len(texts), 768)
                    
            self.retriever = MockRetriever()
            self.reward_encoder = MockEncoder()

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """
        Override the default generation to run the multi-step Search-Reasoning loop.
        """
        # 1. Extract Input IDs
        input_ids = prompts.batch['input_ids'] # Tensor [B, SeqLen]
        
        # 2. Get Tokenizer
        if hasattr(self.inference_engine, 'tokenizer'):
            tokenizer = self.inference_engine.tokenizer
        else:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.model_config.path)

        # 3. Decode Prompts
        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.cpu().tolist()
        prompts_text = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        
        # 4. Initialize Agents
        agents = []
        non_tensor = getattr(prompts, 'non_tensor_batch', {})
        answers = non_tensor.get('answer', [""] * len(prompts_text))
        gold_facts_list = non_tensor.get('gold_facts', [None] * len(prompts_text))
        
        for i, q in enumerate(prompts_text):
            agent = BatchFactStoreAgent(
                None, 
                self.retriever, 
                self.reward_encoder, 
                q, 
                answers[i] if i < len(answers) else "",
                gold_facts_list[i] if i < len(gold_facts_list) else None
            )
            agents.append(agent)
            
        active_agents = agents
        MAX_ROUNDS = 10
        
        # 5. Setup Sampling Params
        # Use config params if available, else defaults
        # We want the model to stop at closing tags or reasonable length
        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.8,
            max_tokens=512,
            stop=["</search>", "</answer>", "</assert>", "</retrieve>"] # Stop at end of action to allow environment intervention
        )
        
        # --- Main Interaction Loop ---
        for _ in range(MAX_ROUNDS):
            if all(a.is_finished for a in active_agents): break
            
            # A. Prepare Prompts for this step
            gen_indices = []
            batch_prompts = []
            
            for i, agent in enumerate(active_agents):
                if not agent.is_finished and agent.next_action == "generate":
                    agent.step() # Updates action_data (messages)
                    
                    # Convert messages to prompt string using chat template
                    prompt_str = tokenizer.apply_chat_template(
                        agent.action_data, 
                        tokenize=False, 
                        add_generation_prompt=True
                    )
                    batch_prompts.append(prompt_str)
                    gen_indices.append(i)
            
            # B. Generate with vLLM
            if batch_prompts:
                # Call vLLM generate
                # Note: In sync rollout, we expect a blocking call that returns results
                outputs = self.inference_engine.generate(
                    prompts=batch_prompts,
                    sampling_params=sampling_params
                )
                
                # Extract text
                responses = []
                for output in outputs:
                    if hasattr(output, 'outputs'):
                        # vLLM RequestOutput
                        # We need to append the stop token if it was stripped, because parse_and_execute expects it?
                        # BatchFactStoreAgent parser uses regex which handles missing closing tag somewhat,
                        # but let's rely on the generated text.
                        responses.append(output.outputs[0].text)
                    else:
                        responses.append(str(output))
                
                # Update Agents
                for idx, resp in zip(gen_indices, responses):
                    active_agents[idx].next_action = "parse"
                    active_agents[idx].step(resp)
            
            # C. Handle Search (Environment Step)
            search_indices = []
            queries = []
            for i, agent in enumerate(active_agents):
                if not agent.is_finished and agent.next_action == "search":
                    queries.append(agent.action_data)
                    search_indices.append(i)
            
            if queries:
                # Execute Search
                # Using the retriever initialized in __init__
                results_batch, _ = self.retriever.search(queries, num=3)
                
                for idx, results in zip(search_indices, results_batch):
                    agent = active_agents[idx]
                    
                    # Format results
                    result_strs = []
                    for k, doc in enumerate(results):
                        contents = doc.get('contents', '')
                        result_strs.append(f"[Doc {k+1}] {contents}")
                    result_str = "\n\n".join(result_strs)
                    
                    # Update Agent State
                    agent.current_observation_str = result_str
                    agent.logs.append({
                        "step": agent.steps_taken,
                        "type": "search_result",
                        "query": agent.action_data, # The query
                        "result_snippet": result_str[:200] + "..."
                    })
                    agent.history_summary.append(f"- Search executed: {agent.action_data}")
                    agent.next_action = "generate"

        # --- Reconstruction & Padding ---
        output_input_ids = []
        output_masks = []
        output_rewards = []
        
        max_len = 0
        
        for agent in agents:
            # Get full trace with mask
            ids, mask = agent.reconstruct_training_sample(tokenizer)
            
            # Calculate final reward
            # Note: We use the outcome reward (EM) for the whole trajectory
            # GRPO will attribute this to all actions in the trajectory
            final_reward = calculate_final_answer_reward(
                agent.final_answer or "",
                agent.answer[0] if isinstance(agent.answer, list) else agent.answer,
                embedding_model=self.reward_encoder
            )
            
            output_input_ids.append(torch.tensor(ids))
            output_masks.append(torch.tensor(mask))
            output_rewards.append(torch.tensor(final_reward))
            
            if len(ids) > max_len:
                max_len = len(ids)
                
        # Pad sequences
        # We need to create a batch tensor [B, MaxLen]
        # Padding value usually 0 or tokenizer.pad_token_id
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        
        padded_input_ids = torch.full((len(agents), max_len), pad_token_id, dtype=torch.long)
        padded_masks = torch.full((len(agents), max_len), 0, dtype=torch.long) # Mask 0 for padding
        padded_attention_mask = torch.full((len(agents), max_len), 0, dtype=torch.long)
        
        for i, (ids, mask) in enumerate(zip(output_input_ids, output_masks)):
            l = len(ids)
            padded_input_ids[i, :l] = ids
            padded_masks[i, :l] = mask
            padded_attention_mask[i, :l] = 1 # 1 for valid tokens
            
        # Construct DataProto
        # Verl expects: input_ids, attention_mask, position_ids, labels (optional)
        # For PPO, we need 'old_log_probs' usually computed by Actor, but Rollout just provides data.
        # We attach 'reward_tensor' to meta_info or batch?
        # GRPO expects rewards in the batch usually.
        
        batch_dict = {
            'input_ids': padded_input_ids,
            'attention_mask': padded_attention_mask,
            'loss_mask': padded_masks, # Verl uses loss_mask
            'reward': torch.stack(output_rewards)
        }
        
        # We must return DataProto
        return DataProto(batch=batch_dict, non_tensor_batch=prompts.non_tensor_batch)
