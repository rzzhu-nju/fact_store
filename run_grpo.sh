#!/bin/bash
# run_grpo.sh: Launch GRPO training for Fact-Store Agent
# This script uses Hydra overrides to configure the training without modifying the base config file.

# === Configuration ===
# Dataset Paths
DATA_DIR="/data1/rzzhu/my_fact_store_project/data/nq_hotpotqa_train_autorefine"
TRAIN_FILES="$DATA_DIR/train.parquet"
VAL_FILES="$DATA_DIR/test.parquet"

# Model Path (Actor & Reference)
MODEL_PATH="/data1/shares/Qwen2.5-7B-Instruct"

# Checkpoint Path
EXP_NAME="fact_store_grpo_v1"
CHECKPOINT_DIR="/data1/rzzhu/checkpoints/$EXP_NAME"

# Custom Reward Function Path
REWARD_FUNC_PATH="$(pwd)/environment/reward_utils.py"

# Rollout Worker
# We registered 'fact_store' in verl/workers/rollout/base.py
ROLLOUT_NAME="fact_store"

# === Environment Setup ===
# Use GPUs 0-3 for Training (4 GPUs)
# GPU 4 is reserved for Retriever/Reference if needed, but here we run everything in one go or distributed.
# The user asked for 0-3 for training, 4 for Retriever/Reference.
# Since we are running PPO, we need to manage this.
# However, Verl handles device placement. We set CUDA_VISIBLE_DEVICES for the whole process.
# We will use 4 GPUs for the experiment as per user request (0-3).
export CUDA_VISIBLE_DEVICES=0,1,2,3

# === Training Command ===
# Key Overrides:
# - algorithm.adv_estimator=grpo: Enable GRPO (Group Relative Policy Optimization)
# - actor_rollout_ref.rollout.name=$ROLLOUT_NAME: Use our custom FactStoreRollout
# - custom_reward_function.path=$REWARD_FUNC_PATH: Use our sparse+dense reward logic
# - trainer.n_gpus_per_node=4: Match available GPUs
# - algorithm.gamma=0.95: Discount factor for trace rewards
# - data.train_files/val_files: Point to our dataset
# - actor_rollout_ref.model.path: Point to base model

echo "Starting GRPO Training..."
echo "Model: $MODEL_PATH"
echo "Data: $DATA_DIR"
echo "Rollout: $ROLLOUT_NAME"

python3 -m verl.trainer.main_ppo \
    data.train_files=$TRAIN_FILES \
    data.val_files=$VAL_FILES \
    data.train_batch_size=128 \
    data.val_batch_size=64 \
    data.max_prompt_length=1024 \
    data.max_response_length=2048 \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.rollout.name=$ROLLOUT_NAME \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.adv_estimator=grpo \
    algorithm.gamma=0.95 \
    custom_reward_function.path=$REWARD_FUNC_PATH \
    custom_reward_function.name=precomputed_reward_fn \
    trainer.project_name=fact_store_rl \
    trainer.experiment_name=$EXP_NAME \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.default_local_dir=$CHECKPOINT_DIR \
    +trainer.val_before_train=False 
