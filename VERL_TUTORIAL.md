# Verl & RLHF 傻瓜式入门指南

## 1. 核心概念解释 (给“智障”看的版本)

别担心，RLHF (基于人类反馈的强化学习) 听起来很吓人，其实就几个角色在演戏：

*   **Actor (演员/学生)**: 这就是我们要训练的大模型 (LLM)。它的任务是做题。
*   **Environment (环境/考场)**: 这里指你的 `generation.py` 代码。它负责出题 (User Query)，给资料 (Search Result)，并告诉学生考完了没。
*   **Rollout (模拟考试)**: 让 Actor 在 Environment 里跑一遍，从拿到题目到写出答案的整个过程，就叫一个 Rollout。生成的记录 (Trace) 就是复习资料。
*   **Reward Model (阅卷老师)**: 考完试后，阅卷老师给分。可以是人工打分，也可以是另一个模型打分 (Embedding 相似度)。
*   **Reference Model (参考答案/旧课本)**: 为了防止学生为了拿高分而“走火入魔”（乱写一些阅卷老师喜欢的但其实不对的东西），我们要求学生写的答案不能偏离“旧课本”太远。这个旧课本通常是训练前的原始模型。
*   **Critic (补习老师)**: 在 PPO 算法里，Critic 负责预测学生大概能拿多少分。如果学生考得比 Critic 预测的好，就说明这次发挥超常，要多学学这次的操作；反之就少学。**注意：GRPO 算法不需要 Critic (补习老师)，因为它直接拿一组同学的平均分做比较。**

## 2. 如何启动训练？ (Online RL & Offline RL)

我们提供了两种模式。既然您需要 **Online RL** (边生成边训练)，请参考下面的 "Online RL 流程"。

### 模式一：Online RL (推荐用于论文复现)
在这个模式下，Verl 会在训练过程中调用 `FactStoreRollout`，实时执行 Search-Reasoning 循环。

**前提条件**:
1.  **Retriever**: 必须在 GPU 4 上运行。
2.  **数据**: 准备一个包含 Prompt 的 Parquet 文件 (可以复用 Offline RL 生成的 parquet，或者只包含 questions)。

**启动命令**:
```bash
# 使用 GPU 0-3 进行训练 (Actor + Rollout)
export CUDA_VISIBLE_DEVICES=0,1,2,3

python3 -m verl.trainer.main_ppo \
    data.train_files=data/train.parquet \
    data.val_files=data/train.parquet \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.rollout.name=fact_store \
    trainer.n_gpus_per_node=4
```
*注意*: `rollout.name=fact_store` 告诉 Verl 使用我们自定义的 Worker。

### 模式二：Offline RL (调试用)
先生成数据，再进行 PPO 训练。流程更可控，适合调试 Mask 和 Reward。

### 第一步：生成数据 (Rollout)
运行你的 Agent 来生成一批包含完整交互轨迹和 Mask 的数据。
*   确保 `retriever_server.py` 已经在 GPU 4 上启动。
*   运行生成脚本 (使用 GPU 0 推理):
```bash
CUDA_VISIBLE_DEVICES=0 python3 environment/generation_no_memory.py
```
这将生成 `batch_results_no_memory.json`。

### 第二步：格式转换
将 JSON 转换为 Verl 需要的 Parquet 格式。
```bash
python3 json_to_parquet.py
```
这将生成 `data/train.parquet`。

### 第三步：启动 Verl PPO 训练 (Offline)
使用生成的数据进行训练。
*   修改 Config (`verl/trainer/config/ppo_trainer.yaml`) 中的数据路径指向 `data/train.parquet`。
*   或者在命令行指定。

```bash
# 使用 GPU 0-3 进行训练
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -m verl.trainer.main_ppo \
    data.train_files=data/train.parquet \
    data.val_files=data/train.parquet \
    actor_rollout_ref.actor.strategy=fsdp \
    trainer.n_gpus_per_node=4
```

## 3. GPU 分配详解 (0-3 训练, 4-7 辅助)

您决定使用 **GPU 0-3** 进行训练，使用 **GPU 4-7** (任选其一) 运行 Embedding/Retriever 模型。这是最标准、最不容易出错的物理隔离方案。

**注意**: Verl **不支持** HuggingFace 风格的 `device_map="auto"`。你必须通过 `CUDA_VISIBLE_DEVICES` 和 Config 显式告诉它使用哪些资源。

**分配方案**:

*   **GPU 4**: **辅助卡**
    *   运行 **Retriever Server**。
    *   这与其他进程完全隔离。

*   **GPU 0, 1, 2, 3**: **Verl 主训练区**
    *   **Actor (Rollout & Train)**: 独占这 4 张卡。
    *   **Reference Model**: 建议开启 `param_offload=True` (卸载到 CPU)，或者让它和 Actor 共享这 4 张卡。

**操作步骤**:

1.  **启动 Retriever (在 GPU 4)**:
    ```bash
    CUDA_VISIBLE_DEVICES=4 python3 environment/retriever_server.py --port 8085
    ```

2.  **配置 Verl (`config.yaml`)**:
    *   告诉 Verl 使用 4 张卡。
    ```yaml
    trainer:
      n_gpus_per_node: 4
    actor_rollout_ref:
      rollout:
        tensor_model_parallel_size: 4  # 使用 4 张卡做推理，速度最快
      ref:
        fsdp_config:
          param_offload: True          # 建议卸载到 CPU，防止显存紧张
    ```

3.  **启动训练 (限制在 GPU 0-3)**:
    ```bash
    # 只暴露 0,1,2,3 给 Verl
    export CUDA_VISIBLE_DEVICES=0,1,2,3
    python3 environment/generation_no_memory.py --mode verl_train
    ```

这样配置后，Verl 会在 GPU 0-3 上全速运行，Retriever 在 GPU 4 上安安静静地提供服务，互不干扰。

## 4. 关于 Masking (梯度屏蔽)

你提到的 "Search-R1 原文说只计算大模型生成部分的梯度"，这是对的。
在代码层面，这是通过 `DataProto` 里的 `loss_mask` 实现的。
*   **User Prompt (题目)**: loss_mask = 0
*   **Retrieved Doc (检索结果)**: loss_mask = 0
*   **Model Thought/Answer (你的回答)**: loss_mask = 1

Verl 会自动处理这个。只要你在构建数据时，把检索结果放在 `Prompt` 字段里，把模型的思考放在 `Response` 字段里，Verl 就会自动把 Prompt mask 掉。**不需要你手动写代码去 mask。**

## 5. 总结

1.  **修改 Reward**: 去 `environment/reward_utils.py` 改。
2.  **启动检索**: 先开一个终端跑 `retriever_server.py` (建议挂在 GPU 0)。
3.  **启动训练**: 再开一个终端跑 `generation_no_memory.py --mode verl_train`。
4.  **GPU**: 1张卡做推理/检索，3张卡做训练，这是合理的。
