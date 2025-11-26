# Fact-Store 项目：待办事项清单 (Todo List)

以下是为使 `fact_store.py` 的实现更贴近研究论文愿景而提出的改进建议列表。

## 考虑做不做，影响roll out 效率

- [ ] **增强事实验证逻辑 (Enhance Fact Verification Logic):**
  - **目标：** 使“可验证性”检查更加严格。
  - **现状：** 验证主要基于与 `gold_facts`（标准事实库）的相似度。
  - **提议：** 实施两步验证机制。首先，使用自然语言推理 (NLI) 模型或 LLM-as-a-judge 检查断言的事实是否直接得到源文本 (`current_observation_str`) 的支持。只有通过此检查后，再与 `gold_facts` 进行比对以获得精度奖励。这能更准确地对未基于证据的事实实施“幻觉惩
  罚 (Hallucination Penalty)”。

- [ ] **优化最终答案生成的约束条件 (Refine Final Answer Generation Constraints):**
  - **目标：** 确保最终答案严格源自于 `Fact-Store`。
  - **现状：** 系统提示 (System Prompt) 指示代理这样做，但没有强制执行的机制。
  - **提议：** 增加一个生成后验证步骤。对于最终答案中的每一个主张，验证它是否可以追溯到 `visible_facts` 列表中的一个或多个事实。这将为最终输出提供一个“可验证性分数”，这个后续可以用于实验评测部分。

## 待做

- [ ] **实现黄金标准事实的预处理 (Implement Gold Standard Fact Pre-processing):**
  - **目标：** 创建一个更准确、更鲁棒的 `Gold_Fact_Store`（黄金标准事实库）。
  - **现状：** 直接使用 HotpotQA 中的原始支持性句子作为 `gold_facts`。
  - **提议：** 创建一个离线脚本，将数据集中的自然语言支持性句子转换为结构化的 `(S, R, O)` 三元组。这将使黄金标准的粒度与代理的 `assert` 动作保持一致，从而产生更精确的奖励信号。这也解决了原始句子作为强化学习目标可能过于稀疏的问题。

- [ ] **提升与强化学习训练框架的兼容性 (Increase Compatibility with RL Training Frameworks):**
  - **目标：** 重构代码，以便更容易地与标准RL库（如 TRL, Stable Baselines）集成。
  - **现状：** 代码作为一个单体脚本运行。
  - **提议：** 将 `FactStoreAgent` 及其交互重构为一个类似 Gym 的环境 (`FactStoreEnv`)。这将涉及创建 `step()` 和 `reset()` 方法，使其成为一个可用于 PPO 或 DPO 训练循环的即插即用组件。

  - [ ] **改写query的逻辑**
  - **目标：** 模型能够更好的改写query
  - **现状：** 在当前的框架下，经过测试，模型在不满意检索结构的情况下，只会小幅度的改写query甚至不改写，导致了一直检索不到的死循环，直到达到max_steps
  - **提议：**  两个解决办法：
    - 目前只用了字符串匹配 (if query in self.searched_queries) 来去重。 问题：Liu Cixin birthplace 和 place of birth of Liu Cixin 字符串不同，但语义完全一样。模型以为自己换了词，实际搜出来的内容是一样的。 解决：利用你已经加载的 E5 Embedding 模型。在执行 Search 前，计算新 Query 与历史 Query 的向量相似度。如果相似度 > 0.85 (阈值)，直接拦截并返回系统警告：“Query 语义重复，请尝试全新的关键词或实体”。
    - <think> 主要是模型自发的碎碎念。 解决：在 Prompt 中增加具体的 Query 改写策略指导，例如：“如果上次搜索失败，尝试：1. 反思关键词的选择；2. 观察observation中doc，思考改进方向”。

