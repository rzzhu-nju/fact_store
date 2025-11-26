import os
import argparse
import pandas as pd


def make_user_prompt(question: str) -> str:
    rules = (
        "你是一个具备检索工具的中文助手。\n"
        "- 可以多次检索：每一步只输出一次<search>查询词</search>，不要输出<information>或<answer>或其它文本。\n"
        "- 信息块由系统以<information>...</information>加入上下文，你不要生成<information>。\n"
        "- 完成至少两次检索后，输出一次<answer>最终答案</answer>。\n"
    )
    return f"{rules}问题：{question}"


def build_rows() -> list[dict]:
    items = []
    dataset = [
        {
            "q": "请回答朱元璋在建立明朝后采用的第一个年号，并解释其含义与出处（须引用典籍来源）。",
            "answers": ["洪武"],
        },
        {
            "q": "AlphaGo 的所属公司与其在 2016 年击败的职业棋手姓名分别是什么？",
            "answers": ["DeepMind", "李世石"],
        },
    ]
    for idx, ex in enumerate(dataset):
        prompt = [
            {"role": "user", "content": make_user_prompt(ex["q"])},
        ]
        row = {
            "data_source": "nq",
            "prompt": prompt,
            "ability": "fact-reasoning",
            "reward_model": {
                "style": "rule",
                "ground_truth": {"target": ex["answers"]},
            },
            "extra_info": {
                "split": "train",
                "index": idx,
            },
        }
        items.append(row)
    return items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="data/online_ppo_demo")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    rows = build_rows()
    df = pd.DataFrame(rows)

    train_path = os.path.join(args.output_dir, "train.parquet")
    test_path = os.path.join(args.output_dir, "test.parquet")

    df.to_parquet(train_path)
    df.to_parquet(test_path)


if __name__ == "__main__":
    main()