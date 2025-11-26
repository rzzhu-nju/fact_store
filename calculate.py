import json

def compute_accuracy(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    total = len(data)
    correct = sum(1 for item in data if item.get("em") == 1)

    accuracy = correct / total if total > 0 else 0.0
    return accuracy, correct, total


if __name__ == "__main__":
    json_file = "batch_results.json"   # 改成你的文件路径
    acc, correct, total = compute_accuracy(json_file)
    print(f"Total: {total}")
    print(f"Correct (em=1): {correct}")
    print(f"Accuracy: {acc:.4f}")
