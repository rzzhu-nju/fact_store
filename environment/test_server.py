import requests
import json
import time
import os

def test_server():
    os.environ['no_proxy'] = 'localhost,127.0.0.1'
    os.environ['NO_PROXY'] = 'localhost,127.0.0.1'
    url = "http://localhost:8085/search"
    
    # 构造测试请求数据
    payload = {
        "queries": [
            "Who is the author of The Three-Body Problem?",
            "What is the capital of France?"
        ],
        "topk": 2
    }
    
    print(f"正在发送请求到 {url} ...")
    start_time = time.time()
    
    try:
        response = requests.post(url, json=payload)
        
        # 检查状态码
        if response.status_code == 200:
            data = response.json()
            latency = (time.time() - start_time) * 1000
            print(f"请求成功! 耗时: {latency:.2f}ms")
            print("-" * 50)
            
            # 打印结果
            results = data.get("result", [])
            scores = data.get("raw_scores", [])
            
            for i, (query, res_str) in enumerate(zip(payload["queries"], results)):
                print(f"\n查询 Q{i+1}: {query}")
                print(f"检索结果:\n{res_str}")
                if scores:
                    print(f"原始分数: {scores[i]}")
                print("-" * 30)
                
        else:
            print(f"请求失败. 状态码: {response.status_code}")
            print(f"错误信息: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("连接失败: 无法连接到服务器。请确保 retriever_server.py 正在运行并且端口是 8000。")
    except Exception as e:
        print(f"发生错误: {e}")

if __name__ == "__main__":
    test_server()