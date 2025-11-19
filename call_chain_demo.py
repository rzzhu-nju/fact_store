#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
最小可运行示例：在不考虑训练与并行的情况下，演示一次完整的“LLM→检索→继续生成→最终答案”的调用链。

约定：
- 检索模型：/data1/shares/bge-large-zh-v1.5（本地路径，兼容 BAII/bge-large-zh-v1.5）
- 生成模型：/data1/shares/Qwen2.5-7B-Instruct（本地路径，Qwen2.5 指令模型）
- 在第 5 号 GPU 上运行（脚本会设置 CUDA_VISIBLE_DEVICES=5，并将模型放到 cuda:0）。

注意：此脚本仅为流程演示，未引入任何训练、并行、服务化；检索在本地内存中的小语料上进行，直接计算余弦相似度得到 Top-K。
"""

import os
import re
import torch
import numpy as np
from typing import List, Dict, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel


# 在导入/初始化 CUDA 相关之前，限制可见 GPU 为 5 号
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "5")


def build_corpus() -> List[Dict[str, str]]:
    """构造一个最小中文示例语料，每条包含 title 与 text。
    contents 字段采用统一格式："标题"\n正文（与项目中检索服务返回格式一致）。
    """
    docs = [
        {"title": "明朝建立者", "text": "明朝由朱元璋建立，年号洪武。朱元璋即明太祖。"},
        {"title": "洪武年号含义", "text": "“洪武”为朱元璋选定的第一个年号。“洪”取自《尚书·洪范》，寓治理国家的大法；“武”体现其以武立国的开国身份与决心。"},
        {"title": "洪范九畴", "text": "《尚书·洪范》提出了治理国家的九个重要范畴，被后世奉为治世之纲。"},
        {"title": "杭州西湖", "text": "西湖位于浙江省杭州市，是中国著名的淡水湖泊与旅游景点。"},
        {"title": "阿尔法狗", "text": "AlphaGo 是 DeepMind 开发的围棋 AI，2016 年击败李世石。"},
        {"title": "二氧化碳化学式", "text": "二氧化碳的化学式为 CO2，是一种温室气体。"},
        {"title": "北京故宫", "text": "故宫又称紫禁城，是明清两代的皇家宫殿，位于北京市中心。"},
    ]
    for d in docs:
        d["contents"] = f'"{d["title"]}"\n{d["text"]}'
    return docs


def load_bge_encoder(model_dir: str):
    """加载 bge 中文大模型编码器与分词器。
    为了简单稳定，这里使用 AutoModel + CLS pooling 获取句向量；按 bge 推荐，对查询添加中文提示前缀。
    """
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_dir, trust_remote_code=True)
    model.eval()
    model.cuda()  # 放到当前可见的 cuda:0（即物理 5 号卡）
    return tokenizer, model


@torch.no_grad()
def encode_texts(tokenizer, model, texts: List[str], is_query: bool, max_length: int = 256) -> np.ndarray:
    """将一批文本编码为向量，返回 float32 的 numpy 数组。
    - bge 中文 v1.5 的推荐查询前缀（中文）："为这个句子生成检索向量: "
    - 文档向量使用 CLS pooling；做 L2 归一化，便于用点积近似余弦相似度。
    """
    if is_query:
        texts = [f"为这个句子生成检索向量: {q}" for q in texts]
    inputs = tokenizer(
        texts,
        max_length=max_length,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    inputs = {k: v.cuda() for k, v in inputs.items()}
    outputs = model(**inputs, return_dict=True)
    # CLS pooling（取 [CLS] 位向量）
    cls_emb = outputs.last_hidden_state[:, 0]
    # L2 归一化，便于余弦相似
    cls_emb = torch.nn.functional.normalize(cls_emb, dim=-1)
    emb = cls_emb.detach().cpu().numpy().astype(np.float32, order="C")
    return emb


def topk_by_cosine(query_emb: np.ndarray, doc_embs: np.ndarray, k: int) -> List[int]:
    """用点积近似余弦相似度（已 L2 归一化），返回 Top-K 文档索引。"""
    # query_emb: (1, dim), doc_embs: (N, dim)
    scores = (query_emb @ doc_embs.T)[0]  # (N,)
    idxs = scores.argsort()[::-1][:k]
    return idxs.tolist()


def format_information_block(corpus: List[Dict[str, str]], idxs: List[int]) -> str:
    """将检索到的若干文档格式化为 <information> 引用块，模仿项目中的参考文档格式。"""
    lines = []
    for i, idx in enumerate(idxs):
        content = corpus[idx]["contents"]
        title = content.split("\n")[0].strip('"')
        text = "\n".join(content.split("\n")[1:])
        lines.append(f"Doc {i+1}(Title: {title}) {text}")
    info = "\n".join(lines)
    return f"\n\n<information>{info}</information>\n\n"


def load_llm(model_dir: str):
    """加载 Qwen2.5-7B-Instruct，本地路径。"""
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        use_flash_attention_2=False
    ).eval().cuda()
    tokenizer.pad_token_id = 151643
    model.config.bos_token_id = 151643
    model.config.pad_token_id = 151643
    model.config.eos_token_id = [151645, 151643]
    return tokenizer, model


def apply_chat(tokenizer, messages: List[Dict[str, str]]) -> torch.Tensor:
    """使用聊天模板将对话消息转换为模型输入张量。"""
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors='pt',
    )
    return input_ids.cuda()


def generate_text(tokenizer, model, input_ids: torch.Tensor, max_new_tokens: int = 128) -> str:
    """执行一次生成，返回解码后的文本。"""
    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            eos_token_id=[151645, 151643],
            bos_token_id=151643,
            pad_token_id=151643,
            repetition_penalty=1.05,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
        )
    # 仅解码新增的生成部分，避免包含提示与历史消息导致解析混淆
    gen_ids = out[0][input_ids.shape[-1]:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return text


def normalize_answer_tags(text: str) -> str:
    """将常见误用标签如 (answer) 与 (/answer) 归一化为 <answer>...</answer>。"""
    text = re.sub(r"\(\s*answer\s*\)", "<answer>", text, flags=re.IGNORECASE)
    text = re.sub(r"\(\s*/\s*answer\s*\)", "</answer>", text, flags=re.IGNORECASE)
    return text


def parse_action(text: str) -> Tuple[str, str]:
    """从模型输出中解析动作与内容。
    返回 (action, content)，action ∈ {"search", "answer", None}。
    """
    # 先归一化可能的错误标签，然后匹配
    text = normalize_answer_tags(text)
    # 允许标签中存在可选空白，且优先选择最后一个动作片段
    pattern = r"<(search|answer)\s*>([\s\S]*?)</\1\s*>"
    matches = list(re.finditer(pattern, text))
    if not matches:
        return None, ""
    last = matches[-1]
    return last.group(1), last.group(2).strip()


def ensure_search_only(tokenizer, model, messages, max_new_tokens: int = 64) -> Tuple[str, str]:
    """生成并强制修正为仅输出一次 <search>...，禁止出现 <information> 或 <answer>。"""
    text = generate_text(tokenizer, model, apply_chat(tokenizer, messages), max_new_tokens=max_new_tokens)
    action, content = parse_action(text)
    violation = (action != "search") or ("<information>" in text) or ("<answer" in text) or ("(answer" in text)
    if violation:
        messages.append({"role": "system", "content": "你违反了步骤规则。现在只输出一次<search>查询词</search>，不要输出<information>或<answer>以及任何其它内容。"})
        text = generate_text(tokenizer, model, apply_chat(tokenizer, messages), max_new_tokens=max_new_tokens)
        action, content = parse_action(text)
    return (content if action == "search" else ""), text


def ensure_answer_only(tokenizer, model, messages, max_new_tokens: int = 128) -> Tuple[str, str]:
    """生成并尽量保证只输出 <answer>...，若仍不含正确标签则返回空答案与原文本。"""
    text = generate_text(tokenizer, model, apply_chat(tokenizer, messages), max_new_tokens=max_new_tokens)
    action, content = parse_action(text)
    if action != "answer":
        messages.append({"role": "system", "content": "只输出一次<answer>最终答案</answer>，不要输出其它内容。"})
        text = generate_text(tokenizer, model, apply_chat(tokenizer, messages), max_new_tokens=max_new_tokens)
        action, content = parse_action(text)
    return (content if action == "answer" else ""), text


def main():
    # 1) 加载生成模型与检索编码器（均在当前可见的 cuda:0，即物理 5 号卡）
    llm_path = "/data1/shares/Qwen2.5-7B-Instruct"
    bge_path = "/data1/shares/bge-large-zh-v1.5"
    tokenizer_llm, model_llm = load_llm(llm_path)
    tokenizer_bge, model_bge = load_bge_encoder(bge_path)

    # 2) 准备小型中文语料，并预编码为向量（演示用，实际可替换为大规模索引/服务）
    corpus = build_corpus()
    doc_texts = [d["contents"] for d in corpus]
    doc_embs = encode_texts(tokenizer_bge, model_bge, doc_texts, is_query=False)

    # 3) 构造系统与用户消息，强约束标签格式，鼓励先搜索后回答
    system_msg = (
        "你是一个具备检索工具的中文助手。严格步骤如下：\n"
        "- 你可以进行多次检索，每一步只输出一次：<search>查询词</search>，不要输出<information>或<answer>或其它文本。\n"
        "- 信息块将由系统以<information>...</information>附加到上下文，你自己不要生成<information>。\n"
        "- 只有在完成至少两次检索后，才可以输出一次：<answer>最终答案</answer>。\n"
        "请严格遵守标签格式与步骤，不要输出任何与规则无关的内容。"
    )
    
    # 创建一个需要至少两次检索的多跳问题：
    # 第一次检索获取“朱元璋的第一个年号”；第二次检索获取“洪武年号的含义与出处”（须引用典籍）。
    user_question = "请回答朱元璋在建立明朝后采用的第一个年号，并详细解释该年号的含义与出处（须引用典籍来源）。"
    user_context = "要求：必须先检索年号，再检索其含义与出处，至少进行两次检索后再给出最终答案。"

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": f"问题：{user_question}\n{user_context}"},
    ]

    # 4) 循环处理多步搜索和回答
    max_steps = 6  # 最大步长限制（允许两次以上检索）
    current_step = 0
    final_answer = None
    search_count = 0
    
    while current_step < max_steps:
        current_step += 1
        print(f"\n=== 第 {current_step} 步 ===")
        
        # 生成模型响应
        input_ids = apply_chat(tokenizer_llm, messages)
        generated_text = generate_text(tokenizer_llm, model_llm, input_ids)
        print(f"[生成结果]\n{generated_text}")
        
        # 解析动作和内容
        action, content = parse_action(generated_text)
        
        if action == "search":
            print(f"[检测到搜索] 查询内容: {content}")
            search_count += 1
            # 若模型违规生成了信息或答案，强制修正为仅<search>
            if ("<information>" in generated_text) or ("<answer" in generated_text) or ("(answer" in generated_text):
                print("[违规] 检测到额外<information>/<answer>，进行纠正...")
                corrected_query, corrected_text = ensure_search_only(tokenizer_llm, model_llm, messages)
                if corrected_query:
                    content = corrected_query
                    generated_text = corrected_text
                    print(f"[纠正后生成]\n{generated_text}")
            
            # 执行检索
            query_emb = encode_texts(tokenizer_bge, model_bge, [content], is_query=True)
            top_idxs = topk_by_cosine(query_emb, doc_embs, k=3)
            info_block = format_information_block(corpus, top_idxs)
            print(f"[检索信息]\n{info_block}")
            
            # 更新对话上下文
            messages.append({"role": "assistant", "content": generated_text})
            if search_count >= 2:
                messages.append({"role": "user", "content": f"请基于以下信息继续检索或回答：{info_block}。若信息已充分，输出一次<answer>最终答案</answer>；否则继续输出<search>查询词</search>以完善信息。"})
            else:
                messages.append({"role": "user", "content": f"请基于以下信息继续检索：{info_block}。你还需要至少 {2 - search_count} 次检索后才可以回答。"})
            
        elif action == "answer":
            if search_count < 2:
                print(f"[回答过早] 当前检索次数：{search_count}，至少需要 2 次检索后才能回答。")
                messages.append({"role": "assistant", "content": generated_text})
                messages.append({"role": "system", "content": "你至少需要进行两次检索后再输出<answer>最终答案</answer>。继续用<search>查询词</search>获取更多信息。"})
                continue
            print(f"[检测到回答] 最终答案: {content}")
            final_answer = content
            break
            
        else:
            print("[未检测到有效动作] 模型可能输出格式不正确")
            # 如果是最后一步且没有回答，强制要求回答
            if current_step == max_steps:
                print("[达到最大步长] 强制要求最终回答")
                messages.append({"role": "assistant", "content": generated_text})
                messages.append({"role": "user", "content": "请基于已有信息给出最终答案，必须使用<answer>...</answer>格式。"})
                
                # 最后一次生成
                input_ids = apply_chat(tokenizer_llm, messages)
                final_text = generate_text(tokenizer_llm, model_llm, input_ids)
                print(f"[最终生成]\n{final_text}")
                
                # 尝试解析最终答案
                final_action, final_content = parse_action(final_text)
                if final_action == "answer":
                    final_answer = final_content
                    print(f"[强制回答成功] 最终答案: {final_answer}")
                else:
                    print("[强制回答失败] 模型仍未输出正确格式")
                    final_answer = final_text
                break
    
    # 输出最终结果
    print(f"\n=== 处理完成 ===")
    if final_answer:
        print(f"[最终答案] {final_answer}")
    else:
        print("[未获得最终答案] 处理过程中未能得到有效回答")


if __name__ == "__main__":
    main()