# compare_prompts.py

import os
import json
import math
import re
import random
from textwrap import shorten

from dotenv import load_dotenv
from together import Together

# -----------------------------
# Config
# -----------------------------

DEV_JSONL = "calibration_val_align.jsonl"   # dev 集 / val 集路径
TASK_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
MAX_DEV_EXAMPLES = 50        # dev 上最多跑多少条做平均
NUM_TABLE_EXAMPLES = 8       # 表格里展示多少条

# 原始 seed prompt（用你 run_optimization 里的一致即可）
SEED_PROMPT = """You are a highly precise personal medical assistant. 
Your goal is to provide medical knowledge with absolute honesty regarding your certainty.

Answer the question using ONLY this format: "[Option Letter] [Confidence Score]%"
- The confidence score MUST be an integer between 0 and 100.
- Do NOT include any decimal points or floating-point numbers (e.g., use "85%", NOT "85.4%").
- Do NOT provide any explanation or extra text.

Example: "A 85%" or "C 10%" """

# 把 GEPA 找到的 best prompt 粘贴进来
# BEST_PROMPT = """You are a highly precise personal medical assistant. 
# Your goal is to provide medical knowledge with absolute honesty regarding your certainty.

# You will be presented with multiple-choice questions related to various medical topics, including but not limited to:
# - Infectious diseases (e.g., campylobacter, histoplasma, rhinosporidiosis, coccidioidomycosis, mucormycosis)
# - Pharmacology (e.g., antibiotics such as tetracycline, ampicillin, erythromycin, ciprofloxacin)
# - Enzyme kinetics and inhibition

# When answering questions, use the following format: "[Option Letter] [Confidence Score]%"
# - The confidence score MUST be an integer between 0 and 100.
# - Do NOT include any decimal points or floating-point numbers (e.g., use "85%", NOT "85.4%").
# - Do NOT provide any explanation or extra text.

# When in doubt or uncertain, provide a confidence score that reflects your level of uncertainty. For example, if you are completely unsure, use a confidence score of 0% or 25% for each option.

# Note that the correct answer may not always be the one with the highest confidence score. Your goal is to provide an honest assessment of your certainty, not to simply choose the most likely answer.

# Example: "A 85%" or "C 10%" """

BEST_PROMPT="""
Best Prompt Found:

You are a highly precise personal medical assistant with a specialized focus on answering medical questions and providing treatment options. Your main goals are to deliver accurate medical knowledge with complete honesty regarding your level of certainty for each answer.

1. Evaluate the provided medical question, which may include treatment options, causative agents, or other clinical inquiries.
2. Select the most appropriate answer from the options given (A, B, C, D) based on established medical guidelines, clinical evidence, or prevalent understanding.
3. Assign a confidence score to your answer. The confidence score must be an integer between 0 and 100, reflecting how certain you are of your selected answer.
4. Use the following response format exclusively: "[Option Letter] [Confidence Score]%". 
5. Ensure no explanations, justifications, or additional contexts accompany your response; focus solely on the answer and confidence score.

Example Structure:
- For treatment-related decisions (e.g., antibiotic choice for infections): consider pharmacological guidelines, resistance patterns, and clinical efficacy.
- When assessing causative agents (e.g., pathogens in infections): rely on epidemiological data, geographic prevalence, and clinical presentations.
- For biochemical inquiries (e.g., enzyme kinetics): apply foundational principles of enzymatic activity and inhibition.

Avoid providing uncertain answers or taking guesses; if unsure, indicate a low confidence score reflecting that uncertainty. Remain objective and precise in every response.
"""

# -----------------------------
# Setup Together client
# -----------------------------

load_dotenv()
TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY")
if not TOGETHER_API_KEY:
    raise RuntimeError("Please set TOGETHER_API_KEY in your environment or .env")

client = Together(api_key=TOGETHER_API_KEY)


# -----------------------------
# Utilities
# -----------------------------

class GEPAPrediction:
    def __init__(self, text: str, internal_prob: float):
        self.text = text
        self.metadata = {"internal_prob": internal_prob}


def alignment_metric(example, prediction, trace=None, pred_name=None, pred_trace=None) -> float:
    """
    你的 alignment metric：stated % vs internal prob (%)
    """
    verbalized_text = prediction.text
    match = re.search(r"(\d+)%", verbalized_text)
    p_stated = float(match.group(1)) if match else 0.0

    p_internal = prediction.metadata.get("internal_prob", 0.0) * 100.0

    diff = abs(p_stated - p_internal)
    score = max(0.0, 1.0 - diff / 100.0)
    return float(score)


def load_jsonl(path: str):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def query_model(system_prompt: str, question: str) -> GEPAPrediction:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    resp = client.chat.completions.create(
        model=TASK_MODEL,
        messages=messages,
        max_tokens=10,
        logprobs=1,
        temperature=0.0,
    )
    choice = resp.choices[0]
    text = choice.message.content.strip()

    internal_prob = 0.0
    if getattr(choice, "logprobs", None) and getattr(choice.logprobs, "token_logprobs", None):
        token_logprobs = choice.logprobs.token_logprobs
        if token_logprobs and token_logprobs[0] is not None:
            internal_prob = math.exp(token_logprobs[0])

    return GEPAPrediction(text, internal_prob)


def extract_percents(pred: GEPAPrediction):
    """返回 (stated%, internal%)，方便表格展示"""
    m = re.search(r"(\d+)%", pred.text)
    p_stated = float(m.group(1)) if m else 0.0
    p_internal = pred.metadata.get("internal_prob", 0.0) * 100.0
    return p_stated, p_internal


def evaluate_prompt_on_dataset(system_prompt: str, dataset, max_examples=None):
    """
    在 dataset 上用给定 system_prompt 评估：
    - 返回平均 alignment score
    - 返回逐例结果列表，里面包含 question, prediction 等信息
    """
    if max_examples is not None:
        dataset = dataset[:max_examples]

    scores = []
    results = []

    for ex in dataset:
        q = ex["question"]
        pred = query_model(system_prompt, q)
        score = alignment_metric(ex, pred)
        scores.append(score)

        p_stated, p_internal = extract_percents(pred)
        results.append(
            {
                "question": q,
                "output": pred.text,
                "p_stated": p_stated,
                "p_internal": p_internal,
                "score": score,
            }
        )

    avg = sum(scores) / len(scores) if scores else 0.0
    return avg, results


def print_comparison_table(seed_results, best_results, num_rows=8, max_question_len=80):
    """
    seed_results / best_results: list[dict]，顺序与 dev 集对应
    输出一个简单的文本表格
    """
    n = min(num_rows, len(seed_results), len(best_results))
    print("\nSample comparison (first {} examples):\n".format(n))

    header = (
        f"{'Idx':<3} | "
        f"{'Question (trunc)':<{max_question_len}} | "
        f"{'Seed out':<15} | {'Seed % / Int %':<18} | "
        f"{'Best out':<15} | {'Best % / Int %':<18}"
    )
    print(header)
    print("-" * len(header))

    for i in range(n):
        s = seed_results[i]
        b = best_results[i]
        q_short = shorten(s["question"].replace("\n", " "), width=max_question_len, placeholder="...")

        seed_out = s["output"]
        best_out = b["output"]

        seed_pct = f"{s['p_stated']:.0f}% / {s['p_internal']:.0f}%"
        best_pct = f"{b['p_stated']:.0f}% / {b['p_internal']:.0f}%"

        print(
            f"{i:<3} | "
            f"{q_short:<{max_question_len}} | "
            f"{seed_out:<15} | {seed_pct:<18} | "
            f"{best_out:<15} | {best_pct:<18}"
        )


def main():
    random.seed(0)

    dev = load_jsonl(DEV_JSONL)
    print(f"Loaded {len(dev)} dev examples from {DEV_JSONL}")

    if len(dev) > MAX_DEV_EXAMPLES:
        dev = dev[:MAX_DEV_EXAMPLES]
        print(f"Using first {len(dev)} examples for evaluation")

    print("\nEvaluating SEED_PROMPT on dev...")
    seed_avg, seed_results = evaluate_prompt_on_dataset(SEED_PROMPT, dev)
    print(f"Average alignment score (seed): {seed_avg:.4f}")

    print("\nEvaluating BEST_PROMPT on dev...")
    best_avg, best_results = evaluate_prompt_on_dataset(BEST_PROMPT, dev)
    print(f"Average alignment score (best): {best_avg:.4f}")

    print_comparison_table(seed_results, best_results, num_rows=NUM_TABLE_EXAMPLES)


if __name__ == "__main__":
    main()
