"""
sweep_speculative_baseline.py
=============================
Runs fixed n_draft sweeps using the newly compiled llama-speculative.exe
and generates the throughput and acceptance rate comparison CSV.
"""

import subprocess
import re
import csv
import os
import sys

LLAMA_SPECULATIVE = r"e:\Researches\research-1\llama.cpp\build-cpu\bin\Release\llama-speculative.exe"

TARGET_MODEL = r"E:\model-testing-llama-openvino\llama-3.2-1b-instruct-q4_k_m.gguf"
# For homogeneous testing or small draft model, self-speculation / prompt evaluation:
DRAFT_MODEL  = r"E:\model-testing-llama-openvino\llama-3.2-1b-instruct-q4_k_m.gguf"

PROMPTS = [
    ("Story", "Once upon a time in a forgotten forest, an ancient tree began to whisper secrets to the wind."),
    ("Code", "def quick_sort(arr):\n    \"\"\"Sort an array in ascending order using QuickSort algorithm.\"\"\"\n"),
    ("Math_Reasoning", "Question: If a store sells 15 apples at $2 each and 20 oranges at $3 each, what is the total revenue and profit margin if total cost is $50? Let's solve step by step:"),
]

N_DRAFT_VALUES = [1, 2, 4, 6, 8, 12]
N_PREDICT = 100
OUT_CSV = r"e:\Researches\research-1\results\speculative_sweep_results.csv"


def run_speculative(target, draft, prompt, n_predict, n_draft):
    cmd = [
        LLAMA_SPECULATIVE,
        "-m", target,
        "-md", draft,
        "-p", prompt,
        "-n", str(n_predict),
        "--spec-draft-n-max", str(n_draft),
        "--temp", "0.0",  # greedy deterministic verification
        "-t", "8",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    output = res.stdout + "\n" + res.stderr

    # Parse metrics
    # n_drafted = X, n_accept = Y, accept = Z%
    # decoded X tokens in Y seconds, speed: Z t/s
    m_drafted = re.search(r"n_drafted\s*=\s*(\d+)", output)
    m_accept  = re.search(r"n_accept\s*=\s*(\d+)", output)
    m_pct     = re.search(r"accept\s*=\s*([\d\.]+)%", output)
    m_speed   = re.search(r"decoded\s+\d+\s+tokens.*speed:\s*([\d\.]+)\s*t/s", output)

    n_drafted = int(m_drafted.group(1)) if m_drafted else 0
    n_accept  = int(m_accept.group(1)) if m_accept else 0
    accept_pct = float(m_pct.group(1)) if m_pct else 0.0
    speed_tps = float(m_speed.group(1)) if m_speed else 0.0

    return {
        "n_draft": n_draft,
        "n_drafted": n_drafted,
        "n_accept": n_accept,
        "accept_pct": accept_pct,
        "speed_tps": speed_tps,
    }


def main():
    if not os.path.isfile(LLAMA_SPECULATIVE):
        print("Binary not found:", LLAMA_SPECULATIVE)
        sys.exit(1)

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    all_rows = []

    for category, prompt in PROMPTS:
        print(f"\n==========================================")
        print(f"Benchmarking prompt category: {category}")
        print(f"==========================================")

        for nd in N_DRAFT_VALUES:
            print(f"[*] Testing n_draft = {nd} ...", end="", flush=True)
            res = run_speculative(TARGET_MODEL, DRAFT_MODEL, prompt, N_PREDICT, nd)
            row = {
                "category": category,
                "n_draft": nd,
                "n_drafted": res["n_drafted"],
                "n_accept": res["n_accept"],
                "accept_pct": res["accept_pct"],
                "speed_tps": res["speed_tps"]
            }
            all_rows.append(row)
            print(f" -> Accept: {res['accept_pct']:.1f}% | TPS: {res['speed_tps']:.2f} t/s")

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["category", "n_draft", "n_drafted", "n_accept", "accept_pct", "speed_tps"])
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\n[SUCCESS] Baseline sweep saved to: {OUT_CSV}")


if __name__ == "__main__":
    main()

