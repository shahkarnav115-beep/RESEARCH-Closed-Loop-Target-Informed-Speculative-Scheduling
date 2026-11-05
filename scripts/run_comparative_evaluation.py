"""
run_comparative_evaluation.py
==============================
Executes a rigorous, end-to-end comparative benchmark between:
1. Static Speculative Baselines (n_draft in {1, 2, 4, 6, 8, 12})
2. CT-Spec (Closed-Loop Target-Informed Adaptive Speculative Scheduling)

Evaluated across four representative task categories:
- Code Generation (Deterministic syntax / high structural predictability)
- Math / Multi-step Reasoning (Mixed structural tokens + arithmetic logic)
- Creative Story (High-entropy vocabulary branching)
- Structured QA / Technical Explanations (Semi-structured factual content)

Outputs:
- results/comparative_benchmark_results.csv
- results/adaptation_traces.json
"""

import subprocess
import re
import csv
import json
import os
import sys
import time

LLAMA_SPECULATIVE = r"e:\Researches\research-1\llama.cpp\build-cpu\bin\Release\llama-speculative.exe"
TARGET_MODEL = r"E:\model-testing-llama-openvino\llama-3.2-1b-instruct-q4_k_m.gguf"
DRAFT_MODEL  = r"E:\model-testing-llama-openvino\llama-3.2-1b-instruct-q4_k_m.gguf"

BENCHMARK_PROMPTS = [
    {
        "category": "Code",
        "name": "quick_sort_impl",
        "prompt": "def quick_sort(arr):\n    \"\"\"Sort an array in ascending order using QuickSort algorithm.\"\"\"\n"
    },
    {
        "category": "Math_Reasoning",
        "name": "gsm8k_arithmetic",
        "prompt": "Question: A store owner bought 120 apples at $2 each and 80 oranges at $3 each. He sold 75% of the apples at $4 each and all the oranges at $5 each. Calculate the total profit step by step:"
    },
    {
        "category": "Creative_Story",
        "name": "fantasy_narrative",
        "prompt": "Write an imaginative fantasy narrative about an alchemist who discovers that twilight moonlight can be crystallized into memory glass:"
    },
    {
        "category": "Technical_QA",
        "name": "networking_protocols",
        "prompt": "Explain the architectural differences between TCP and UDP transport layer protocols in computer networks, including reliability and headers:"
    }
]

STATIC_DRAFT_VALUES = [1, 2, 4, 6, 8, 12]
ADAPTIVE_INITIAL_DRAFT = 4
N_PREDICT = 100
N_THREADS = 8

OUT_CSV = r"e:\Researches\research-1\results\comparative_benchmark_results.csv"
OUT_TRACES = r"e:\Researches\research-1\results\adaptation_traces.json"


def execute_run(prompt_text, n_draft, is_adaptive):
    env = os.environ.copy()
    env["LLAMA_SPEC_ADAPTIVE"] = "1" if is_adaptive else "0"

    cmd = [
        LLAMA_SPECULATIVE,
        "-m", TARGET_MODEL,
        "-md", DRAFT_MODEL,
        "-p", prompt_text,
        "-n", str(N_PREDICT),
        "--spec-draft-n-max", str(n_draft),
        "--temp", "0.0",
        "-t", str(N_THREADS),
    ]

    t0 = time.time()
    res = subprocess.run(cmd, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
    elapsed_wall = time.time() - t0

    out = res.stdout + "\n" + res.stderr

    # Parse standard llama-speculative metrics
    m_drafted = re.search(r"n_drafted\s*=\s*(\d+)", out)
    m_accept  = re.search(r"n_accept\s*=\s*(\d+)", out)
    m_pct     = re.search(r"accept\s*=\s*([\d\.]+)%", out)
    m_speed   = re.search(r"decoded\s+(\d+)\s+tokens.*speed:\s*([\d\.]+)\s*t/s", out)
    m_time    = re.search(r"decoded\s+\d+\s+tokens\s+in\s+([\d\.]+)\s+seconds", out)

    n_drafted = int(m_drafted.group(1)) if m_drafted else 0
    n_accept  = int(m_accept.group(1)) if m_accept else 0
    accept_pct = float(m_pct.group(1)) if m_pct else 0.0
    tokens_decoded = int(m_speed.group(1)) if m_speed else 0
    speed_tps = float(m_speed.group(2)) if m_speed else 0.0
    decode_sec = float(m_time.group(1)) if m_time else elapsed_wall

    # Parse adaptation trace points if adaptive
    adapt_trace = []
    if is_adaptive:
        trace_matches = re.findall(
            r"\[CT-Spec Adapt\]\s+rolling_acc=([\d\.]+),\s+H_t=([\d\.]+)\s+bits\s+->\s+adjusted n_draft:\s+(\d+)\s+->\s+(\d+)",
            out
        )
        for step_idx, (r_acc, h_t, prev_nd, next_nd) in enumerate(trace_matches):
            adapt_trace.append({
                "step": step_idx + 1,
                "rolling_acc": float(r_acc),
                "target_entropy": float(h_t),
                "prev_n_draft": int(prev_nd),
                "n_draft": int(next_nd)
            })

    return {
        "n_drafted": n_drafted,
        "n_accept": n_accept,
        "accept_pct": accept_pct,
        "tokens_decoded": tokens_decoded,
        "speed_tps": speed_tps,
        "decode_sec": decode_sec,
        "adapt_trace": adapt_trace,
    }


def main():
    if not os.path.isfile(LLAMA_SPECULATIVE):
        print("Error: Speculative binary not found at:", LLAMA_SPECULATIVE)
        sys.exit(1)

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    all_rows = []
    traces_dict = {}

    print("=================================================================")
    print("      STARTING CT-SPEC COMPREHENSIVE COMPARATIVE BENCHMARK       ")
    print("=================================================================")

    for task in BENCHMARK_PROMPTS:
        category = task["category"]
        prompt = task["prompt"]
        traces_dict[category] = {}

        print(f"\n>>> Running Evaluation Category: [{category}] <<<")

        # 1. Run Static Baselines
        for nd in STATIC_DRAFT_VALUES:
            print(f"  [Static Baseline] n_draft={nd:2d} ... ", end="", flush=True)
            res = execute_run(prompt, nd, is_adaptive=False)
            row = {
                "category": category,
                "strategy": "Static",
                "config_label": f"Static (n={nd})",
                "n_draft_param": nd,
                "is_adaptive": 0,
                "n_drafted": res["n_drafted"],
                "n_accept": res["n_accept"],
                "accept_pct": res["accept_pct"],
                "tokens_decoded": res["tokens_decoded"],
                "speed_tps": res["speed_tps"],
                "decode_sec": round(res["decode_sec"], 3)
            }
            all_rows.append(row)
            print(f"TPS: {res['speed_tps']:5.2f} t/s | Accept: {res['accept_pct']:5.1f}% | Time: {res['decode_sec']:5.2f}s")

        # 2. Run CT-Spec Adaptive
        print(f"  [CT-Spec Adaptive] init_n={ADAPTIVE_INITIAL_DRAFT} ... ", end="", flush=True)
        res_adapt = execute_run(prompt, ADAPTIVE_INITIAL_DRAFT, is_adaptive=True)
        row_adapt = {
            "category": category,
            "strategy": "CT-Spec",
            "config_label": "CT-Spec (Adaptive)",
            "n_draft_param": ADAPTIVE_INITIAL_DRAFT,
            "is_adaptive": 1,
            "n_drafted": res_adapt["n_drafted"],
            "n_accept": res_adapt["n_accept"],
            "accept_pct": res_adapt["accept_pct"],
            "tokens_decoded": res_adapt["tokens_decoded"],
            "speed_tps": res_adapt["speed_tps"],
            "decode_sec": round(res_adapt["decode_sec"], 3)
        }
        all_rows.append(row_adapt)
        traces_dict[category]["CT-Spec"] = res_adapt["adapt_trace"]

        print(f"TPS: {res_adapt['speed_tps']:5.2f} t/s | Accept: {res_adapt['accept_pct']:5.1f}% | Time: {res_adapt['decode_sec']:5.2f}s")
        print(f"    -> Controller steps logged: {len(res_adapt['adapt_trace'])}")

    # Save CSV
    fieldnames = [
        "category", "strategy", "config_label", "n_draft_param", "is_adaptive",
        "n_drafted", "n_accept", "accept_pct", "tokens_decoded", "speed_tps", "decode_sec"
    ]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    # Save Traces JSON
    with open(OUT_TRACES, "w", encoding="utf-8") as f:
        json.dump(traces_dict, f, indent=2)

    print("\n=================================================================")
    print(f"[COMPLETED] Benchmark successfully saved:")
    print(f" - CSV Results : {OUT_CSV}")
    print(f" - Traces JSON : {OUT_TRACES}")
    print("=================================================================")


if __name__ == "__main__":
    main()

