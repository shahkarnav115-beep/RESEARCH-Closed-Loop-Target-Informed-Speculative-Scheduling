"""
plot_comparative_evaluation.py
==============================
Generates publication-ready IEEE style figures from comparative benchmark
using standard csv, json, numpy, and matplotlib.

Outputs:
1. results/figures/fig_throughput_comparison.png
2. results/figures/fig_pareto_frontier.png
3. results/figures/fig_adaptive_dynamic_trace.png
"""

import csv
import json
import os
import numpy as np
import matplotlib.pyplot as plt

CSV_PATH = r"e:\Researches\research-1\results\comparative_benchmark_results.csv"
TRACES_PATH = r"e:\Researches\research-1\results\adaptation_traces.json"
FIG_DIR = r"e:\Researches\research-1\results\figures"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.titlesize": 14,
    "axes.grid": True,
    "grid.alpha": 0.3,
})


def load_csv(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "category": r["category"],
                "strategy": r["strategy"],
                "config_label": r["config_label"],
                "n_draft_param": int(r["n_draft_param"]),
                "is_adaptive": int(r["is_adaptive"]),
                "n_drafted": int(r["n_drafted"]),
                "n_accept": int(r["n_accept"]),
                "accept_pct": float(r["accept_pct"]),
                "speed_tps": float(r["speed_tps"]),
                "decode_sec": float(r["decode_sec"]),
            })
    return rows


def plot_throughput(rows):
    categories = sorted(list(set(r["category"] for r in rows)))
    configs = [
        "Static (n=1)", "Static (n=2)", "Static (n=4)",
        "Static (n=6)", "Static (n=8)", "Static (n=12)",
        "CT-Spec (Adaptive)"
    ]
    
    # Palette
    colors = [
        "#9ecae1", "#6baed6", "#4292c6", "#2171b5", "#08519c", "#08306b",
        "#e6550d" # Highlight CT-Spec in high-contrast vermilion
    ]
    
    fig, ax = plt.subplots(figsize=(11, 5.8), dpi=300)
    
    x = np.arange(len(categories))
    width = 0.11
    
    for i, cfg in enumerate(configs):
        y_vals = []
        for cat in categories:
            match = [r for r in rows if r["category"] == cat and r["config_label"] == cfg]
            if match:
                y_vals.append(match[0]["speed_tps"])
            else:
                y_vals.append(0.0)
        
        offset = (i - len(configs) / 2) * width + width / 2
        rects = ax.bar(x + offset, y_vals, width, label=cfg, color=colors[i], edgecolor="black", linewidth=0.6)

    ax.set_title("Inference Decoding Throughput: Static Baselines vs. Proposed CT-Spec", fontweight="bold", pad=12)
    ax.set_ylabel("Throughput (Tokens / Sec)", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", " ") for c in categories], fontweight="bold")
    ax.legend(title="Decoding Strategy", bbox_to_anchor=(1.01, 1), loc="upper left", frameon=True)
    plt.tight_layout()
    
    out_path = os.path.join(FIG_DIR, "fig_throughput_comparison.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


def plot_pareto(rows):
    fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=300)
    
    categories = sorted(list(set(r["category"] for r in rows)))
    markers = {"Code": "o", "Math_Reasoning": "s", "Creative_Story": "^", "Technical_QA": "D"}
    cat_colors = {"Code": "#1f77b4", "Math_Reasoning": "#ff7f0e", "Creative_Story": "#2ca02c", "Technical_QA": "#9467bd"}
    
    for cat in categories:
        sub = [r for r in rows if r["category"] == cat]
        static_sub = sorted([r for r in sub if r["strategy"] == "Static"], key=lambda x: x["n_draft_param"])
        adapt_sub = [r for r in sub if r["strategy"] == "CT-Spec"][0]
        
        sx = [r["speed_tps"] for r in static_sub]
        sy = [r["accept_pct"] for r in static_sub]
        
        ax.plot(sx, sy, linestyle="--", alpha=0.4, color=cat_colors.get(cat, "gray"))
        ax.scatter(sx, sy, label=f"{cat.replace('_', ' ')} (Static $n$)", marker=markers.get(cat, "o"),
                   color=cat_colors.get(cat, "gray"), s=60, alpha=0.8, edgecolor="black")
        
        ax.scatter([adapt_sub["speed_tps"]], [adapt_sub["accept_pct"]],
                   label=f"{cat.replace('_', ' ')} (CT-Spec)", marker=markers.get(cat, "o"),
                   facecolors=cat_colors.get(cat, "gray"), edgecolors="red", s=180, linewidth=2.5, zorder=5)

    ax.set_title("Pareto Efficiency: Throughput vs. Verification Acceptance Rate", fontweight="bold", pad=12)
    ax.set_xlabel("Throughput (Tokens / Sec)", fontweight="bold")
    ax.set_ylabel("Speculative Acceptance Rate (%)", fontweight="bold")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)
    plt.tight_layout()
    
    out_path = os.path.join(FIG_DIR, "fig_pareto_frontier.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


def plot_traces(traces_path):
    if not os.path.isfile(traces_path):
        return
        
    with open(traces_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    
    colors = {"Code": "#1f77b4", "Math_Reasoning": "#ff7f0e", "Creative_Story": "#2ca02c", "Technical_QA": "#d62728"}
    
    has_points = False
    for cat, traces in data.items():
        ct_trace = traces.get("CT-Spec", [])
        if not ct_trace:
            continue
        has_points = True
        steps = [pt["step"] for pt in ct_trace]
        n_drafts = [pt["n_draft"] for pt in ct_trace]
        
        lbl = cat.replace("_", " ")
        ax.step(steps, n_drafts, where="post", label=lbl, color=colors.get(cat, "blue"), linewidth=2.2)
        ax.scatter(steps, n_drafts, color=colors.get(cat, "blue"), s=45, zorder=5)

    if has_points:
        ax.set_title("CT-Spec Online Adaptation Dynamics: Speculative Horizon $n_{\\mathrm{draft}}$", fontweight="bold", pad=12)
        ax.set_xlabel("Speculative Verification Round", fontweight="bold")
        ax.set_ylabel("Active Speculative Horizon ($n_{\\mathrm{draft}}$)", fontweight="bold")
        ax.set_yticks([1, 2, 4, 6, 8, 10, 12, 14, 16])
        ax.legend(title="Task Category", frameon=True)
        plt.tight_layout()
        
        out_path = os.path.join(FIG_DIR, "fig_adaptive_dynamic_trace.png")
        plt.savefig(out_path)
        plt.close()
        print(f"Saved: {out_path}")


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    rows = load_csv(CSV_PATH)
    plot_throughput(rows)
    plot_pareto(rows)
    plot_traces(TRACES_PATH)
    print("\nAll comparative figures successfully generated.")


if __name__ == "__main__":
    main()
