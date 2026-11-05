"""
plot_motivation.py
==================
PURPOSE: Generate the MOTIVATION FIGURE for the paper.
         Two-panel plot:
           Panel 1 (top):    Per-step logit entropy timeseries
           Panel 2 (bottom): Per-round acceptance rate at different fixed n_draft values
         
         EXPECTED RESULT: entropy and acceptance rate are anti-correlated.
         This one figure justifies the entire paper.

USAGE:
  # First run measure_entropy.py and measure_speculative.py, then:
  python plot_motivation.py \
      --entropy  entropy_baseline.csv \
      --perstep  speculative_perstep.csv \
      --sweep    speculative_sweep.csv \
      --out      motivation_figure.png

ALSO GENERATES:
  - correlation_plot.png  (scatter: entropy vs acceptance rate per round)
  - sweep_plot.png        (bar: TPS / accept% vs n_draft)
"""

import csv
import math
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--entropy",  default="entropy_baseline.csv")
    p.add_argument("--perstep", default="speculative_perstep.csv")
    p.add_argument("--sweep",   default="speculative_sweep.csv")
    p.add_argument("--out-dir", default=".", help="Output directory for plots")
    return p.parse_args()


def read_csv(path):
    if not os.path.isfile(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ─── Plot 1: Motivation Figure (2 panels) ────────────────────────────────────

def plot_motivation(entropy_rows, perstep_rows, out_path):
    """
    Two-panel figure:
      Top:    entropy timeseries (per step)
      Bottom: acceptance rate timeseries at n_draft=8 (per round)
    Shows visually that high entropy → low acceptance.
    """
    if not entropy_rows:
        print(f"[SKIP] No entropy data found, skipping motivation figure.")
        return

    steps   = [int(r["step"])          for r in entropy_rows]
    entropy = [float(r["entropy_bits"]) for r in entropy_rows]

    # Smooth entropy with a rolling window for readability
    window = 10
    entropy_smooth = np.convolve(entropy, np.ones(window)/window, mode="same")

    fig = plt.figure(figsize=(12, 7))
    gs  = gridspec.GridSpec(2, 1, hspace=0.45)

    # Panel 1: Entropy timeseries
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(steps, entropy, color="#aaaaaa", linewidth=0.6, alpha=0.5, label="Raw entropy")
    ax1.plot(steps, entropy_smooth, color="#e74c3c", linewidth=1.8, label=f"Smoothed (w={window})")
    ax1.set_xlabel("Generation step (token index)", fontsize=11)
    ax1.set_ylabel("Logit entropy (bits)", fontsize=11)
    ax1.set_title("(a)  Per-step logit entropy during generation", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Acceptance rate timeseries at a chosen n_draft value
    # Pick n_draft=8 (or the largest available)
    if perstep_rows:
        available_nd = sorted(set(int(r["n_draft"]) for r in perstep_rows))
        nd_show = 8 if 8 in available_nd else available_nd[-1]
        rows_nd = [r for r in perstep_rows if int(r["n_draft"]) == nd_show]

        token_starts  = [int(r["token_start"])  for r in rows_nd]
        accept_rates  = [float(r["accept_rate"]) for r in rows_nd]
        avg_entropy   = [float(r["avg_entropy_bits"]) for r in rows_nd]

        # Smooth
        ar_smooth = np.convolve(accept_rates, np.ones(window)/window, mode="same")

        ax2 = fig.add_subplot(gs[1])
        ax2.plot(token_starts, accept_rates,  color="#aaaaaa", linewidth=0.6, alpha=0.5, label="Raw accept rate")
        ax2.plot(token_starts, ar_smooth,     color="#2980b9", linewidth=1.8, label=f"Smoothed (w={window})")
        ax2.set_xlabel("Token position", fontsize=11)
        ax2.set_ylabel(f"Acceptance rate\n(n_draft={nd_show})", fontsize=11)
        ax2.set_title(f"(b)  Per-round acceptance rate at fixed n_draft={nd_show}", fontsize=12, fontweight="bold")
        ax2.set_ylim(0, 1.05)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)

        # Add secondary y-axis: overlay entropy
        ax2b = ax2.twinx()
        ax2b.plot(token_starts, avg_entropy, color="#e74c3c", linewidth=1.0, linestyle="--", alpha=0.6, label="Avg entropy (right)")
        ax2b.set_ylabel("Avg entropy (bits)", fontsize=10, color="#e74c3c")
        ax2b.tick_params(axis="y", labelcolor="#e74c3c")
        ax2b.legend(loc="upper right", fontsize=9)
    else:
        ax2 = fig.add_subplot(gs[1])
        ax2.text(0.5, 0.5, "No per-step speculative data.\nRun measure_speculative.py first.",
                 ha="center", va="center", fontsize=12, transform=ax2.transAxes)

    fig.suptitle("Motivation: Logit Entropy Anti-Correlates with Draft Acceptance Rate\n"
                 "(High entropy → low acceptance → fixed n_draft is suboptimal)",
                 fontsize=13, fontweight="bold", y=1.02)

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[OK] Motivation figure → {out_path}")
    plt.close()


# ─── Plot 2: Correlation scatter ─────────────────────────────────────────────

def plot_correlation(perstep_rows, out_path):
    """
    Scatter plot: avg entropy (x) vs. acceptance rate (y) per round.
    Fits a linear regression line and reports Pearson r.
    This is THE KEY RESULT that motivates the method.
    """
    if not perstep_rows:
        print("[SKIP] No per-step data for correlation plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    available_nd = sorted(set(int(r["n_draft"]) for r in perstep_rows))
    colors = plt.cm.tab10(np.linspace(0, 1, len(available_nd)))

    # Left plot: all n_draft values overlaid
    ax = axes[0]
    for nd, color in zip(available_nd, colors):
        rows_nd = [r for r in perstep_rows if int(r["n_draft"]) == nd]
        H = [float(r["avg_entropy_bits"]) for r in rows_nd]
        A = [float(r["accept_rate"])      for r in rows_nd]
        ax.scatter(H, A, s=15, alpha=0.5, color=color, label=f"n_draft={nd}")

    ax.set_xlabel("Avg logit entropy (bits)", fontsize=11)
    ax.set_ylabel("Acceptance rate", fontsize=11)
    ax.set_title("(a)  Entropy vs. Acceptance Rate\n(all n_draft values)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Right plot: single n_draft with regression line
    nd_show = 8 if 8 in available_nd else available_nd[-1]
    rows_nd = [r for r in perstep_rows if int(r["n_draft"]) == nd_show]
    H = np.array([float(r["avg_entropy_bits"]) for r in rows_nd])
    A = np.array([float(r["accept_rate"])      for r in rows_nd])

    ax = axes[1]
    ax.scatter(H, A, s=20, alpha=0.6, color="#2980b9", label=f"n_draft={nd_show}")

    if len(H) > 2:
        slope, intercept, r_val, p_val, _ = stats.linregress(H, A)
        x_line = np.linspace(H.min(), H.max(), 100)
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, color="#e74c3c", linewidth=2,
                label=f"Linear fit (r={r_val:.3f}, p={p_val:.2e})")
        print(f"[STAT] n_draft={nd_show}: Pearson r = {r_val:.3f}, p = {p_val:.2e}")

    ax.set_xlabel("Avg logit entropy (bits)", fontsize=11)
    ax.set_ylabel("Acceptance rate", fontsize=11)
    ax.set_title(f"(b)  Entropy vs. Acceptance Rate\n(n_draft={nd_show}, with regression)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Key Finding: Higher Entropy → Lower Draft Acceptance\n"
                 "(justifies entropy-adaptive draft length scheduling)",
                 fontsize=13, fontweight="bold")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[OK] Correlation figure → {out_path}")
    plt.close()


# ─── Plot 3: n_draft sweep ────────────────────────────────────────────────────

def plot_sweep(perstep_rows, out_path):
    """
    Bar chart: for each fixed n_draft value, show:
      - Overall acceptance rate
      - Total tokens generated per round (throughput proxy)
    Shows the classic "peak exists but is static" problem.
    """
    if not perstep_rows:
        print("[SKIP] No sweep data for sweep plot.")
        return

    available_nd = sorted(set(int(r["n_draft"]) for r in perstep_rows))
    accept_rates = []
    tokens_per_round = []

    for nd in available_nd:
        rows_nd = [r for r in perstep_rows if int(r["n_draft"]) == nd]
        total_drafted  = sum(int(r["n_drafted"])  for r in rows_nd)
        total_accepted = sum(int(r["n_accepted"]) for r in rows_nd)
        overall = total_accepted / total_drafted if total_drafted > 0 else 0
        avg_accepted_per_round = total_accepted / len(rows_nd) if rows_nd else 0
        accept_rates.append(overall)
        tokens_per_round.append(avg_accepted_per_round)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(available_nd))
    nd_labels = [str(nd) for nd in available_nd]

    # Acceptance rate
    ax = axes[0]
    bars = ax.bar(x, [a * 100 for a in accept_rates], color="#2980b9", width=0.5, edgecolor="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(nd_labels)
    ax.set_xlabel("Fixed n_draft", fontsize=11)
    ax.set_ylabel("Overall acceptance rate (%)", fontsize=11)
    ax.set_title("(a)  Acceptance Rate Drops as n_draft Grows\n(static n_draft cannot be optimal for all steps)", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 105)
    for bar, val in zip(bars, accept_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{val*100:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    # Tokens accepted per round (throughput proxy)
    ax = axes[1]
    bars = ax.bar(x, tokens_per_round, color="#27ae60", width=0.5, edgecolor="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(nd_labels)
    ax.set_xlabel("Fixed n_draft", fontsize=11)
    ax.set_ylabel("Avg tokens accepted per round", fontsize=11)
    ax.set_title("(b)  Effective Throughput Peaks at an Intermediate n_draft\n(adaptive scheduling finds this peak dynamically)", fontsize=11, fontweight="bold")
    for bar, val in zip(bars, tokens_per_round):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f"{val:.2f}", ha="center", va="bottom", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Static n_draft Sweep — Motivation for Adaptive Draft Scheduling",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[OK] Sweep figure → {out_path}")
    plt.close()


# ─── Plot 4: Entropy distribution histogram ───────────────────────────────────

def plot_entropy_histogram(entropy_rows, out_path):
    """
    Histogram of entropy values across all steps.
    Shows that entropy is NOT uniform — it has a wide spread,
    which is what makes adaptive draft length worth doing.
    """
    if not entropy_rows:
        print("[SKIP] No entropy data for histogram.")
        return

    entropy = [float(r["entropy_bits"]) for r in entropy_rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(entropy, bins=40, color="#e74c3c", edgecolor="black", linewidth=0.6, alpha=0.8)
    ax.axvline(np.mean(entropy),   color="#2980b9", linewidth=2, linestyle="--", label=f"Mean: {np.mean(entropy):.2f} bits")
    ax.axvline(np.median(entropy), color="#27ae60", linewidth=2, linestyle="-.",  label=f"Median: {np.median(entropy):.2f} bits")
    ax.set_xlabel("Logit entropy (bits)", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title("Distribution of Per-Step Logit Entropy\n"
                 "(Wide spread proves entropy-adaptive scheduling is non-trivial)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    std = np.std(entropy)
    ax.text(0.98, 0.95, f"σ = {std:.2f} bits", ha="right", va="top",
            transform=ax.transAxes, fontsize=11,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[OK] Entropy histogram → {out_path}")
    plt.close()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    entropy_rows  = read_csv(args.entropy)
    perstep_rows  = read_csv(args.perstep)

    print(f"[*] Loaded {len(entropy_rows)} entropy rows")
    print(f"[*] Loaded {len(perstep_rows)} per-step speculative rows")

    os.makedirs(args.out_dir, exist_ok=True)

    plot_motivation(
        entropy_rows, perstep_rows,
        os.path.join(args.out_dir, "fig1_motivation.png")
    )
    plot_correlation(
        perstep_rows,
        os.path.join(args.out_dir, "fig2_correlation.png")
    )
    plot_sweep(
        perstep_rows,
        os.path.join(args.out_dir, "fig3_sweep.png")
    )
    plot_entropy_histogram(
        entropy_rows,
        os.path.join(args.out_dir, "fig4_entropy_hist.png")
    )

    print("\n[DONE] All figures written.")
    print("       Open the figures in out_dir to inspect the results.")


if __name__ == "__main__":
    main()

