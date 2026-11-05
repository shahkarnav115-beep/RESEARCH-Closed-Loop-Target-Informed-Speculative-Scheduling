import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

csv_path = r"e:\Researches\research-1\results\speculative_sweep_results.csv"
out_fig = r"e:\Researches\research-1\results\figures\fig_speculative_baseline_sweep.png"

data = {}
with open(csv_path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        cat = row["category"]
        if cat not in data:
            data[cat] = {"n_draft": [], "accept_pct": [], "speed_tps": []}
        data[cat]["n_draft"].append(int(row["n_draft"]))
        data[cat]["accept_pct"].append(float(row["accept_pct"]))
        data[cat]["speed_tps"].append(float(row["speed_tps"]))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

colors = {'Story': '#e74c3c', 'Code': '#2980b9', 'Math_Reasoning': '#27ae60'}
markers = {'Story': 'o', 'Code': 's', 'Math_Reasoning': '^'}

# 1. Acceptance Rate Plot
for cat, vals in data.items():
    ax1.plot(vals['n_draft'], vals['accept_pct'], label=cat, color=colors.get(cat, 'black'),
             marker=markers.get(cat, 'o'), linewidth=2, markersize=7)

ax1.set_title('(a) Verification Acceptance Rate vs. Draft Length ($n_{draft}$)', fontsize=12, fontweight='bold')
ax1.set_xlabel('Draft Length ($n_{draft}$)', fontsize=11)
ax1.set_ylabel('Acceptance Rate (%)', fontsize=11)
ax1.set_ylim(65, 105)
ax1.legend(title="Task Category", fontsize=10)
ax1.grid(True, linestyle='--', alpha=0.6)

# 2. Decoding Throughput (TPS) Plot
for cat, vals in data.items():
    ax2.plot(vals['n_draft'], vals['speed_tps'], label=cat, color=colors.get(cat, 'black'),
             marker=markers.get(cat, 'o'), linewidth=2, markersize=7)

ax2.set_title('(b) Inference Speedup (TPS) vs. Draft Length ($n_{draft}$)', fontsize=12, fontweight='bold')
ax2.set_xlabel('Draft Length ($n_{draft}$)', fontsize=11)
ax2.set_ylabel('Generation Throughput (Tokens/s)', fontsize=11)
ax2.legend(title="Task Category", fontsize=10)
ax2.grid(True, linestyle='--', alpha=0.6)

plt.suptitle('Empirical Proof of Static $n_{draft}$ Suboptimality in Speculative Decoding', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()

os.makedirs(os.path.dirname(out_fig), exist_ok=True)
plt.savefig(out_fig, dpi=300, bbox_inches='tight')
print("Figure generated at:", out_fig)
