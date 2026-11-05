# Measurement Scripts — Entropy-Adaptive Speculative Decoding

Three scripts to generate all baseline measurements and motivation figures for the paper.

---

## Prerequisites

```
Python 3.12  (already installed)
numpy        (already installed)
matplotlib   (already installed)
scipy        (pip install scipy)
```

Install scipy:
```powershell
pip install scipy
```

---

## Step 1 — Measure Per-Step Entropy (Baseline)

```powershell
python scripts\measure_entropy.py `
    --model  "path\to\your_model.gguf" `
    --prompt "Once upon a time in a land far away," `
    --n-tokens 400 `
    --n-probs  200 `
    --out      results\entropy_baseline.csv
```

**Output:** `entropy_baseline.csv`  
Columns: `step, token, entropy_bits, top1_prob, n_probs_captured`

---

## Step 2 — Measure Speculative Acceptance Sweep

### Option A — Simulate offline (recommended, no two models needed)
```powershell
python scripts\measure_speculative.py `
    --mode simulate `
    --target "path\to\large_model.gguf" `
    --draft  "path\to\small_model.gguf" `
    --prompt "Once upon a time in a land far away," `
    --n-tokens 400 `
    --n-draft-values 2 4 6 8 10 12 `
    --out-sweep   results\speculative_sweep.csv `
    --out-perstep results\speculative_perstep.csv
```

### Option B — Use actual binary (needs both models loaded)
```powershell
python scripts\measure_speculative.py `
    --mode binary `
    --target "path\to\large_model.gguf" `
    --draft  "path\to\small_model.gguf" `
    --prompt "Once upon a time in a land far away," `
    --n-tokens 400 `
    --n-draft-values 2 4 6 8 10 12 `
    --out-sweep results\speculative_sweep.csv
```

**Output files:**
- `speculative_sweep.csv`   — aggregate stats per n_draft value
- `speculative_perstep.csv` — per-round data (round, accept_rate, avg_entropy)

---

## Step 3 — Generate All Figures

```powershell
python scripts\plot_motivation.py `
    --entropy  results\entropy_baseline.csv `
    --perstep  results\speculative_perstep.csv `
    --sweep    results\speculative_sweep.csv `
    --out-dir  results\figures
```

**Output figures:**

| File | What it shows |
|------|---------------|
| `fig1_motivation.png` | 2-panel: entropy timeseries + acceptance rate timeseries |
| `fig2_correlation.png` | Scatter: entropy vs. acceptance rate + regression |
| `fig3_sweep.png` | Bar chart: acceptance % and effective TPS per n_draft value |
| `fig4_entropy_hist.png` | Histogram of entropy values (proves non-uniform distribution) |

---

## What You Are Looking For

| Figure | Expected Result | Paper Interpretation |
|--------|----------------|----------------------|
| fig1 | Entropy spikes and acceptance rate dips at the same positions | Anti-correlation is real |
| fig2 | Negative slope, Pearson r < −0.4 (ideally < −0.6) | Entropy predicts acceptance |
| fig3 | Acceptance % decreases as n_draft grows | No single static value is optimal |
| fig4 | Wide σ (> 1.0 bit) | Entropy varies enough that adaptation is worthwhile |

---

## Recommended Models to Use

For the motivation figure, any two related models work:
- Target: `llama-3.2-3B-Q4_K_M.gguf`
- Draft:  `llama-3.2-1B-Q4_K_M.gguf`

Or any Qwen/Mistral pair with the same tokenizer.

---

## Results Directory Structure

```
results\
  entropy_baseline.csv
  speculative_sweep.csv
  speculative_perstep.csv
  figures\
    fig1_motivation.png
    fig2_correlation.png
    fig3_sweep.png
    fig4_entropy_hist.png
```

