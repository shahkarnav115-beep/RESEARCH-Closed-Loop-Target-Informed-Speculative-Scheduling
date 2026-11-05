# Zenodo Deposit Metadata & Step-by-Step Publishing Guide

**Target Deposit:** Preprint Publication with Reproducible Benchmark Artifacts  
**Zenodo URL:** [https://zenodo.org/deposit/new](https://zenodo.org/deposit/new)

---

## 1. Files to Upload

Upload the following files to your new Zenodo deposit:
1. **`CT-Spec-Paper.pdf`**: The compiled camera-ready PDF (compile via Overleaf by uploading `CT-Spec-Paper-LaTeX-Source.zip` or local TeX).
2. **`CT-Spec-Benchmark-Artifacts.zip`**: Contains the raw comparative benchmark CSV (`comparative_benchmark_results.csv`), JSON adaptation traces, and execution scripts.
3. **`CT-Spec-Paper-LaTeX-Source.zip`**: (Optional) The full LaTeX source package.

Both zip archives are located in your working directory:
- `e:\Researches\research-1\CT-Spec-Paper-LaTeX-Source.zip`
- `e:\Researches\research-1\CT-Spec-Benchmark-Artifacts.zip`

---

## 2. Zenodo Form Fields (Copy-Paste Ready)

### Resource Type
- **Resource type:** `Publication`
- **Publication type:** `Preprint`

### Basic Information
- **Title:**
  ```text
  CT-Spec: Closed-Loop Target-Informed Speculative Scheduling for Accelerated Edge LLM Inference
  ```

- **Authors:**
  - **Family name:** `Shah`
  - **Given names:** `Karnav Jayshkumar`
  - **Affiliation:** `[Your Institution / Department]`
  - **ORCID:** `[Optional: e.g. 0000-000X-XXXX-XXXX]`

- **Publication date:** `2026-11-05`

- **Description / Abstract (HTML/Markdown):**
  ```html
  <p>Speculative decoding has emerged as a predominant paradigm to circumvent the memory-bandwidth bottleneck of autoregressive Large Language Model (LLM) inference by interleaving lightweight candidate draft generation with parallel target model verification. However, production inference engines—such as <code>llama.cpp</code> and <code>vLLM</code>—employ static speculative draft horizons (<em>n</em><sub>draft</sub>). In this paper, we show that static scheduling introduces a severe efficiency dilemma: shallow horizons (<em>n</em> &le; 4) starve target compute during deterministic syntax generation, whereas deep horizons (<em>n</em> &ge; 8) incur crippling compute waste and rejection penalties on high-entropy creative branching, degrading verification acceptance down to 75.8%.</p>

  <p>Existing adaptive heuristics rely strictly on open-loop draft-model confidence, suffering from <em>draft blindness</em> where small models remain confidently misaligned with the target distribution. To overcome this limitation, we present <strong>CT-Spec</strong> (Closed-Loop Target-Informed Speculative Scheduling), an online stochastic control framework integrated natively into the core C++ decoding loop of <code>llama.cpp</code>. CT-Spec establishes a closed feedback loop: it dynamically tracks ground-truth target verification acceptance rates &alpha;<sub>t</sub> and target logit entropy <em>H</em>(<em>P</em><sub>tgt</sub>) to modulate the speculative horizon in real time (<em>n</em><sub>draft</sub> &isin; [1, 16]) with zero per-token drafting overhead.</p>

  <p>Extensive empirical evaluations across diverse cognitive domains (Code, Multi-Step Reasoning, Creative Storytelling, and Technical QA) demonstrate that CT-Spec eliminates over-drafting penalties, attaining up to <strong>28.62 tokens/s (a 49.3% speedup over <em>n</em>=1 and 29.3% faster than <em>n</em>=12)</strong> while sustaining 100% acceptance without requiring offline profiling or auxiliary model training.</p>
  ```

- **License:**
  `Creative Commons Attribution 4.0 International (CC-BY-4.0)`

### Keywords / Tags
Add the following tags (press Enter after each):
- `Large Language Models`
- `Speculative Decoding`
- `Edge AI`
- `Inference Acceleration`
- `Adaptive Scheduling`
- `Stochastic Control`
- `llama.cpp`
- `Systems for Machine Learning`

### Language
- `English`

---

## 3. How to Obtain the PDF via Overleaf (2-Minute Workflow)

1. Go to [https://www.overleaf.com/project](https://www.overleaf.com/project)
2. Click **New Project** &rarr; **Upload Project**.
3. Drag and drop `e:\Researches\research-1\CT-Spec-Paper-LaTeX-Source.zip`.
4. Overleaf will unpack `main.tex` and the `figures/` folder automatically.
5. Click **Recompile**, then click the **Download PDF** icon next to the Recompile button.
6. Upload that PDF directly into your Zenodo deposit!

---

## 4. After Publishing on Zenodo

Zenodo will immediately mint a permanent **DOI** (e.g. `10.5281/zenodo.XXXXXXX`).

You can then cite your paper anywhere using standard BibTeX:

```bibtex
@article{shah2026ctspec,
  author       = {Karnav Jayshkumar Shah},
  title        = {{CT-Spec: Closed-Loop Target-Informed Speculative Scheduling for Accelerated Edge LLM Inference}},
  journal      = {Zenodo Preprint},
  year         = {2026},
  month        = nov,
  doi          = {10.5281/zenodo.YOUR_ASSIGNED_DOI},
  url          = {https://doi.org/10.5281/zenodo.YOUR_ASSIGNED_DOI}
}
```

