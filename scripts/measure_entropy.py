"""
measure_entropy.py
==================
PURPOSE: Measure per-step logit entropy during normal (non-speculative) generation.
         This produces the BASELINE ENTROPY TIMESERIES needed for the motivation figure.

HOW IT WORKS:
  1. Runs llama-cli with --verbose-prompt and --log-prefix flags to dump per-step logprobs
  2. We use the --n-probs flag which makes llama-cli print the top-K token probabilities
     for each generated token — we compute entropy from those.
  3. Outputs a CSV: step, token, top_prob, entropy_approx

USAGE:
  python measure_entropy.py --model path/to/model.gguf --prompt "Your prompt here" --n-tokens 300

NOTES:
  - Use --n-probs 100 (top-100 probs) for a reasonable entropy approximation.
    Full-vocab entropy requires all logits; top-K is a lower bound but sufficient
    for the correlation study (the shape of the curve is what matters).
  - For better accuracy use a larger --n-probs value (up to vocab size ~32000).
"""

import subprocess
import re
import csv
import math
import argparse
import os
import sys

LLAMA_CLI = r"e:\Researches\research-1\llama.cpp\build\bin\Release\llama-cli.exe"


def parse_args():
    p = argparse.ArgumentParser(description="Measure per-step logit entropy from llama-cli")
    p.add_argument("--model",    required=True,  help="Path to GGUF model file")
    p.add_argument("--prompt",   default="Once upon a time in a land far away,", help="Generation prompt")
    p.add_argument("--n-tokens", type=int, default=300,  help="Number of tokens to generate")
    p.add_argument("--n-probs",  type=int, default=200,  help="Top-K probs to collect per step (higher = more accurate entropy)")
    p.add_argument("--out",      default="entropy_baseline.csv", help="Output CSV file")
    p.add_argument("--ctx-size", type=int, default=2048, help="Context window size")
    p.add_argument("--seed",     type=int, default=42,   help="RNG seed for reproducibility")
    return p.parse_args()


def run_llama_cli(model, prompt, n_tokens, n_probs, ctx_size, seed):
    """Run llama-cli and capture stdout which contains the probability lines."""
    cmd = [
        LLAMA_CLI,
        "--model",     model,
        "--prompt",    prompt,
        "--n-predict", str(n_tokens),
        "--n-probs",   str(n_probs),
        "--ctx-size",  str(ctx_size),
        "--seed",      str(seed),
        "--temp",      "0.0",          # greedy — deterministic, clean entropy measurement
        "--log-disable",               # suppress ggml debug logs to stderr
        "--no-display-prompt",         # don't echo the prompt back
    ]
    print(f"[*] Running: {' '.join(cmd[:4])} ... (n_tokens={n_tokens})")
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    return result.stdout, result.stderr


def parse_probs_from_output(raw_stdout):
    """
    llama-cli with --n-probs prints lines like:
        token '<word>' logprob=-0.123 prob=0.8845
    or in some versions:
        logprobs: [{"id":1234,"logprob":-0.12,"token":"hello"},...] after each generated token

    We detect both formats. Returns list of dicts:
        [{"step": int, "token": str, "probs": [float, ...]}, ...]
    """
    steps = []

    # Format 1: newer llama-cli outputs a JSON-like block after each token
    # Format 2: older format prints "token 'X' prob=Y" lines
    # We try to match the probability table format llama.cpp uses with --n-probs

    # llama.cpp --n-probs outputs lines like:
    #   "logprobs:" followed by JSON array, one per generated token
    json_pattern = re.compile(r'logprobs:\s*(\[.*?\])', re.DOTALL)
    matches = json_pattern.findall(raw_stdout)

    if matches:
        import json
        for step_idx, match in enumerate(matches):
            try:
                entries = json.loads(match)
                probs = [e.get("prob", math.exp(e.get("logprob", -20))) for e in entries]
                token = entries[0].get("token", "?") if entries else "?"
                steps.append({"step": step_idx, "token": token, "probs": probs})
            except Exception:
                continue
        return steps

    # Fallback: line-by-line prob format
    # Lines look like:  " 'token'  logit=X  prob=Y"
    prob_line = re.compile(r"prob\s*=\s*([0-9.eE+\-]+)")
    token_line = re.compile(r"token\s+'([^']*)'")
    current_probs = []
    current_token = None
    step_idx = 0

    for line in raw_stdout.splitlines():
        t_match = token_line.search(line)
        p_match = prob_line.search(line)

        if t_match and not current_probs:
            current_token = t_match.group(1)

        if p_match:
            current_probs.append(float(p_match.group(1)))

        # Heuristic: a blank line separates token steps
        if line.strip() == "" and current_probs:
            steps.append({"step": step_idx, "token": current_token or "?", "probs": current_probs})
            step_idx += 1
            current_probs = []
            current_token = None

    if current_probs:
        steps.append({"step": step_idx, "token": current_token or "?", "probs": current_probs})

    return steps


def compute_entropy(probs):
    """
    Shannon entropy H = -sum(p * log2(p)) in bits.
    Uses only the provided top-K probs. Treats remaining mass as a single
    uniform bucket for a lower-bound estimate.
    """
    probs = [p for p in probs if p > 0]
    if not probs:
        return 0.0
    total = sum(probs)
    # normalize (they should already sum to ~1 but may not due to top-K truncation)
    normed = [p / total for p in probs]
    h = -sum(p * math.log2(p) for p in normed if p > 0)
    return h


def main():
    args = parse_args()

    if not os.path.isfile(LLAMA_CLI):
        print(f"[ERROR] llama-cli not found at: {LLAMA_CLI}")
        print("        Edit LLAMA_CLI variable at top of script to point to your binary.")
        sys.exit(1)

    if not os.path.isfile(args.model):
        print(f"[ERROR] Model not found: {args.model}")
        sys.exit(1)

    stdout, stderr = run_llama_cli(
        args.model, args.prompt, args.n_tokens,
        args.n_probs, args.ctx_size, args.seed
    )

    if not stdout.strip():
        print("[WARN] llama-cli produced no stdout. stderr below:")
        print(stderr[:2000])
        print("\n[HINT] Try running llama-cli manually to verify the binary works.")
        sys.exit(1)

    steps = parse_probs_from_output(stdout)

    if not steps:
        print("[WARN] Could not parse probability output. Raw stdout (first 2000 chars):")
        print(stdout[:2000])
        print("\n[HINT] Make sure --n-probs is supported by your llama-cli build.")
        sys.exit(1)

    print(f"[*] Parsed {len(steps)} generation steps.")

    # Write CSV
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "token", "entropy_bits", "top1_prob", "n_probs_captured"])
        writer.writeheader()
        for s in steps:
            h = compute_entropy(s["probs"])
            top1 = max(s["probs"]) if s["probs"] else 0.0
            writer.writerow({
                "step":             s["step"],
                "token":            s["token"],
                "entropy_bits":     round(h, 4),
                "top1_prob":        round(top1, 4),
                "n_probs_captured": len(s["probs"]),
            })

    print(f"[OK] Written to: {args.out}")
    print(f"     Preview (first 5 rows):")
    with open(args.out, encoding="utf-8") as f:
        for i, line in enumerate(f):
            print("     ", line.rstrip())
            if i >= 5:
                break


if __name__ == "__main__":
    main()

