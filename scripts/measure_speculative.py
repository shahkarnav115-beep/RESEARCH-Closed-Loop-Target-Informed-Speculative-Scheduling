"""
measure_speculative.py
======================
PURPOSE: Run speculative decoding at multiple FIXED n_draft values and record
         per-step acceptance rate. This is the BASELINE for our method.

HOW IT WORKS:
  Uses llama-speculative.exe which already exists in the build.
  We parse its output for the lines it prints at the end:
      n_drafted = X
      n_accept  = Y
      accept    = Z%
  AND we instrument it at multiple n_draft values in a sweep.

  For per-STEP acceptance rate (needed for the correlation figure), we use
  a different approach: we simulate speculative decoding in Python using
  llama-cli's --n-probs output from BOTH a draft model and a target model,
  then apply the standard acceptance criterion r <= p_tgt / p_dft per step.

USAGE:
  python measure_speculative.py \
      --target path/to/large_model.gguf \
      --draft  path/to/small_model.gguf \
      --prompt "Your prompt here" \
      --n-tokens 300 \
      --n-draft-values 2 4 6 8 10 12 \
      --out speculative_results.csv

NOTE ON PER-STEP SIMULATION:
  We collect per-step logprob distributions from both models independently,
  then replay the speculative acceptance criterion offline. This avoids
  having to modify the C++ binary.
"""

import subprocess
import re
import csv
import math
import argparse
import os
import sys
import random
import json

LLAMA_CLI        = r"e:\Researches\research-1\llama.cpp\build\bin\Release\llama-cli.exe"
LLAMA_SPECULATIVE = r"e:\Researches\research-1\llama.cpp\build\bin\Release\llama-speculative.exe"


def parse_args():
    p = argparse.ArgumentParser(description="Sweep fixed n_draft values in speculative decoding")
    p.add_argument("--target",          required=True,  help="Path to target (large) GGUF model")
    p.add_argument("--draft",           required=True,  help="Path to draft (small) GGUF model")
    p.add_argument("--prompt",          default="Once upon a time in a land far away,")
    p.add_argument("--n-tokens",        type=int, default=300)
    p.add_argument("--n-probs",         type=int, default=200,  help="Top-K probs per step")
    p.add_argument("--n-draft-values",  type=int, nargs="+", default=[2,4,6,8,10,12])
    p.add_argument("--ctx-size",        type=int, default=2048)
    p.add_argument("--seed",            type=int, default=42)
    p.add_argument("--out-sweep",       default="speculative_sweep.csv",    help="Aggregate results CSV")
    p.add_argument("--out-perstep",     default="speculative_perstep.csv",  help="Per-step acceptance CSV (simulation mode)")
    p.add_argument("--mode",            choices=["binary","simulate"], default="simulate",
                   help="'binary'   = call llama-speculative.exe directly (needs both models); "
                        "'simulate' = collect logprobs separately then replay offline")
    return p.parse_args()


# ─── Helper: run llama-cli and collect per-step probability distributions ────

def collect_logprobs(model, prompt, n_tokens, n_probs, ctx_size, seed):
    """
    Runs llama-cli --n-probs and parses the per-step probability output.
    Returns list of {"step": int, "token_id": int, "token": str, "top_probs": [(id,prob),...]}
    """
    cmd = [
        LLAMA_CLI,
        "--model",     model,
        "--prompt",    prompt,
        "--n-predict", str(n_tokens),
        "--n-probs",   str(n_probs),
        "--ctx-size",  str(ctx_size),
        "--seed",      str(seed),
        "--temp",      "1.0",          # keep temp=1 to get realistic probs for simulation
        "--log-disable",
        "--no-display-prompt",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    return parse_logprob_output(result.stdout)


def parse_logprob_output(stdout):
    """Parse llama-cli --n-probs output into per-step distributions."""
    steps = []

    # Try JSON logprobs format first (newer llama.cpp)
    json_re = re.compile(r'logprobs:\s*(\[.*?\])', re.DOTALL)
    matches = json_re.findall(stdout)
    if matches:
        for step_idx, match in enumerate(matches):
            try:
                entries = json.loads(match)
                top_probs = []
                for e in entries:
                    prob = e.get("prob", math.exp(max(e.get("logprob", -20), -20)))
                    top_probs.append((e.get("id", -1), prob))
                token = entries[0].get("token", "?") if entries else "?"
                token_id = entries[0].get("id", -1) if entries else -1
                steps.append({
                    "step":      step_idx,
                    "token":     token,
                    "token_id":  token_id,
                    "top_probs": top_probs,
                })
            except Exception:
                continue
        return steps

    # Fallback: line-by-line format
    # (older llama.cpp prints "token 'X' prob=Y logit=Z" per candidate)
    step_data = []
    current = []
    current_token = None
    step_idx = 0

    token_re = re.compile(r"^\s*token\s+'([^']*)'\s+.*prob\s*=\s*([0-9.eE+\-]+)")
    for line in stdout.splitlines():
        m = token_re.match(line)
        if m:
            current.append((m.group(1), float(m.group(2))))
        elif line.strip() == "" and current:
            top = sorted(current, key=lambda x: -x[1])
            token = top[0][0] if top else "?"
            step_data.append({
                "step":      step_idx,
                "token":     token,
                "token_id":  -1,
                "top_probs": [(-1, p) for _, p in top],
            })
            step_idx += 1
            current = []
    return step_data


# ─── Simulation: replay speculative acceptance offline ────────────────────────

def simulate_speculative(target_steps, draft_steps, n_draft, seed=42):
    """
    Offline simulation of speculative decoding acceptance.

    Given two aligned lists of per-step distributions (target and draft),
    simulates what would happen with a fixed n_draft:
      - Draft proposes n_draft tokens
      - Target verifies each via r <= p_tgt / p_dft

    Returns list of per-round dicts:
      {"round": int, "n_drafted": int, "n_accepted": int, "accept_rate": float,
       "avg_entropy_before": float}
    """
    rng = random.Random(seed)
    n_steps = min(len(target_steps), len(draft_steps))

    results = []
    t = 0  # current token position

    while t < n_steps:
        # Draft phase: collect n_draft proposals (use next n_draft draft steps)
        draft_window = draft_steps[t : t + n_draft]
        target_window = target_steps[t : t + n_draft]
        actual_draft = min(len(draft_window), len(target_window))

        if actual_draft == 0:
            break

        # Average entropy of target logits in this window (BEFORE verification)
        avg_entropy = sum(
            compute_entropy([p for _, p in s["top_probs"]])
            for s in target_window[:actual_draft]
        ) / actual_draft

        # Acceptance check per drafted token
        n_accepted = 0
        for i in range(actual_draft):
            # Find p_tgt and p_dft for the token the draft would generate
            # (we use the draft model's top-1 token as the proposed token)
            draft_top = draft_window[i]["top_probs"]
            target_top = target_window[i]["top_probs"]

            if not draft_top:
                break

            # Proposed token = draft top-1
            proposed_id, p_dft = draft_top[0]

            # Find this token's probability in the target distribution
            p_tgt = 0.0
            for tid, tp in target_top:
                if tid == proposed_id:
                    p_tgt = tp
                    break
            # If not found in top-K, treat as very low probability
            if p_tgt == 0.0:
                p_tgt = 1e-6

            # Standard speculative acceptance: accept if r <= p_tgt / p_dft
            r = rng.random()
            ratio = p_tgt / max(p_dft, 1e-9)
            if r <= ratio:
                n_accepted += 1
            else:
                break  # reject early, stop this draft window

        results.append({
            "round":               len(results),
            "token_start":         t,
            "n_drafted":           actual_draft,
            "n_accepted":          n_accepted,
            "accept_rate":         n_accepted / actual_draft if actual_draft > 0 else 0,
            "avg_entropy_bits":    round(avg_entropy, 4),
        })

        # Advance by n_accepted+1 (the target generates one new token after any rejection)
        t += max(n_accepted + 1, 1)

    return results


def compute_entropy(probs):
    probs = [p for p in probs if p > 0]
    if not probs:
        return 0.0
    total = sum(probs)
    normed = [p / total for p in probs]
    return -sum(p * math.log2(p) for p in normed if p > 0)


# ─── Binary mode: call llama-speculative.exe directly ────────────────────────

def run_speculative_binary(target, draft, prompt, n_tokens, n_draft, ctx_size, seed):
    """Call the actual llama-speculative binary and parse aggregate stats."""
    cmd = [
        LLAMA_SPECULATIVE,
        "--model",        target,
        "--model-draft",  draft,
        "--prompt",       prompt,
        "--n-predict",    str(n_tokens),
        "--n-draft",      str(n_draft),
        "--ctx-size",     str(ctx_size),
        "--seed",         str(seed),
        "--log-disable",
        "--no-display-prompt",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    out = result.stdout + result.stderr

    n_drafted = int(re.search(r"n_drafted\s*=\s*(\d+)", out).group(1)) if re.search(r"n_drafted\s*=\s*(\d+)", out) else 0
    n_accept  = int(re.search(r"n_accept\s*=\s*(\d+)",  out).group(1)) if re.search(r"n_accept\s*=\s*(\d+)",  out) else 0
    accept_pct = float(re.search(r"accept\s*=\s*([0-9.]+)%", out).group(1)) if re.search(r"accept\s*=\s*([0-9.]+)%", out) else 0.0

    # Parse timing: "decoded X tokens in Y seconds, speed: Z t/s"
    tps_match = re.search(r"speed:\s*([0-9.]+)\s*t/s", out)
    tps = float(tps_match.group(1)) if tps_match else 0.0

    return {
        "n_draft":    n_draft,
        "n_drafted":  n_drafted,
        "n_accept":   n_accept,
        "accept_pct": accept_pct,
        "tokens_per_sec": tps,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if args.mode == "binary":
        print("[MODE] binary — calling llama-speculative.exe for each n_draft value")
        rows = []
        for nd in args.n_draft_values:
            print(f"  → n_draft={nd} ...", end="", flush=True)
            row = run_speculative_binary(
                args.target, args.draft, args.prompt,
                args.n_tokens, nd, args.ctx_size, args.seed
            )
            rows.append(row)
            print(f"  accept={row['accept_pct']:.1f}%  TPS={row['tokens_per_sec']:.1f}")

        with open(args.out_sweep, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["n_draft","n_drafted","n_accept","accept_pct","tokens_per_sec"])
            w.writeheader()
            w.writerows(rows)
        print(f"[OK] Sweep results → {args.out_sweep}")

    else:
        print("[MODE] simulate — collecting logprobs then replaying acceptance offline")
        print("[*] Collecting target model logprobs ...")
        target_steps = collect_logprobs(
            args.target, args.prompt, args.n_tokens,
            args.n_probs, args.ctx_size, args.seed
        )
        print(f"    → {len(target_steps)} steps from target model")

        print("[*] Collecting draft model logprobs ...")
        draft_steps = collect_logprobs(
            args.draft, args.prompt, args.n_tokens,
            args.n_probs, args.ctx_size, args.seed
        )
        print(f"    → {len(draft_steps)} steps from draft model")

        # Sweep n_draft values
        sweep_rows = []
        perstep_rows = []

        for nd in args.n_draft_values:
            print(f"  → Simulating n_draft={nd} ...", end="", flush=True)
            rounds = simulate_speculative(target_steps, draft_steps, nd, seed=args.seed)

            total_drafted  = sum(r["n_drafted"]  for r in rounds)
            total_accepted = sum(r["n_accepted"] for r in rounds)
            overall_accept = total_accepted / total_drafted if total_drafted > 0 else 0

            sweep_rows.append({
                "n_draft":          nd,
                "total_rounds":     len(rounds),
                "total_drafted":    total_drafted,
                "total_accepted":   total_accepted,
                "overall_accept_rate": round(overall_accept, 4),
            })
            print(f"  accept={overall_accept*100:.1f}%  rounds={len(rounds)}")

            for r in rounds:
                perstep_rows.append({
                    "n_draft":          nd,
                    "round":            r["round"],
                    "token_start":      r["token_start"],
                    "n_drafted":        r["n_drafted"],
                    "n_accepted":       r["n_accepted"],
                    "accept_rate":      round(r["accept_rate"], 4),
                    "avg_entropy_bits": r["avg_entropy_bits"],
                })

        with open(args.out_sweep, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["n_draft","total_rounds","total_drafted","total_accepted","overall_accept_rate"])
            w.writeheader()
            w.writerows(sweep_rows)

        with open(args.out_perstep, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["n_draft","round","token_start","n_drafted","n_accepted","accept_rate","avg_entropy_bits"])
            w.writeheader()
            w.writerows(perstep_rows)

        print(f"[OK] Sweep results  → {args.out_sweep}")
        print(f"[OK] Per-step data  → {args.out_perstep}")


if __name__ == "__main__":
    main()

