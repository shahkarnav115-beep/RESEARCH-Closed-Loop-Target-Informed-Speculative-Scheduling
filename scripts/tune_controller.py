path = r"e:\Researches\research-1\llama.cpp\examples\speculative\speculative.cpp"
with open(path, "r", encoding="utf-8") as f:
    code = f.read()

# Make the rolling rate reflect real batch accept percentage directly
old_acc_update = """            if (round_drafted > 0) {
                float cur_round_acc = (float)round_accepted / (float)round_drafted;
                rolling_accept_rate = (1.0f - momentum) * rolling_accept_rate + momentum * cur_round_acc;
            }"""

new_acc_update = """            if (round_drafted > 0) {
                float cur_round_acc = (float)round_accepted / (float)round_drafted;
                // Fast-reacting EMA for speculative rounds
                rolling_accept_rate = 0.5f * rolling_accept_rate + 0.5f * cur_round_acc;
            }"""

old_adapt = """            // Closed-loop adaptive control formula
            int prev_n_draft = n_draft;
            if (rolling_accept_rate >= 0.80f) {
                n_draft = std::min(n_draft_max, n_draft + 2);
            } else if (rolling_accept_rate < 0.60f) {
                n_draft = std::max(n_draft_min, n_draft - 2);
            }
            fprintf(stdout, "\\n[CT-Spec Adapt] rolling_acc=%.2f, H_t=%.2f bits -> adjusted n_draft: %d -> %d\\n",
                    rolling_accept_rate, last_target_entropy, prev_n_draft, n_draft);
            fflush(stdout);"""

new_adapt = """            // Closed-loop adaptive control formula
            int prev_n_draft = n_draft;
            if (rolling_accept_rate >= 0.75f) {
                n_draft = std::min(n_draft_max, n_draft + 2);
            } else if (rolling_accept_rate < 0.50f) {
                n_draft = std::max(n_draft_min, n_draft - 2);
            }
            fprintf(stdout, "\\n[CT-Spec Adapt] rolling_acc=%.2f, H_t=%.2f bits -> adjusted n_draft: %d -> %d\\n",
                    rolling_accept_rate, last_target_entropy, prev_n_draft, n_draft);
            fflush(stdout);"""

assert old_acc_update in code, "old_acc_update not found"
code = code.replace(old_acc_update, new_acc_update, 1)

assert old_adapt in code, "old_adapt not found"
code = code.replace(old_adapt, new_adapt, 1)

with open(path, "w", encoding="utf-8") as f:
    f.write(code)
print("Updated fast-reacting controller parameters!")
