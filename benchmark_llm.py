import os
import time
import pandas as pd
import numpy as np
import re
from pathlib import Path
from collections import Counter
from openai import OpenAI

# ============================================================
# Configuration
# ============================================================
MODEL = "gpt-5.4-mini"
API_KEY = os.getenv("OPENAI_API_KEY")
SYNTHETIC_DATA_PATH = Path("final_synthetic_superpopulation.csv")
N_BENCHMARK = 200
FEW_SHOT_M = 14  # 7 treatment + 7 control

if not API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY before running.")

client = OpenAI(api_key=API_KEY)

# ============================================================
# Ordinal outcome mapping
# ============================================================
SEVERITY_LABELS = ["Minimal", "Mild", "Moderate", "Moderately severe", "Severe"]
SEVERITY_TO_NUM = {label: i + 1 for i, label in enumerate(SEVERITY_LABELS)}
NUM_TO_SEVERITY = {v: k for k, v in SEVERITY_TO_NUM.items()}

def numeric_to_severity(y):
    """Map continuous Y_obs to ordinal category (paper Section 5.2)."""
    y_round = int(round(y))
    return NUM_TO_SEVERITY.get(np.clip(y_round, 1, 5), "Moderate")

def extract_severity(text):
    """Extract a valid severity label from LLM output."""
    text = text.strip()
    for label in SEVERITY_LABELS:
        if label.lower() in text.lower():
            return label
    return None

def majority_vote(labels):
    """Ordinal-aware majority vote with median tiebreak."""
    counts = Counter(labels)
    if not counts:
        return None
    max_count = max(counts.values())
    modes = [k for k, v in counts.items() if v == max_count]
    if len(modes) == 1:
        return modes[0]
    indices = sorted([SEVERITY_LABELS.index(m) for m in modes])
    return SEVERITY_LABELS[indices[len(indices) // 2]]

# ============================================================
# Prompt design — minimal, no post-treatment data
# ============================================================
system_prompt = """
Predict a mental health RCT participant's follow-up PHQ-9 severity from baseline covariates and assigned treatment.
Categories (ordinal): Minimal < Mild < Moderate < Moderately severe < Severe.
Treatment effects are modest on average. Many participants remain in the same or adjacent category.
Return exactly one of: Minimal, Mild, Moderate, Moderately severe, Severe.
""".strip()

def de_standardize_age(val):
    return int(round((val * 15) + 45))

def reverse_one_hot(row, prefix):
    cols = [c for c in row.index if c.startswith(prefix + "_")]
    for col in cols:
        if row[col] == 1:
            return col.replace(prefix + "_", "").replace(".", " ")
    return "Unknown"

def builder_prompt(row, treatment):
    parts = [
        "Participant baseline profile:",
        f"- Sex (categorical): {reverse_one_hot(row, 'gender')}",
        f"- Age: {de_standardize_age(row['age'])}",
        f"- Education (categorical): {reverse_one_hot(row, 'education')}",
        f"- Enrollment reason (free text): {row.get('reason_to_enroll_recovered', 'No response')}",
        f"- Treatment (categorical): {treatment}",
    ]
    return "\n".join(parts)

# ============================================================
# Few-shot: stratified by treatment arm, no test overlap
# ============================================================
def get_few_shot_context(df_pool, n=14, seed=123):
    """Draw n/2 treated + n/2 control, convert outcome to categorical."""
    half = n // 2
    treated = df_pool[df_pool["T"] == 1].sample(n=half, random_state=seed)
    control = df_pool[df_pool["T"] == 0].sample(n=half, random_state=seed + 1)
    examples = pd.concat([treated, control]).sample(frac=1, random_state=seed + 2)

    context = []
    for _, ex in examples.iterrows():
        arm_label = "EVO/iPST (Active)" if ex["T"] == 1 else "Health Tips (Control)"
        ex_prompt = builder_prompt(ex, arm_label)
        ex_severity = numeric_to_severity(ex["Y_obs"])
        context.append(f"--- Example ---\n{ex_prompt}\nFollow-up severity: {ex_severity}")
    return "\n\n".join(context)

# ============================================================
# Prediction with self-consistency
# ============================================================
def predict_severity(system_prompt, prompt_text, n_samples=5):
    labels = []
    for k in range(n_samples):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt_text},
                ],
                temperature=0.3,
            )
            label = extract_severity(resp.choices[0].message.content)
            if label:
                labels.append(label)
        except Exception as e:
            print(f"  API error (attempt {k}): {e}")
            time.sleep(2**k)
    if not labels:
        return "PARSE_FAIL"  # Don't silently default — flag it
    return majority_vote(labels)

# ============================================================
# Main benchmark
# ============================================================
def run_benchmark(iteration_seed=123):
    df_syn = pd.read_csv(SYNTHETIC_DATA_PATH)

    # Split: test sample vs few-shot pool (no overlap)
    sample_df = df_syn.sample(n=N_BENCHMARK, random_state=iteration_seed + 1)
    pool_df = df_syn.drop(sample_df.index)

    few_shot_str = get_few_shot_context(pool_df, n=FEW_SHOT_M, seed=iteration_seed)

    results = []
    parse_fails = 0
    print(f"Benchmark: m={FEW_SHOT_M}, N={N_BENCHMARK}")

    for idx, row in sample_df.iterrows():
        prompt_y1 = f"{few_shot_str}\n\n--- Predict ---\n{builder_prompt(row, 'EVO/iPST (Active)')}"
        prompt_y0 = f"{few_shot_str}\n\n--- Predict ---\n{builder_prompt(row, 'Health Tips (Control)')}"

        y1_label = predict_severity(system_prompt, prompt_y1)
        y0_label = predict_severity(system_prompt, prompt_y0)

        if y1_label == "PARSE_FAIL" or y0_label == "PARSE_FAIL":
            parse_fails += 1

        true_severity = numeric_to_severity(row["Y_obs"])

        results.append({
            "y1_pred": y1_label,
            "y0_pred": y0_label,
            "y1_num": SEVERITY_TO_NUM.get(y1_label),
            "y0_num": SEVERITY_TO_NUM.get(y0_label),
            "y_true_severity": true_severity,
            "y_true_num": row["Y_obs"],
            "synthetic_t": row["T"],
        })

    if parse_fails:
        print(f"WARNING: {parse_fails}/{N_BENCHMARK} rows had parse failures")

    pd.DataFrame(results).to_csv("llm_benchmark_results.csv", index=False)
    print("Done. Results saved.")

if __name__ == "__main__":
    run_benchmark()