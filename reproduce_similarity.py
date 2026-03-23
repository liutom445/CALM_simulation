from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "LLM_GPT" / "data"
MODEL_NAME = "all-MiniLM-L6-v2"
OUTPUT_CSV = DATA_DIR.parent / "reproduce_similarity_results.csv"

PHQ_LABELS = [
    "Minimal depression",
    "Mild depression",
    "Moderate depression",
    "Moderately severe depression",
    "Severe depression",
]

PHQ_RANK = {
    "Minimal depression": 1,
    "Mild depression": 2,
    "Moderate depression": 3,
    "Moderately severe depression": 4,
    "Severe depression": 5,
}

REASON_MAP = {
    "happ_bh": "brain health",
    "happ_f": "fun",
    "happ_fts": "did it for the study",
    "happ_inc": "financial incentives",
    "happ_ir": "improve relationships",
    "happ_iw": "improve work",
    "happ_m": "mood",
    "happ_mdi": "manage daily issues",
    "happ_mh": "mental health",
    "happ_o": "other",
}

FIGURE_SPECS = [
    {"column": "device", "label": "Device", "group": "App User Behavior (Structured)", "kind": "categorical"},
    {"column": "heard_about_us", "label": "Heard about us", "group": "App User Behavior (Structured)", "kind": "categorical"},
    {"column": "tell_friends", "label": "Tell friends?", "group": "App User Behavior (Structured)", "kind": "categorical"},
    {"column": "incoming_call_duration", "label": "Incoming call duration", "group": "App User Behavior (Structured)", "kind": "numeric"},
    {"column": "outgoing_call_duration", "label": "Outgoing call duration", "group": "App User Behavior (Structured)", "kind": "numeric"},
    {"column": "working", "label": "Working status", "group": "Demographics (Structured)", "kind": "categorical"},
    {"column": "gender", "label": "Sex", "group": "Demographics (Structured)", "kind": "categorical"},
    {"column": "marital_status", "label": "Marital status", "group": "Demographics (Structured)", "kind": "categorical"},
    {"column": "education", "label": "Education", "group": "Demographics (Structured)", "kind": "categorical"},
    {"column": "race", "label": "Race", "group": "Demographics (Structured)", "kind": "categorical"},
    {"column": "age", "label": "Age", "group": "Demographics (Structured)", "kind": "numeric"},
    {"column": "reason_to_enroll", "label": "Reason to enroll", "group": "Unstructured Text", "kind": "text"},
    {"column": "app_satisfaction_feedback", "label": "App satisfaction feedback", "group": "Unstructured Text", "kind": "text"},
]

GROUP_ORDER = {
    "App User Behavior (Structured)": 0,
    "Demographics (Structured)": 1,
    "Unstructured Text": 2,
}


def categorize_phq9(score):
    if pd.isna(score):
        return "Unknown"
    if score <= 4:
        return "Minimal depression"
    if score <= 9:
        return "Mild depression"
    if score <= 14:
        return "Moderate depression"
    if score <= 19:
        return "Moderately severe depression"
    return "Severe depression"


def clean_text(value):
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text in {"", "<no response>", "0", "nan", "None"}:
        return ""
    return text


def build_reason_to_enroll(row):
    labels = [REASON_MAP[col] for col in REASON_MAP if row.get(col) == 1]
    other_text = clean_text(row.get("happ_o_description"))
    if "other" in labels:
        labels = [other_text if label == "other" and other_text else label for label in labels]
    labels = [label for label in labels if label]
    return "; ".join(labels) if labels else "No response"


def combine_text_responses(series):
    parts = [clean_text(value) for value in series]
    parts = [part for part in parts if part]
    return " ".join(parts) if parts else "No response"


def load_sentence_model():
    try:
        return SentenceTransformer(MODEL_NAME, local_files_only=True)
    except Exception:
        return SentenceTransformer(MODEL_NAME)


def load_data():
    df_demo = pd.read_csv(DATA_DIR / "Baseline_Demographics.csv")
    df_reasons = pd.read_csv(DATA_DIR / "Study_App_Download_Reason.csv")
    df_satisfaction = pd.read_csv(DATA_DIR / "Study_App_Satisfaction.csv")
    df_phq9 = pd.read_csv(DATA_DIR / "PHQ_9.csv")

    baseline_reasons = (
        df_reasons.sort_values(["participant_id", "week", "dt_response"])
        .groupby("participant_id")
        .first()
        .reset_index()
    )
    baseline_reasons["reason_to_enroll"] = baseline_reasons.apply(build_reason_to_enroll, axis=1)
    baseline_reasons = baseline_reasons[["participant_id", "reason_to_enroll"]]

    app_feedback = (
        df_satisfaction.sort_values(["participant_id", "week", "dt_response"])
        .groupby("participant_id")["sat_1"]
        .apply(combine_text_responses)
        .reset_index(name="app_satisfaction_feedback")
    )

    outcomes = (
        df_phq9.sort_values(["participant_id", "week"])
        .groupby("participant_id")
        .last()
        .reset_index()
    )
    outcomes["outcome_text"] = outcomes["sum_phq9"].apply(categorize_phq9)
    outcomes["outcome_rank"] = outcomes["outcome_text"].map(PHQ_RANK)
    outcomes = outcomes[["participant_id", "outcome_text", "outcome_rank"]]

    demo_cols = [
        "participant_id",
        "working",
        "gender",
        "marital_status",
        "education",
        "race",
        "age",
        "heard_about_us",
        "device",
        "study_arm",
    ]

    df = (
        outcomes.merge(df_demo[demo_cols], on="participant_id", how="left")
        .merge(baseline_reasons, on="participant_id", how="left")
        .merge(app_feedback, on="participant_id", how="left")
    )

    text_cols = [
        "working",
        "gender",
        "marital_status",
        "education",
        "race",
        "heard_about_us",
        "device",
        "reason_to_enroll",
        "app_satisfaction_feedback",
    ]
    for col in text_cols:
        df[col] = df[col].map(lambda x: clean_text(x) or "No response")

    return df


def text_similarity(series, outcome_series, model):
    x_values = pd.Series(series, dtype="string").fillna("No response").astype(str).unique().tolist()
    y_values = pd.Series(outcome_series, dtype="string").fillna("Unknown").astype(str).unique().tolist()
    x_embeddings = model.encode(x_values, normalize_embeddings=True)
    y_embeddings = model.encode(y_values, normalize_embeddings=True)
    return float(cosine_similarity([x_embeddings.mean(axis=0)], [y_embeddings.mean(axis=0)])[0][0])


def categorical_similarity(series, outcome_series):
    x = pd.Series(series, dtype="string").fillna("No response").astype(str).to_frame(name="x")
    y = pd.Series(outcome_series, dtype="string").fillna("Unknown").astype(str).to_frame(name="y")
    x_encoder = OneHotEncoder(sparse_output=False)
    y_encoder = OneHotEncoder(sparse_output=False)
    x_matrix = x_encoder.fit_transform(x)
    y_matrix = y_encoder.fit_transform(y)
    sims = cosine_similarity(x_matrix.T, y_matrix.T)
    return float(np.mean(sims))


def numeric_similarity(series, outcome_rank):
    x = pd.to_numeric(series, errors="coerce")
    y = pd.to_numeric(outcome_rank, errors="coerce")
    keep = x.notna() & y.notna()
    x = x[keep].to_numpy(dtype=float)
    y = y[keep].to_numpy(dtype=float)
    if len(x) == 0:
        return np.nan
    x_std = np.std(x)
    y_std = np.std(y)
    if x_std == 0 or y_std == 0:
        return np.nan
    x_centered = ((x - np.mean(x)) / x_std).reshape(1, -1)
    y_centered = ((y - np.mean(y)) / y_std).reshape(1, -1)
    return float(abs(cosine_similarity(x_centered, y_centered)[0][0]))


def calculate_figure_approximation(df):
    model = load_sentence_model()
    rows = []
    missing_figure_vars = []

    for spec in FIGURE_SPECS:
        col = spec["column"]
        if col not in df.columns:
            missing_figure_vars.append(spec["label"])
            continue

        if spec["kind"] == "text":
            score = text_similarity(df[col], df["outcome_text"], model)
        elif spec["kind"] == "categorical":
            score = categorical_similarity(df[col], df["outcome_text"])
        else:
            score = numeric_similarity(df[col], df["outcome_rank"])

        rows.append(
            {
                "group": spec["group"],
                "variable": spec["label"],
                "column": col,
                "kind": spec["kind"],
                "cosine_similarity": score,
            }
        )

    results = pd.DataFrame(rows)
    results["group_order"] = results["group"].map(GROUP_ORDER)
    results["score_rank"] = results["cosine_similarity"].rank(ascending=False, method="min")
    results = results.sort_values(["group_order", "cosine_similarity"], ascending=[True, False]).reset_index(drop=True)
    return results, missing_figure_vars


def print_results(results, missing_figure_vars):
    print(f"Total participants for analysis: {results.attrs['n_participants']}")
    print("\nFigure 1(B) approximation")
    print("Method:")
    print("- Text covariates: cosine between mean MiniLM embeddings of unique responses and unique PHQ-9 outcome categories")
    print("- Categorical covariates: mean cosine between one-hot category representations and one-hot outcome representations")
    print("- Age: absolute centered cosine with ordinal PHQ-9 severity rank")

    for group, group_df in results.groupby("group", sort=False):
        print(f"\n--- {group} ---")
        for _, row in group_df.iterrows():
            print(f"{row['variable']:26}: {row['cosine_similarity']:.4f}")

    if missing_figure_vars:
        missing_text = ", ".join(missing_figure_vars)
        print(f"\nUnavailable Figure 1(B) variables in this local extract: {missing_text}")


if __name__ == "__main__":
    data = load_data()
    results, missing_figure_vars = calculate_figure_approximation(data)
    results.attrs["n_participants"] = len(data)
    print_results(results, missing_figure_vars)
    results.drop(columns=["group_order"]).to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved results to: {OUTPUT_CSV}")
