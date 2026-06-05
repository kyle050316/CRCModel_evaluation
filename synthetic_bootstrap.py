"""
Synthetic validation through the real two-list evaluation entry point.

This script differs from run_list_state_simulation.py in one important way:
it creates an explicit synthetic model_df and then uses the same matcher-based
path as real data. The CRC/bootstrap TP labels therefore come from
character matching between visible human lists and model predictions, not from
directly reading the hidden synthetic truth label.
"""

import json
import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from evaluate_two_lists_model import (
    EPS,
    MAX_ABS_WEIGHT,
    QFunction,
    TrainConfig,
    canonical_type,
    evaluate_two_lists_with_model,
    make_evaluation_table,
    normalize_text as eval_normalize_text,
    train_q_from_table,
    write_table,
)
from run_list_state_simulation import (
    INPUT_XLSX,
    load_full_terms_from_xlsx,
    relpath,
    simulate_two_lists,
)
from synthetic_pipeline import (
    BOOTSTRAP_B,
    BOOTSTRAP_BASE_SEED,
    SIM_SEED,
    load_synthetic_full_terms,
    plot_hist_two,
    simulated_match_probability,
    stable_unit_interval,
)


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "simulation_outputs" / "real_entry"
DATA_DIR = OUT_DIR / "data"
MODEL_DIR = OUT_DIR / "models" / "pubmedbert"
POINT_EVAL_DIR = OUT_DIR / "point_evaluation"
PLOT_DIR = OUT_DIR / "plots"
SUMMARY_PATH = OUT_DIR / "CRC_metrics_summary.json"

SEMANTIC_TYPES = [
    "diagnosis",
    "symptom",
    "finding",
    "medication",
    "therapy",
    "procedure",
    "imaging",
    "lab",
    "microbiology",
]


def canonicalize_terms(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["doc_id"] = out["doc_id"].astype(int)
    out["phrase"] = out["phrase"].map(eval_normalize_text)
    out["type"] = out["type"].map(canonical_type)
    out["context"] = out["context"].map(eval_normalize_text)
    return out


def simulated_type_match_probability(type_text: str) -> float:
    typ = type_text.lower()
    if any(key in typ for key in ["diagnosis", "symptom", "finding"]):
        return 0.92
    if any(key in typ for key in ["medication", "therapy", "procedure", "imaging", "lab"]):
        return 0.86
    return 0.80


def wrong_type(type_text: str) -> str:
    current = canonical_type(type_text)
    for candidate in SEMANTIC_TYPES:
        if canonical_type(candidate) != current:
            return candidate
    return "finding"


def add_model_truth_columns(full_df: pd.DataFrame) -> pd.DataFrame:
    out = full_df.copy()
    phrase_match = []
    type_match = []
    for _, row in out.iterrows():
        p_phrase = simulated_match_probability(str(row["type"]))
        phrase_ok = stable_unit_interval(row["doc_id"], row["phrase"], row["type"], "phrase_match") < p_phrase
        phrase_match.append(1 if phrase_ok else 0)
        if not phrase_ok:
            type_match.append(0)
            continue
        p_type = simulated_type_match_probability(str(row["type"]))
        type_ok = stable_unit_interval(row["doc_id"], row["phrase"], row["type"], "type_match") < p_type
        type_match.append(1 if type_ok else 0)
    out["phrase_match_truth"] = phrase_match
    out["type_match_truth"] = type_match
    out["matched_truth"] = (out["phrase_match_truth"].astype(int) * out["type_match_truth"].astype(int)).astype(int)
    return out


def build_synthetic_model_df(full_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in full_df.iterrows():
        if int(row["phrase_match_truth"]) != 1:
            continue
        pred_type = str(row["type"]) if int(row["type_match_truth"]) == 1 else wrong_type(str(row["type"]))
        rows.append(
            {
                "doc_id": int(row["doc_id"]),
                "phrase": str(row["phrase"]),
                "type": pred_type,
                "source": "synthetic_phrase_match",
            }
        )

    for doc_id in sorted(int(v) for v in full_df["doc_id"].unique()):
        n_fp = 1 + int(stable_unit_interval(doc_id, "fp") < 0.45)
        for j in range(n_fp):
            rows.append(
                {
                    "doc_id": doc_id,
                    "phrase": f"synthetic unmatched prediction {doc_id} {j + 1}",
                    "type": "finding" if j % 2 == 0 else "diagnosis",
                    "source": "synthetic_false_positive",
                }
            )
    return pd.DataFrame(rows).sort_values(["doc_id", "phrase", "type"]).reset_index(drop=True)


def split_full_terms(full_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    doc_ids = sorted(int(v) for v in full_df["doc_id"].unique())
    split = math.ceil(len(doc_ids) * 0.6)
    train_doc_ids = doc_ids[:split]
    test_doc_ids = doc_ids[split:]
    train_full = full_df[full_df["doc_id"].isin(train_doc_ids)].copy()
    test_full = full_df[full_df["doc_id"].isin(test_doc_ids)].copy()
    return train_full, test_full


def train_or_load_q(train_visible: pd.DataFrame, config: TrainConfig) -> Tuple[QFunction, Dict[str, object]]:
    q_path = MODEL_DIR / "q_function.pt"
    train_path = DATA_DIR / "train_reconstructed_visible.csv"
    write_table(train_visible, train_path)
    if q_path.exists():
        return QFunction.load(q_path), {"source": "loaded_existing_q_function", "q_function_path": relpath(q_path)}
    return train_q_from_table(train_path, MODEL_DIR, config)


def compute_estimate_from_eval_df(eval_df: pd.DataFrame, q_by_key: pd.DataFrame, pred_total: float) -> Dict[str, float]:
    work = canonicalize_terms(eval_df)
    merge_cols = ["doc_id", "phrase", "type", "context"]
    work = work.merge(q_by_key, on=merge_cols, how="left", validate="many_to_one")
    if work[["q1", "q2", "q12"]].isna().any().any():
        missing = int(work[["q1", "q2", "q12"]].isna().any(axis=1).sum())
        raise ValueError(f"Missing q predictions for {missing} visible rows")

    tp_corr = 0.0
    true_corr = 0.0
    matched = work["matched"].astype(float).to_numpy()
    states = work["state"].astype(str).to_numpy()
    for i, state in enumerate(states):
        r1 = 1 if state in {"10", "11"} else 0
        r2 = 1 if state in {"01", "11"} else 0
        r12 = 1 if state == "11" else 0
        q1 = float(work.iloc[i]["q1"])
        q2 = float(work.iloc[i]["q2"])
        q12 = float(work.iloc[i]["q12"])
        pi_hat = min(MAX_ABS_WEIGHT, max(EPS, q12 / max(EPS, q1 * q2)))
        w = (r1 / q1) + (r2 / q2) - (r12 / q12)
        w = max(-MAX_ABS_WEIGHT, min(MAX_ABS_WEIGHT, w / pi_hat))
        tp_corr += matched[i] * w
        true_corr += w

    tp_naive = float(matched.sum())
    true_naive = float(len(work))
    return {
        "n_visible_rows": int(len(work)),
        "n_matches": int(tp_naive),
        "corrected_precision": float(tp_corr / pred_total) if pred_total > 0 else 0.0,
        "corrected_recall": float(tp_corr / true_corr) if true_corr > 0 else 0.0,
        "naive_precision": float(tp_naive / pred_total) if pred_total > 0 else 0.0,
        "naive_recall": float(tp_naive / true_naive) if true_naive > 0 else 0.0,
        "corrected_tp_total": float(tp_corr),
        "corrected_true_terms_total": float(true_corr),
        "naive_tp_total": tp_naive,
        "naive_true_terms_total": true_naive,
    }


def add_match_columns(eval_df: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    out = eval_df.copy().reset_index(drop=True)
    out["phrase_match"] = out["matched"].astype(int)
    out["type_match"] = 0
    if not matches.empty:
        for _, match in matches.iterrows():
            human_row = int(match["human_row"])
            if 0 <= human_row < len(out):
                out.loc[human_row, "phrase_match"] = 1
                out.loc[human_row, "type_match"] = int(bool(match.get("type_match", False)))
    out["both_phrase_type_match"] = (out["phrase_match"].astype(int) * out["type_match"].astype(int)).astype(int)
    return out


def compute_type_accuracy_from_eval_df(eval_df: pd.DataFrame, matches: pd.DataFrame, q_by_key: pd.DataFrame) -> Dict[str, float]:
    work = add_match_columns(eval_df, matches)
    work = canonicalize_terms(work)
    merge_cols = ["doc_id", "phrase", "type", "context"]
    work = work.merge(q_by_key, on=merge_cols, how="left", validate="many_to_one")
    if work[["q1", "q2", "q12"]].isna().any().any():
        missing = int(work[["q1", "q2", "q12"]].isna().any(axis=1).sum())
        raise ValueError(f"Missing q predictions for {missing} visible rows")

    phrase_corr = 0.0
    type_corr = 0.0
    phrase = work["phrase_match"].astype(float).to_numpy()
    typ = work["both_phrase_type_match"].astype(float).to_numpy()
    states = work["state"].astype(str).to_numpy()
    for i, state in enumerate(states):
        r1 = 1 if state in {"10", "11"} else 0
        r2 = 1 if state in {"01", "11"} else 0
        r12 = 1 if state == "11" else 0
        q1 = float(work.iloc[i]["q1"])
        q2 = float(work.iloc[i]["q2"])
        q12 = float(work.iloc[i]["q12"])
        pi_hat = min(MAX_ABS_WEIGHT, max(EPS, q12 / max(EPS, q1 * q2)))
        w = (r1 / q1) + (r2 / q2) - (r12 / q12)
        w = max(-MAX_ABS_WEIGHT, min(MAX_ABS_WEIGHT, w / pi_hat))
        phrase_corr += phrase[i] * w
        type_corr += typ[i] * w

    phrase_naive = float(phrase.sum())
    type_naive = float(typ.sum())
    return {
        "n_phrase_matches": int(phrase_naive),
        "n_type_matches": int(type_naive),
        "corrected_type_accuracy": float(type_corr / phrase_corr) if phrase_corr > 0 else 0.0,
        "naive_type_accuracy": float(type_naive / phrase_naive) if phrase_naive > 0 else 0.0,
        "corrected_type_match_total": float(type_corr),
        "corrected_phrase_match_total": float(phrase_corr),
        "naive_type_match_total": type_naive,
        "naive_phrase_match_total": phrase_naive,
    }


def run_bootstrap(
    test_full: pd.DataFrame,
    test_model_df: pd.DataFrame,
    q_by_key: pd.DataFrame,
    pred_total: float,
) -> Dict[str, object]:
    prec_corr = np.zeros(BOOTSTRAP_B, dtype=np.float64)
    rec_corr = np.zeros(BOOTSTRAP_B, dtype=np.float64)
    prec_naive = np.zeros(BOOTSTRAP_B, dtype=np.float64)
    rec_naive = np.zeros(BOOTSTRAP_B, dtype=np.float64)
    type_corr = np.zeros(BOOTSTRAP_B, dtype=np.float64)
    type_naive = np.zeros(BOOTSTRAP_B, dtype=np.float64)

    for b in range(BOOTSTRAP_B):
        list1, list2, _ = simulate_two_lists(test_full, seed=BOOTSTRAP_BASE_SEED + b)
        list1 = list1[["doc_id", "phrase", "type", "context"]].copy()
        list2 = list2[["doc_id", "phrase", "type", "context"]].copy()
        phrase_eval_df, phrase_matches = make_evaluation_table(
            list1,
            list2,
            test_model_df,
            method="character",
            require_type_match=False,
        )
        precision_eval_df = add_match_columns(phrase_eval_df, phrase_matches)
        precision_eval_df["matched"] = precision_eval_df["both_phrase_type_match"]
        one = compute_estimate_from_eval_df(precision_eval_df, q_by_key, pred_total)
        type_one = compute_type_accuracy_from_eval_df(phrase_eval_df, phrase_matches, q_by_key)
        prec_corr[b] = one["corrected_precision"]
        rec_corr[b] = one["corrected_recall"]
        prec_naive[b] = one["naive_precision"]
        rec_naive[b] = one["naive_recall"]
        type_corr[b] = type_one["corrected_type_accuracy"]
        type_naive[b] = type_one["naive_type_accuracy"]

    return {
        "precision_corrected": prec_corr,
        "recall_corrected": rec_corr,
        "precision_naive": prec_naive,
        "recall_naive": rec_naive,
        "type_accuracy_corrected": type_corr,
        "type_accuracy_naive": type_naive,
    }


def run() -> Dict[str, object]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    POINT_EVAL_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    if INPUT_XLSX.exists():
        full_df = load_full_terms_from_xlsx(INPUT_XLSX)
    else:
        full_df = load_synthetic_full_terms(INPUT_XLSX)
    full_df = add_model_truth_columns(full_df)
    train_full, test_full = split_full_terms(full_df)
    all_model_df = build_synthetic_model_df(full_df)
    test_model_df = all_model_df[all_model_df["doc_id"].isin(test_full["doc_id"].unique())].copy().reset_index(drop=True)

    train_list1, train_list2, _ = simulate_two_lists(train_full, seed=SIM_SEED)
    train_visible, _ = make_evaluation_table(
        train_list1[["doc_id", "phrase", "type", "context"]],
        train_list2[["doc_id", "phrase", "type", "context"]],
        all_model_df[all_model_df["doc_id"].isin(train_full["doc_id"].unique())],
        method="character",
        require_type_match=True,
    )
    q_train_visible = train_visible.drop(columns=["matched"]).copy()

    config = TrainConfig(
        model_name="real_entry_pubmedbert",
        hidden_dim=64,
        dropout=0.35,
        lr=1e-3,
        weight_decay=5e-4,
        epochs=40,
        patience=8,
        batch_size_embed=64,
        batch_size_head=64,
        max_length=96,
        val_frac=0.2,
        seed=2026,
        device_name="cpu",
        use_context=True,
    )
    q_function, train_summary = train_or_load_q(q_train_visible, config)

    test_list1, test_list2, _ = simulate_two_lists(test_full, seed=SIM_SEED + 1)
    test_list1 = test_list1[["doc_id", "phrase", "type", "context"]].copy()
    test_list2 = test_list2[["doc_id", "phrase", "type", "context"]].copy()
    write_table(test_list1, DATA_DIR / "test_list1.csv")
    write_table(test_list2, DATA_DIR / "test_list2.csv")
    write_table(test_model_df, DATA_DIR / "test_model_predictions.csv")
    write_table(
        test_full[
            [
                "doc_id",
                "phrase",
                "type",
                "context",
                "phrase_match_truth",
                "type_match_truth",
                "matched_truth",
            ]
        ],
        DATA_DIR / "test_full_truth.csv",
    )

    point_summary = evaluate_two_lists_with_model(
        list1_df=test_list1,
        list2_df=test_list2,
        model_df=test_model_df,
        output_dir=POINT_EVAL_DIR,
        method="character",
        require_type_match=True,
        pred_total=len(test_model_df),
        q_function=q_function,
    )
    point_phrase_eval, point_phrase_matches = make_evaluation_table(
        test_list1,
        test_list2,
        test_model_df,
        method="character",
        require_type_match=False,
    )

    q_input = canonicalize_terms(test_full[["doc_id", "phrase", "type", "context"]].copy())
    q_pred = q_function.predict_dataframe(q_input.assign(state="11"))
    q_by_key = q_input.copy()
    q_by_key["q1"] = q_pred[:, 0]
    q_by_key["q2"] = q_pred[:, 1]
    q_by_key["q12"] = q_pred[:, 2]
    q_by_key = q_by_key.drop_duplicates(subset=["doc_id", "phrase", "type", "context"], keep="first")

    pred_total = float(len(test_model_df))
    tp_true = float(test_full["matched_truth"].sum())
    true_total = float(len(test_full))
    truth = {
        "precision_true": tp_true / pred_total if pred_total > 0 else 0.0,
        "recall_true": tp_true / true_total if true_total > 0 else 0.0,
        "type_accuracy_true": (
            tp_true / float(test_full["phrase_match_truth"].sum())
            if float(test_full["phrase_match_truth"].sum()) > 0
            else 0.0
        ),
        "tp_total": tp_true,
        "true_total": true_total,
        "phrase_match_total": float(test_full["phrase_match_truth"].sum()),
        "pred_total": pred_total,
        "n_model_rows": int(len(test_model_df)),
    }
    point_type_accuracy = compute_type_accuracy_from_eval_df(point_phrase_eval, point_phrase_matches, q_by_key)

    boot = run_bootstrap(test_full, test_model_df, q_by_key, pred_total)
    precision_plot = PLOT_DIR / "CRC_precision.png"
    recall_plot = PLOT_DIR / "CRC_recall.png"
    type_accuracy_plot = PLOT_DIR / "CRC_type_accuracy.png"
    plot_hist_two(
        boot["precision_corrected"],
        boot["precision_naive"],
        truth["precision_true"],
        "CRC precision",
        "Precision",
        precision_plot,
    )
    plot_hist_two(
        boot["recall_corrected"],
        boot["recall_naive"],
        truth["recall_true"],
        "CRC recall",
        "Recall",
        recall_plot,
    )
    plot_hist_two(
        boot["type_accuracy_corrected"],
        boot["type_accuracy_naive"],
        truth["type_accuracy_true"],
        "CRC type accuracy",
        "Type accuracy",
        type_accuracy_plot,
    )

    summary = {
        "method": "synthetic_model_df_through_real_entry",
        "matching_method": "character",
        "precision_recall_require_type_match": True,
        "type_accuracy_definition": "phrase and type match total / phrase match total",
        "truth": truth,
        "point_estimate_from_evaluate_two_lists_with_model": point_summary["estimate"],
        "point_type_accuracy_from_matcher": point_type_accuracy,
        "bootstrap": {
            "n_resamples": BOOTSTRAP_B,
            "corrected": {
                "precision_mean": float(boot["precision_corrected"].mean()),
                "precision_std": float(boot["precision_corrected"].std(ddof=1)),
                "recall_mean": float(boot["recall_corrected"].mean()),
                "recall_std": float(boot["recall_corrected"].std(ddof=1)),
                "type_accuracy_mean": float(boot["type_accuracy_corrected"].mean()),
                "type_accuracy_std": float(boot["type_accuracy_corrected"].std(ddof=1)),
            },
            "naive": {
                "precision_mean": float(boot["precision_naive"].mean()),
                "precision_std": float(boot["precision_naive"].std(ddof=1)),
                "recall_mean": float(boot["recall_naive"].mean()),
                "recall_std": float(boot["recall_naive"].std(ddof=1)),
                "type_accuracy_mean": float(boot["type_accuracy_naive"].mean()),
                "type_accuracy_std": float(boot["type_accuracy_naive"].std(ddof=1)),
            },
        },
        "paths": {
            "test_list1": relpath(DATA_DIR / "test_list1.csv"),
            "test_list2": relpath(DATA_DIR / "test_list2.csv"),
            "test_model_predictions": relpath(DATA_DIR / "test_model_predictions.csv"),
            "test_full_truth": relpath(DATA_DIR / "test_full_truth.csv"),
            "point_evaluation_summary": relpath(POINT_EVAL_DIR / "evaluation_summary.json"),
            "precision_hist": relpath(precision_plot),
            "recall_hist": relpath(recall_plot),
            "type_accuracy_hist": relpath(type_accuracy_plot),
            "summary": relpath(SUMMARY_PATH),
        },
        "train_summary": train_summary,
    }
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


if __name__ == "__main__":
    run()
