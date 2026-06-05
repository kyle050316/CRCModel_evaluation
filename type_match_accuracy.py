"""
Bootstrap validation for type correctness conditional on phrase correctness.

Metric:
    type_given_phrase = #(phrase_match and type_match) / #(phrase_match)

CRC correction uses the same list-capture weight as precision/recall:
    corrected = sum_i w_i * both_i / sum_i w_i * phrase_i
"""

import json
import math
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from crc_functions import PUBMEDBERT_PATH, QFunction, TrainConfig, train_q_from_excel, write_table
from run_list_state_simulation import (
    DATA_DIR,
    INPUT_XLSX,
    MODEL_DIR,
    OUT_DIR,
    PLOT_DIR,
    load_full_terms_from_xlsx,
    reconstruct_visible_from_lists,
    relpath,
    simulate_two_lists,
)
from synthetic_pipeline import (
    BOOTSTRAP_B,
    BOOTSTRAP_BASE_SEED,
    EPS,
    MAX_ABS_WEIGHT,
    SIM_SEED,
    load_synthetic_full_terms,
    plot_hist_two,
    sampling_probabilities,
    stable_unit_interval,
)


Q_PATH = MODEL_DIR / "q_function.pt"
SUMMARY_PATH = OUT_DIR / "type_match_metric_summary.json"
PLOT_PATH = PLOT_DIR / "type_given_phrase_hist.png"


def simulated_type_match_probability(type_text: str) -> float:
    typ = type_text.lower()
    if any(key in typ for key in ["diagnosis", "symptom", "finding"]):
        return 0.92
    if any(key in typ for key in ["medication", "therapy", "procedure", "imaging", "lab"]):
        return 0.86
    return 0.80


def add_phrase_and_type_truth(full_df: pd.DataFrame) -> pd.DataFrame:
    out = full_df.copy()
    out["phrase_match"] = out["matched"].astype(int)
    type_matches = []
    for _, row in out.iterrows():
        if int(row["phrase_match"]) == 0:
            type_matches.append(0)
            continue
        p = simulated_type_match_probability(str(row["type"]))
        u = stable_unit_interval(row["doc_id"], row["phrase"], row["type"], "type_match")
        type_matches.append(1 if u < p else 0)
    out["type_match"] = type_matches
    out["both_phrase_type_match"] = (out["phrase_match"].astype(int) * out["type_match"].astype(int)).astype(int)
    return out


def truth_for_type_given_phrase(test_df: pd.DataFrame) -> Dict[str, float]:
    phrase_total = float(test_df["phrase_match"].sum())
    both_total = float(test_df["both_phrase_type_match"].sum())
    return {
        "type_given_phrase_true": both_total / phrase_total if phrase_total > 0 else 0.0,
        "both_phrase_type_total": both_total,
        "phrase_match_total": phrase_total,
        "n_terms": int(len(test_df)),
        "n_docs": int(test_df["doc_id"].nunique()),
    }


def bootstrap_type_given_phrase_once(test_df: pd.DataFrame, q: np.ndarray, seed: int) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    naive_both = 0.0
    naive_phrase = 0.0
    corrected_both = 0.0
    corrected_phrase = 0.0

    for i, row in test_df.iterrows():
        p1, p2 = sampling_probabilities(str(row["type"]))
        r1 = 1 if rng.random() < p1 else 0
        r2 = 1 if rng.random() < p2 else 0
        r12 = 1 if r1 == 1 and r2 == 1 else 0

        phrase = float(row["phrase_match"])
        both = float(row["both_phrase_type_match"])
        if r1 + r2 >= 1:
            naive_phrase += phrase
            naive_both += both

        q1 = float(q[i, 0])
        q2 = float(q[i, 1])
        q12 = float(q[i, 2])
        pi_hat = min(MAX_ABS_WEIGHT, max(EPS, q12 / max(EPS, q1 * q2)))
        w = 0.0
        if r1 == 1:
            w += 1.0 / q1
        if r2 == 1:
            w += 1.0 / q2
        if r12 == 1:
            w -= 1.0 / q12
        w = max(-MAX_ABS_WEIGHT, min(MAX_ABS_WEIGHT, w / pi_hat))

        corrected_phrase += phrase * w
        corrected_both += both * w

    return {
        "corrected_type_given_phrase": corrected_both / corrected_phrase if corrected_phrase > 0 else 0.0,
        "naive_type_given_phrase": naive_both / naive_phrase if naive_phrase > 0 else 0.0,
    }


def load_or_train_q(train_path: Path, config: TrainConfig):
    if Q_PATH.exists():
        return QFunction.load(Q_PATH), {"source": "loaded_existing_q_function", "q_function_path": relpath(Q_PATH)}
    return train_q_from_excel(train_path, MODEL_DIR, config)


def run() -> Dict[str, object]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    if INPUT_XLSX.exists():
        full_df = load_full_terms_from_xlsx(INPUT_XLSX)
    else:
        full_df = load_synthetic_full_terms(INPUT_XLSX)
    full_df = add_phrase_and_type_truth(full_df)

    doc_ids = sorted(int(v) for v in full_df["doc_id"].unique())
    split = math.ceil(len(doc_ids) * 0.6)
    train_doc_ids = doc_ids[:split]
    test_doc_ids = doc_ids[split:]

    train_full = full_df[full_df["doc_id"].isin(train_doc_ids)].copy()
    test_full = full_df[full_df["doc_id"].isin(test_doc_ids)].copy()

    train_list1, train_list2, _ = simulate_two_lists(train_full, seed=SIM_SEED)
    train_visible = reconstruct_visible_from_lists(train_list1, train_list2)
    train_path = DATA_DIR / "train_reconstructed_visible.csv"
    write_table(train_visible, train_path)

    config = TrainConfig(
        model_name="list_state_pubmedbert",
        model_path=PUBMEDBERT_PATH,
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
    q_function, train_summary = load_or_train_q(train_path, config)

    pred_input = test_full[["doc_id", "phrase", "type", "context", "matched"]].copy()
    pred_input["state"] = "11"
    pred_input = pred_input[["doc_id", "phrase", "type", "state", "context", "matched"]]
    q = q_function.predict_dataframe(pred_input)

    truth = truth_for_type_given_phrase(test_full)
    corrected = np.zeros(BOOTSTRAP_B, dtype=np.float64)
    naive = np.zeros(BOOTSTRAP_B, dtype=np.float64)
    test_full_reset = test_full.reset_index(drop=True)
    for b in range(BOOTSTRAP_B):
        one = bootstrap_type_given_phrase_once(test_full_reset, q, BOOTSTRAP_BASE_SEED + b)
        corrected[b] = one["corrected_type_given_phrase"]
        naive[b] = one["naive_type_given_phrase"]

    plot_hist_two(
        corrected,
        naive,
        truth["type_given_phrase_true"],
        "Type correctness among phrase matches",
        "Type given phrase match",
        PLOT_PATH,
    )

    type_truth_path = DATA_DIR / "test_type_match_truth.csv"
    write_table(
        test_full[
            [
                "doc_id",
                "phrase",
                "type",
                "context",
                "phrase_match",
                "type_match",
                "both_phrase_type_match",
            ]
        ],
        type_truth_path,
    )

    summary = {
        "method": "crc_type_given_phrase",
        "definition": {
            "ground_truth": "sum(phrase_match * type_match) / sum(phrase_match)",
            "crc_corrected": "sum(w * phrase_match * type_match) / sum(w * phrase_match)",
            "naive": "sum(visible * phrase_match * type_match) / sum(visible * phrase_match)",
        },
        "truth": truth,
        "bootstrap": {
            "n_resamples": BOOTSTRAP_B,
            "corrected": {
                "mean": float(corrected.mean()),
                "std": float(corrected.std(ddof=1)),
            },
            "naive": {
                "mean": float(naive.mean()),
                "std": float(naive.std(ddof=1)),
            },
        },
        "paths": {
            "plot": relpath(PLOT_PATH),
            "test_type_match_truth": relpath(type_truth_path),
            "q_function": relpath(Q_PATH),
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
