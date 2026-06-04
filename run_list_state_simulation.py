import json
import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from crc_functions import (
    PUBMEDBERT_PATH,
    TrainConfig,
    build_state_table_from_two_lists,
    draw_metric_barplot,
    train_q_from_excel,
    write_table,
)
from synthetic_pipeline import (
    BOOTSTRAP_B,
    BOOTSTRAP_BASE_SEED,
    SIM_SEED,
    bootstrap_once,
    normalize_text,
    plot_hist_two,
    read_xlsx_sheet,
    sampling_probabilities,
    simulated_match_probability,
    stable_unit_interval,
    synthetic_pred_totals,
    truth_for_docs,
)

ROOT = Path(__file__).resolve().parent
INPUT_XLSX = ROOT / "mimic_iii_synthetic_term_extraction_50_long_full_context-2 2.xlsx"
OUT_DIR = ROOT / "simulation_outputs"
DATA_DIR = OUT_DIR / "data"
MODEL_DIR = OUT_DIR / "models" / "pubmedbert"
PLOT_DIR = OUT_DIR / "plots"


def load_full_terms_from_xlsx(path: Path) -> pd.DataFrame:
    terms = read_xlsx_sheet(path, "xl/worksheets/sheet1.xml")
    notes = read_xlsx_sheet(path, "xl/worksheets/sheet2.xml")
    note_text = dict(zip(notes["row_id"].astype(str), notes["text"].astype(str)))
    df = pd.DataFrame(
        {
            "doc_id": terms["source_row_id"].astype(int),
            "phrase": terms["mention"].map(normalize_text),
            "type": terms["type"].map(normalize_text),
            "context": terms["context"].fillna("").astype(str).map(normalize_text),
            "source_category": terms["source_category"].map(normalize_text),
            "note_text": terms["source_row_id"].astype(str).map(note_text).fillna(""),
        }
    )
    matched = []
    for _, row in df.iterrows():
        p = simulated_match_probability(row["type"])
        matched.append(1 if stable_unit_interval(row["doc_id"], row["phrase"], row["type"], "match") < p else 0)
    df["matched"] = matched
    return df


def simulate_two_lists(full_df: pd.DataFrame, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = random.Random(seed)
    rows: List[Dict[str, object]] = []
    for _, row in full_df.iterrows():
        p1, p2 = sampling_probabilities(str(row["type"]))
        r1 = 1 if rng.random() < p1 else 0
        r2 = 1 if rng.random() < p2 else 0
        state = f"{r1}{r2}"
        rows.append(
            {
                "doc_id": int(row["doc_id"]),
                "phrase": str(row["phrase"]),
                "type": str(row["type"]),
                "context": str(row.get("context", "")),
                "matched": int(row.get("matched", 0)),
                "state": state,
                "r1": r1,
                "r2": r2,
            }
        )
    sim = pd.DataFrame(rows)
    list1 = sim[sim["r1"] == 1][["doc_id", "phrase", "type", "context", "matched"]].copy().reset_index(drop=True)
    list2 = sim[sim["r2"] == 1][["doc_id", "phrase", "type", "context", "matched"]].copy().reset_index(drop=True)
    expected_visible = sim[sim["state"] != "00"][["doc_id", "phrase", "type", "state", "context", "matched"]].copy().reset_index(drop=True)
    return list1, list2, expected_visible


def reconstruct_visible_from_lists(list1: pd.DataFrame, list2: pd.DataFrame) -> pd.DataFrame:
    return build_state_table_from_two_lists(
        list1,
        list2,
        key_cols=("doc_id", "phrase", "type", "context"),
        context_col="context",
        matched_col="matched",
    )


def run() -> Dict[str, object]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_XLSX.exists():
        raise FileNotFoundError(f"Input xlsx not found: {INPUT_XLSX}")
    full_df = load_full_terms_from_xlsx(INPUT_XLSX)
    doc_ids = sorted(int(v) for v in full_df["doc_id"].unique())
    split = math.ceil(len(doc_ids) * 0.6)
    train_doc_ids = doc_ids[:split]
    test_doc_ids = doc_ids[split:]

    train_full = full_df[full_df["doc_id"].isin(train_doc_ids)].copy()
    test_full = full_df[full_df["doc_id"].isin(test_doc_ids)].copy()

    train_list1, train_list2, train_expected_visible = simulate_two_lists(train_full, seed=SIM_SEED)
    test_list1, test_list2, test_expected_visible = simulate_two_lists(test_full, seed=SIM_SEED + 1)

    train_visible = reconstruct_visible_from_lists(train_list1, train_list2)
    test_visible = reconstruct_visible_from_lists(test_list1, test_list2)

    write_table(train_list1, DATA_DIR / "train_list1.csv")
    write_table(train_list2, DATA_DIR / "train_list2.csv")
    write_table(test_list1, DATA_DIR / "test_list1.csv")
    write_table(test_list2, DATA_DIR / "test_list2.csv")
    write_table(train_expected_visible, DATA_DIR / "train_expected_visible.csv")
    write_table(test_expected_visible, DATA_DIR / "test_expected_visible.csv")
    write_table(train_visible, DATA_DIR / "train_reconstructed_visible.csv")
    write_table(test_visible, DATA_DIR / "test_reconstructed_visible.csv")

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

    train_path = DATA_DIR / "train_reconstructed_visible.csv"
    q_function, train_summary = train_q_from_excel(train_path, MODEL_DIR, config)

    pred_totals = synthetic_pred_totals(full_df)
    truth = truth_for_docs(full_df, test_doc_ids, pred_totals)

    pred_input = test_full[["doc_id", "phrase", "type", "context", "matched"]].copy()
    pred_input["state"] = "11"
    pred_input = pred_input[["doc_id", "phrase", "type", "state", "context", "matched"]]
    q = q_function.predict_dataframe(pred_input)

    prec_corr = np.zeros(BOOTSTRAP_B, dtype=np.float64)
    rec_corr = np.zeros(BOOTSTRAP_B, dtype=np.float64)
    prec_naive = np.zeros(BOOTSTRAP_B, dtype=np.float64)
    rec_naive = np.zeros(BOOTSTRAP_B, dtype=np.float64)
    test_full_reset = test_full.reset_index(drop=True)
    for b in range(BOOTSTRAP_B):
        one = bootstrap_once(test_full_reset, q, truth["pred_total"], BOOTSTRAP_BASE_SEED + b)
        prec_corr[b] = one["corrected_precision"]
        rec_corr[b] = one["corrected_recall"]
        prec_naive[b] = one["naive_precision"]
        rec_naive[b] = one["naive_recall"]

    plot_hist_two(prec_corr, prec_naive, truth["precision_true"], "List-based simulation precision", "Precision", PLOT_DIR / "precision_hist.png")
    plot_hist_two(rec_corr, rec_naive, truth["recall_true"], "List-based simulation recall", "Recall", PLOT_DIR / "recall_hist.png")

    bar_rows = [
        {"method": "Full truth", "precision": truth["precision_true"], "recall": truth["recall_true"]},
        {"method": "Naive visible mean", "precision": float(prec_naive.mean()), "recall": float(rec_naive.mean())},
        {"method": "CRC corrected mean", "precision": float(prec_corr.mean()), "recall": float(rec_corr.mean())},
    ]
    draw_metric_barplot(bar_rows, PLOT_DIR / "summary_barplot.png")

    summary = {
        "method": "list_to_state_then_crc",
        "input_xlsx": str(INPUT_XLSX),
        "simulation_seed_train": SIM_SEED,
        "simulation_seed_test": SIM_SEED + 1,
        "n_full_terms": int(len(full_df)),
        "n_train_visible_reconstructed": int(len(train_visible)),
        "n_test_visible_reconstructed": int(len(test_visible)),
        "train_visible_state_counts": {k: int(v) for k, v in train_visible["state"].value_counts().to_dict().items()},
        "test_visible_state_counts": {k: int(v) for k, v in test_visible["state"].value_counts().to_dict().items()},
        "truth": truth,
        "bootstrap": {
            "n_resamples": BOOTSTRAP_B,
            "corrected": {
                "precision_mean": float(prec_corr.mean()),
                "precision_std": float(prec_corr.std(ddof=1)),
                "recall_mean": float(rec_corr.mean()),
                "recall_std": float(rec_corr.std(ddof=1)),
            },
            "naive": {
                "precision_mean": float(prec_naive.mean()),
                "precision_std": float(prec_naive.std(ddof=1)),
                "recall_mean": float(rec_naive.mean()),
                "recall_std": float(rec_naive.std(ddof=1)),
            },
        },
        "paths": {
            "precision_hist": str(PLOT_DIR / "precision_hist.png"),
            "recall_hist": str(PLOT_DIR / "recall_hist.png"),
            "summary_barplot": str(PLOT_DIR / "summary_barplot.png"),
            "train_list1": str(DATA_DIR / "train_list1.csv"),
            "train_list2": str(DATA_DIR / "train_list2.csv"),
            "train_visible": str(DATA_DIR / "train_reconstructed_visible.csv"),
            "test_list1": str(DATA_DIR / "test_list1.csv"),
            "test_list2": str(DATA_DIR / "test_list2.csv"),
            "test_visible": str(DATA_DIR / "test_reconstructed_visible.csv"),
        },
        "train_summary": train_summary,
    }

    with open(OUT_DIR / "simulation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


if __name__ == "__main__":
    run()
