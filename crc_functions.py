import os
from pathlib import Path
from typing import Dict, List

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-crc-evaluation")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/crc-evaluation-cache")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from evaluate_two_lists_model import (
    EPS,
    MAX_ABS_WEIGHT,
    PUBMEDBERT_PATH,
    QFunction,
    TrainConfig,
    build_state_table_from_two_lists as _build_state_table_from_two_lists,
    estimate_precision_recall,
    read_table,
    train_q_from_table,
    write_table,
)


def train_q_from_excel(input_path, out_dir, config):
    return train_q_from_table(input_path, out_dir, config)


def _normalize_key_text(value: object) -> str:
    return " ".join(str(value or "").strip().split()).lower()


def build_state_table_from_two_lists(
    list1_df,
    list2_df,
    key_cols=("doc_id", "phrase", "type", "context"),
    context_col="context",
    matched_col=None,
):
    if context_col == "context" and matched_col is None:
        return _build_state_table_from_two_lists(list1_df, list2_df, key_cols=key_cols)

    for c in key_cols:
        if c not in list1_df.columns or c not in list2_df.columns:
            raise ValueError(f"Missing key column in list inputs: {c}")
    if matched_col is not None and matched_col not in list1_df.columns and matched_col not in list2_df.columns:
        raise ValueError(f"matched_col '{matched_col}' not found in either list input")

    def canonicalize(df):
        out = df.copy()
        for c in key_cols:
            if c == "doc_id":
                out[c] = out[c].astype(int)
            else:
                out[c] = out[c].map(_normalize_key_text)
        if context_col in out.columns:
            out[context_col] = out[context_col].fillna("").astype(str).map(_normalize_key_text)
        else:
            out[context_col] = ""
        keep = list(dict.fromkeys(list(key_cols) + [context_col]))
        if matched_col is not None and matched_col in out.columns:
            keep.append(matched_col)
        return out[keep].drop_duplicates(subset=list(key_cols), keep="first").reset_index(drop=True)

    a = canonicalize(list1_df)
    b = canonicalize(list2_df)
    merged = a.merge(b, on=list(key_cols), how="outer", suffixes=("_1", "_2"), indicator=True)
    merged["state"] = merged["_merge"].map({"left_only": "10", "right_only": "01", "both": "11"}).astype(str)
    if context_col in key_cols:
        merged["context"] = merged[context_col].fillna("")
    else:
        merged["context"] = merged[f"{context_col}_1"].where(
            merged[f"{context_col}_1"].astype(str).str.len() > 0,
            merged[f"{context_col}_2"],
        ).fillna("")

    out_cols = list(dict.fromkeys(list(key_cols) + ["state", "context"]))
    out = merged[out_cols].copy()
    if matched_col is not None:
        left = f"{matched_col}_1"
        right = f"{matched_col}_2"
        if left in merged.columns or right in merged.columns:
            lv = merged[left] if left in merged.columns else 0
            rv = merged[right] if right in merged.columns else 0
            out[matched_col] = np.maximum(
                np.asarray(lv.fillna(0), dtype=float),
                np.asarray(rv.fillna(0), dtype=float),
            ).astype(int)
    return out.sort_values(["doc_id", "phrase", "type"]).reset_index(drop=True)


def draw_metric_barplot(rows: List[Dict[str, object]], out_path: str | os.PathLike) -> None:
    labels = [str(r["method"]) for r in rows]
    precision = [float(r["precision"]) for r in rows]
    recall = [float(r["recall"]) for r in rows]
    x = np.arange(len(labels))
    width = 0.36
    plt.figure(figsize=(9, 5))
    plt.bar(x - width / 2, precision, width, label="Precision")
    plt.bar(x + width / 2, recall, width, label="Recall")
    plt.xticks(x, labels, rotation=20, ha="right")
    plt.ylim(0, max(precision + recall) * 1.2 if precision or recall else 1.0)
    plt.ylabel("Metric")
    plt.title("CRC estimates by model")
    plt.legend()
    plt.tight_layout()
    plt.savefig(Path(out_path), dpi=220)
    plt.close()
