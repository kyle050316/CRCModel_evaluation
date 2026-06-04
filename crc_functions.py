import os
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

EPS = 1e-6
MAX_ABS_WEIGHT = 1e6


def write_table(df: pd.DataFrame, path: str | os.PathLike) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        df.to_excel(path, index=False)
    else:
        df.to_csv(path, index=False)


def _normalize_key_text(value: object) -> str:
    return " ".join(str(value or "").strip().split()).lower()


def build_state_table_from_two_lists(
    list1_df: pd.DataFrame,
    list2_df: pd.DataFrame,
    key_cols: Sequence[str] = ("doc_id", "phrase", "type", "context"),
    context_col: str = "context",
    matched_col: Optional[str] = None,
) -> pd.DataFrame:
    for col in key_cols:
        if col not in list1_df.columns or col not in list2_df.columns:
            raise ValueError(f"Missing key column in list inputs: {col}")
    if matched_col is not None and matched_col not in list1_df.columns and matched_col not in list2_df.columns:
        raise ValueError(f"matched_col '{matched_col}' not found in either list input")

    def canonicalize(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for col in key_cols:
            if col == "doc_id":
                out[col] = out[col].astype(int)
            else:
                out[col] = out[col].map(_normalize_key_text)
        if context_col in out.columns:
            out[context_col] = out[context_col].fillna("").astype(str).map(_normalize_key_text)
        else:
            out[context_col] = ""
        keep = list(dict.fromkeys(list(key_cols) + [context_col]))
        if matched_col is not None and matched_col in out.columns:
            keep.append(matched_col)
        return out[keep].drop_duplicates(subset=list(key_cols), keep="first").reset_index(drop=True)

    left = canonicalize(list1_df)
    right = canonicalize(list2_df)
    merged = left.merge(right, on=list(key_cols), how="outer", suffixes=("_1", "_2"), indicator=True)
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
        left_col = f"{matched_col}_1"
        right_col = f"{matched_col}_2"
        if left_col in merged.columns or right_col in merged.columns:
            left_values = pd.to_numeric(merged[left_col], errors="coerce").fillna(0) if left_col in merged.columns else 0
            right_values = pd.to_numeric(merged[right_col], errors="coerce").fillna(0) if right_col in merged.columns else 0
            out[matched_col] = pd.concat([pd.Series(left_values), pd.Series(right_values)], axis=1).max(axis=1).astype(int)
    return out.sort_values(["doc_id", "phrase", "type"]).reset_index(drop=True)
