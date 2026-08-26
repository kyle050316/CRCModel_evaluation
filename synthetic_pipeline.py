import os
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
from zipfile import ZipFile
import xml.etree.ElementTree as ET

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-crc-evaluation")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/crc-evaluation-cache")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from crc_functions import EPS, MAX_ABS_WEIGHT


ROOT = Path(__file__).resolve().parent
SYNTHETIC_XLSX = ROOT / "mimic_iii_synthetic_term_extraction_50_long_full_context-2.xlsx"
ALT_SYNTHETIC_XLSX = ROOT / "mimic_iii_synthetic_term_extraction_50_long_full_context-2 2.xlsx"
SAMPLE_CSV = ROOT / "sample_full_terms.csv"

BOOTSTRAP_B = 1000
BOOTSTRAP_BASE_SEED = 2025
SIM_SEED = 20260422


def column_index(cell_ref: str) -> int:
    letters = "".join(ch for ch in cell_ref if ch.isalpha())
    value = 0
    for ch in letters:
        value = value * 26 + (ord(ch.upper()) - ord("A") + 1)
    return value - 1


def cell_text(cell: ET.Element, ns: Dict[str, str]) -> str:
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        parts = [t.text or "" for t in cell.findall(".//m:t", ns)]
        return "".join(parts)
    value = cell.find("m:v", ns)
    return value.text if value is not None and value.text is not None else ""


def read_xlsx_sheet(path: Path, sheet_xml: str) -> pd.DataFrame:
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with ZipFile(path) as zf:
        root = ET.fromstring(zf.read(sheet_xml))
    rows: List[List[str]] = []
    for row in root.findall(".//m:sheetData/m:row", ns):
        vals: Dict[int, str] = {}
        for cell in row.findall("m:c", ns):
            vals[column_index(cell.attrib["r"])] = cell_text(cell, ns)
        if vals:
            width = max(vals) + 1
            rows.append([vals.get(i, "") for i in range(width)])
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows[1:], columns=rows[0])


def stable_unit_interval(*parts: object, seed: int = SIM_SEED) -> float:
    text = "|".join(str(p) for p in parts) + f"|{seed}"
    h = 2166136261
    for b in text.encode("utf-8", errors="ignore"):
        h ^= b
        h = (h * 16777619) & 0xFFFFFFFF
    return h / 0xFFFFFFFF


def normalize_text(value: object) -> str:
    return " ".join(str(value or "").strip().split())


def sampling_probabilities(type_text: str) -> Tuple[float, float]:
    typ = type_text.lower()
    if any(key in typ for key in ["diagnosis", "symptom", "finding", "microbiology"]):
        return 0.82, 0.72
    if any(key in typ for key in ["medication", "therapy", "procedure", "imaging", "lab"]):
        return 0.70, 0.58
    return 0.62, 0.48


def simulated_match_probability(type_text: str) -> float:
    typ = type_text.lower()
    if any(key in typ for key in ["diagnosis", "procedure", "medication"]):
        return 0.78
    if any(key in typ for key in ["symptom", "finding", "lab", "microbiology", "imaging"]):
        return 0.68
    return 0.58


def load_synthetic_full_terms(path: Path = SYNTHETIC_XLSX) -> pd.DataFrame:
    if not path.exists() and path == SYNTHETIC_XLSX and ALT_SYNTHETIC_XLSX.exists():
        path = ALT_SYNTHETIC_XLSX
    if not path.exists() and SAMPLE_CSV.exists():
        return pd.read_csv(SAMPLE_CSV)
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


def synthetic_pred_totals(full_df: pd.DataFrame) -> Dict[int, int]:
    totals: Dict[int, int] = {}
    for doc_id, group in full_df.groupby("doc_id"):
        tp = int(group["matched"].sum())
        false_positive = 1 + int(stable_unit_interval(doc_id, "fp") < 0.45)
        totals[int(doc_id)] = tp + false_positive
    return totals


def truth_for_docs(full_df: pd.DataFrame, test_doc_ids: Sequence[int], pred_totals: Dict[int, int]) -> Dict[str, float]:
    test = full_df[full_df["doc_id"].isin(test_doc_ids)]
    tp_total = float(test["matched"].sum())
    true_total = float(len(test))
    pred_total = float(sum(pred_totals[int(doc_id)] for doc_id in test_doc_ids))
    return {
        "precision_true": tp_total / pred_total if pred_total else 0.0,
        "recall_true": tp_total / true_total if true_total else 0.0,
        "tp_total": tp_total,
        "true_total": true_total,
        "pred_total": pred_total,
        "n_docs": len(test_doc_ids),
    }


def bootstrap_once(test_df: pd.DataFrame, q: np.ndarray, pred_total: float, seed: int) -> Dict[str, float]:
    rng = random.Random(seed)
    tp_corr_sum = 0.0
    true_corr_sum = 0.0
    tp_naive_sum = 0.0
    true_naive_sum = 0.0
    for i, row in test_df.iterrows():
        p1, p2 = sampling_probabilities(str(row["type"]))
        r1 = 1 if rng.random() < p1 else 0
        r2 = 1 if rng.random() < p2 else 0
        r12 = 1 if r1 == 1 and r2 == 1 else 0
        matched = float(row["matched"])
        if r1 + r2 >= 1:
            true_naive_sum += 1.0
            tp_naive_sum += matched
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
        tp_corr_sum += matched * w
        true_corr_sum += w
    return {
        "corrected_precision": tp_corr_sum / pred_total if pred_total else 0.0,
        "corrected_recall": tp_corr_sum / true_corr_sum if true_corr_sum > 0 else 0.0,
        "naive_precision": tp_naive_sum / pred_total if pred_total else 0.0,
        "naive_recall": tp_naive_sum / true_naive_sum if true_naive_sum > 0 else 0.0,
    }


def plot_hist_two(
    dist_a: np.ndarray,
    dist_b: np.ndarray,
    vline: float,
    title: str,
    xlabel: str,
    outpath: Path,
    truth_label: str = "Full-list truth",
) -> None:
    plt.figure()
    plt.hist(dist_a, bins=20, alpha=0.6, label="CRC correction")
    plt.hist(dist_b, bins=20, alpha=0.6, label="Naive (biased)")
    plt.axvline(vline, linestyle="--", linewidth=2, label=truth_label)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=220)
    plt.close()
