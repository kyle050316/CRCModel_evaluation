import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
from zipfile import ZipFile
import xml.etree.ElementTree as ET

import pandas as pd

ROOT = Path(__file__).resolve().parent
SYNTHETIC_XLSX = ROOT / "mimic_iii_synthetic_term_extraction_50_long_full_context-2.xlsx"
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
    if any(key in typ for key in ["diagnosis", "symptom", "finding", "microbiology", "disease", "sign"]):
        return 0.82, 0.72
    if any(key in typ for key in ["medication", "therapy", "procedure", "imaging", "lab", "drug", "substance"]):
        return 0.70, 0.58
    return 0.62, 0.48


def simulated_match_probability(type_text: str) -> float:
    typ = type_text.lower()
    if any(key in typ for key in ["diagnosis", "procedure", "medication", "disease"]):
        return 0.78
    if any(key in typ for key in ["symptom", "finding", "lab", "microbiology", "imaging", "sign"]):
        return 0.68
    return 0.58


def load_synthetic_full_terms(path: Path = SYNTHETIC_XLSX) -> pd.DataFrame:
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
