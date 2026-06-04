import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from crc_functions import build_state_table_from_two_lists, write_table
from synthetic_pipeline import SIM_SEED, load_synthetic_full_terms, sampling_probabilities


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"


def simulate_two_lists(full_df: pd.DataFrame, seed: int = SIM_SEED) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = random.Random(seed)
    rows: List[Dict] = []
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
    sim_df = pd.DataFrame(rows)
    list1 = sim_df[sim_df["r1"] == 1][["doc_id", "phrase", "type", "context", "matched"]].copy()
    list2 = sim_df[sim_df["r2"] == 1][["doc_id", "phrase", "type", "context", "matched"]].copy()
    expected_visible = sim_df[sim_df["state"] != "00"][["doc_id", "phrase", "type", "state", "context", "matched"]].copy()
    return list1.reset_index(drop=True), list2.reset_index(drop=True), expected_visible.reset_index(drop=True)


def _norm_text(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.lower().str.split().str.join(" ")


def validate_reconstruction(reconstructed: pd.DataFrame, expected_visible: pd.DataFrame, n_list1: int, n_list2: int) -> Dict[str, object]:
    a = reconstructed.copy()
    b = expected_visible.copy()
    for df in (a, b):
        df["phrase"] = _norm_text(df["phrase"])
        df["type"] = _norm_text(df["type"])
        df["context"] = _norm_text(df["context"])
        df["state"] = df["state"].astype(str)
        if "matched" in df.columns:
            df["matched"] = pd.to_numeric(df["matched"], errors="coerce").fillna(0).astype(int)

    key_cols = ["doc_id", "phrase", "type", "context"]
    merged = a.merge(b, on=key_cols, how="outer", suffixes=("_recon", "_expected"), indicator=True)
    keys_equal = bool((merged["_merge"] == "both").all())
    both = merged[merged["_merge"] == "both"].copy()
    state_mismatch = int((both["state_recon"] != both["state_expected"]).sum())
    state_equal = bool(state_mismatch == 0 and len(merged) == len(a) == len(b))
    matched_equal = True
    if "matched_recon" in both.columns and "matched_expected" in both.columns:
        matched_equal = bool((both["matched_recon"] == both["matched_expected"]).all())
    state_counts = reconstructed["state"].astype(str).value_counts().to_dict()
    return {
        "n_list1": int(n_list1),
        "n_list2": int(n_list2),
        "n_reconstructed_visible": int(len(reconstructed)),
        "n_expected_visible": int(len(expected_visible)),
        "keys_equal": keys_equal,
        "state_equal": state_equal,
        "matched_equal": matched_equal,
        "state_mismatch_count": state_mismatch,
        "reconstructed_state_counts": {k: int(v) for k, v in state_counts.items()},
    }


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    full_df = load_synthetic_full_terms()
    list1, list2, expected_visible = simulate_two_lists(full_df, seed=SIM_SEED)

    reconstructed = build_state_table_from_two_lists(
        list1,
        list2,
        key_cols=("doc_id", "phrase", "type", "context"),
        context_col="context",
        matched_col="matched",
    )

    write_table(list1, DATA_DIR / "simulated_list1.csv")
    write_table(list2, DATA_DIR / "simulated_list2.csv")
    write_table(expected_visible, DATA_DIR / "expected_visible_from_simulation.csv")
    write_table(reconstructed, DATA_DIR / "reconstructed_visible_states.csv")

    report = validate_reconstruction(reconstructed, expected_visible, n_list1=len(list1), n_list2=len(list2))
    report["paths"] = {
        "list1": str(DATA_DIR / "simulated_list1.csv"),
        "list2": str(DATA_DIR / "simulated_list2.csv"),
        "expected": str(DATA_DIR / "expected_visible_from_simulation.csv"),
        "reconstructed": str(DATA_DIR / "reconstructed_visible_states.csv"),
    }
    with open(DATA_DIR / "validation_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
