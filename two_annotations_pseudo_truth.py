"""Validate CRC when two annotations, but no complete ground truth, are available.

The union of the two original annotation lists is treated as ground truth.
Every ground-truth term is then sampled independently into list 1 and list 2
with term-specific probabilities p1(z) and p2(z). The two simulated lists are
passed to the unchanged CRC entry point, and the merged ground truth is used
only to validate the simulation.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from evaluate_two_lists_model import (
    QFunction,
    TrainConfig,
    build_state_table_from_two_lists,
    evaluate_two_lists_with_model,
    make_evaluation_table,
    train_q_from_table,
    write_table,
)
from run_list_state_simulation import INPUT_XLSX, load_full_terms_from_xlsx, relpath, simulate_two_lists
from synthetic_bootstrap import (
    add_match_columns,
    add_model_truth_columns,
    build_synthetic_model_df,
    canonicalize_terms,
    compute_estimate_from_eval_df,
    compute_type_accuracy_from_eval_df,
    split_full_terms,
)
from synthetic_pipeline import (
    BOOTSTRAP_BASE_SEED,
    SIM_SEED,
    load_synthetic_full_terms,
    plot_hist_two,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_OUT_DIR = ROOT / "simulation_outputs" / "two_annotations_pseudo_truth_p_z"
DEFAULT_BOOTSTRAP_RESAMPLES = 1000

# ---------------------------------------------------------------------------
# Tunable simulation settings
# ---------------------------------------------------------------------------
# Base deletion proportions. For a type absent from TYPE_PROBABILITY_ADJUSTMENT,
# p1(z)=1-LIST1_BASE_DELETION_PROPORTION and
# p2(z)=1-LIST2_BASE_DELETION_PROPORTION.
LIST1_BASE_DELETION_PROPORTION = 0.15
LIST2_BASE_DELETION_PROPORTION = 0.35

# p(z) currently uses type(z), one component of
# z=(doc_id, phrase, type, context). Positive values make that type more likely
# to remain in a simulated list; negative values make it more likely to be
# deleted. Add or edit exact lowercase type names here.
TYPE_PROBABILITY_ADJUSTMENT = {
    "diagnosis": -0.25,
    "medication": -0.25,
    "procedure": -0.25,
    "symptom": -0.10,
    "finding": -0.10,
    "imaging_finding": -0.10,
    "imaging": -0.10,
    "lab": -0.10,
    "comorbidity": 0.15,
    "treatment": 0.15,
    "sign": 0.15,
    "risk_factor": 0.15,
    "test": 0.15,
    "vital_sign": 0.15,
    "anatomy": 0.15,
    "specialty": 0.15,
    "organism": 0.15,
    "score": 0.15,
}

# Final p1(z)/p2(z) values are clipped to this interval to avoid zero capture
# probabilities and near-certain captures.
MIN_KEEP_PROBABILITY = 0.05
MAX_KEEP_PROBABILITY = 0.95

# Fixed seeds make the point simulation reproducible. Bootstrap replicate b
# uses BOOTSTRAP_BASE_SEED+b.
TRAIN_THINNING_SEED = SIM_SEED + 100
TEST_THINNING_SEED = SIM_SEED + 101

KEY_COLUMNS = ["doc_id", "phrase", "type", "context"]


def _validate_deletion_proportion(name: str, value: float) -> None:
    if not 0.0 <= value < 1.0:
        raise ValueError(f"{name} must be in [0, 1), got {value}")


def _p_of_z(row: pd.Series, base_deletion_proportion: float) -> float:
    """p(z) based on type(z); edit the settings above to change its behavior."""
    base_keep_probability = 1.0 - base_deletion_proportion
    type_name = str(row.get("type", "") or "").strip().lower()
    adjustment = TYPE_PROBABILITY_ADJUSTMENT.get(type_name, 0.0)
    return float(np.clip(base_keep_probability + adjustment, MIN_KEEP_PROBABILITY, MAX_KEEP_PROBABILITY))


def p1_of_z(row: pd.Series, base_deletion_proportion: float = LIST1_BASE_DELETION_PROPORTION) -> float:
    return _p_of_z(row, base_deletion_proportion)


def p2_of_z(row: pd.Series, base_deletion_proportion: float = LIST2_BASE_DELETION_PROPORTION) -> float:
    return _p_of_z(row, base_deletion_proportion)


def p_z_configuration(list1_deletion_proportion: float, list2_deletion_proportion: float) -> Dict[str, object]:
    return {
        "definition": "pj(z)=clip(1-base_deletion_j+type_adjustment[type(z)], min_keep, max_keep)",
        "sampling_source": "each merged-ground-truth term is independently eligible for both lists",
        "merge_strategy": "multiset union: preserve internal duplicates and pair cross-annotation duplicates",
        "z_fields": KEY_COLUMNS,
        "active_z_features": ["type"],
        "list1_base_deletion_proportion": list1_deletion_proportion,
        "list2_base_deletion_proportion": list2_deletion_proportion,
        "type_probability_adjustment": TYPE_PROBABILITY_ADJUSTMENT,
        "min_keep_probability": MIN_KEEP_PROBABILITY,
        "max_keep_probability": MAX_KEEP_PROBABILITY,
    }


def prepare_annotations_and_ground_truth(
    list1_df: pd.DataFrame,
    list2_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Normalize annotations and form a multiset union without internal deduplication."""
    state_table = build_state_table_from_two_lists(list1_df, list2_df, key_cols=KEY_COLUMNS)
    normalized_list1 = state_table.loc[state_table["state"].isin(["10", "11"]), KEY_COLUMNS].reset_index(drop=True)
    normalized_list2 = state_table.loc[state_table["state"].isin(["01", "11"]), KEY_COLUMNS].reset_index(drop=True)
    ground_truth = state_table[KEY_COLUMNS].copy().reset_index(drop=True)
    return normalized_list1, normalized_list2, ground_truth


def thin_annotation_lists(
    list1_df: pd.DataFrame,
    list2_df: pd.DataFrame,
    seed: int,
    list1_deletion_proportion: float = LIST1_BASE_DELETION_PROPORTION,
    list2_deletion_proportion: float = LIST2_BASE_DELETION_PROPORTION,
) -> tuple:
    """Independently retain each row according to p1(z) or p2(z)."""
    _validate_deletion_proportion("list1_deletion_proportion", list1_deletion_proportion)
    _validate_deletion_proportion("list2_deletion_proportion", list2_deletion_proportion)

    rng = np.random.default_rng(seed)
    probabilities1 = list1_df.apply(p1_of_z, axis=1, base_deletion_proportion=list1_deletion_proportion)
    probabilities2 = list2_df.apply(p2_of_z, axis=1, base_deletion_proportion=list2_deletion_proportion)
    draws1 = rng.random(len(list1_df))
    draws2 = rng.random(len(list2_df))
    keep1 = draws1 < probabilities1.to_numpy()
    keep2 = draws2 < probabilities2.to_numpy()
    thinned1 = list1_df.loc[keep1, KEY_COLUMNS].copy().reset_index(drop=True)
    thinned2 = list2_df.loc[keep2, KEY_COLUMNS].copy().reset_index(drop=True)
    return thinned1, thinned2


def ground_truth_metrics(ground_truth: pd.DataFrame, model_df: pd.DataFrame) -> Tuple[Dict[str, float], pd.DataFrame]:
    """Compute direct reference metrics against the merged annotation union."""
    phrase_eval, phrase_matches = make_evaluation_table(
        ground_truth,
        ground_truth,
        model_df,
        method="character",
        require_type_match=False,
    )
    labeled = add_match_columns(phrase_eval, phrase_matches)
    phrase_total = float(labeled["phrase_match"].sum())
    both_total = float(labeled["both_phrase_type_match"].sum())
    pred_total = float(len(model_df))
    true_total = float(len(ground_truth))
    metrics = {
        "precision_true": both_total / pred_total if pred_total > 0 else 0.0,
        "recall_true": both_total / true_total if true_total > 0 else 0.0,
        "type_accuracy_true": both_total / phrase_total if phrase_total > 0 else 0.0,
        "tp_total": both_total,
        "true_total": true_total,
        "phrase_match_total": phrase_total,
        "pred_total": pred_total,
        "n_model_rows": int(len(model_df)),
    }
    return metrics, labeled


def train_or_load_q(
    train_visible: pd.DataFrame,
    config: TrainConfig,
    data_dir: Path,
    model_dir: Path,
    simulation_config: Dict[str, object],
) -> Tuple[QFunction, Dict[str, object]]:
    q_path = model_dir / "q_function.pt"
    config_path = model_dir / "simulation_config.json"
    train_path = data_dir / "train_reconstructed_visible.csv"
    write_table(train_visible, train_path)
    saved_config = None
    if config_path.exists():
        saved_config = json.loads(config_path.read_text(encoding="utf-8"))
    if q_path.exists() and saved_config == simulation_config:
        return QFunction.load(q_path), {
            "source": "loaded_existing_q_function",
            "q_function_path": relpath(q_path),
        }
    q_function, train_summary = train_q_from_table(train_path, model_dir, config)
    config_path.write_text(json.dumps(simulation_config, indent=2, ensure_ascii=False), encoding="utf-8")
    return q_function, train_summary


def evaluate_one_thinning(
    list1_df: pd.DataFrame,
    list2_df: pd.DataFrame,
    model_df: pd.DataFrame,
    q_by_key: pd.DataFrame,
    pred_total: float,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    phrase_eval, phrase_matches = make_evaluation_table(
        list1_df,
        list2_df,
        model_df,
        method="character",
        require_type_match=False,
    )
    precision_eval = add_match_columns(phrase_eval, phrase_matches)
    precision_eval["matched"] = precision_eval["both_phrase_type_match"]
    precision_recall = compute_estimate_from_eval_df(precision_eval, q_by_key, pred_total)
    type_accuracy = compute_type_accuracy_from_eval_df(phrase_eval, phrase_matches, q_by_key)
    return precision_recall, type_accuracy


def run_bootstrap(
    ground_truth: pd.DataFrame,
    model_df: pd.DataFrame,
    q_by_key: pd.DataFrame,
    pred_total: float,
    list1_deletion_proportion: float,
    list2_deletion_proportion: float,
    n_resamples: int,
) -> Dict[str, np.ndarray]:
    outputs = {
        "precision_corrected": np.zeros(n_resamples),
        "recall_corrected": np.zeros(n_resamples),
        "precision_naive": np.zeros(n_resamples),
        "recall_naive": np.zeros(n_resamples),
        "type_accuracy_corrected": np.zeros(n_resamples),
        "type_accuracy_naive": np.zeros(n_resamples),
    }
    for b in range(n_resamples):
        list1, list2 = thin_annotation_lists(
            ground_truth,
            ground_truth,
            seed=BOOTSTRAP_BASE_SEED + b,
            list1_deletion_proportion=list1_deletion_proportion,
            list2_deletion_proportion=list2_deletion_proportion,
        )
        estimate, type_accuracy = evaluate_one_thinning(list1, list2, model_df, q_by_key, pred_total)
        outputs["precision_corrected"][b] = estimate["corrected_precision"]
        outputs["recall_corrected"][b] = estimate["corrected_recall"]
        outputs["precision_naive"][b] = estimate["naive_precision"]
        outputs["recall_naive"][b] = estimate["naive_recall"]
        outputs["type_accuracy_corrected"][b] = type_accuracy["corrected_type_accuracy"]
        outputs["type_accuracy_naive"][b] = type_accuracy["naive_type_accuracy"]
    return outputs


def distribution_summary(boot: Dict[str, np.ndarray], n_resamples: int) -> Dict[str, object]:
    return {
        "n_resamples": n_resamples,
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
    }


def run(
    list1_deletion_proportion: float = LIST1_BASE_DELETION_PROPORTION,
    list2_deletion_proportion: float = LIST2_BASE_DELETION_PROPORTION,
    output_dir: Path = DEFAULT_OUT_DIR,
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
) -> Dict[str, object]:
    if n_resamples < 2:
        raise ValueError("n_resamples must be at least 2")
    _validate_deletion_proportion("list1_deletion_proportion", list1_deletion_proportion)
    _validate_deletion_proportion("list2_deletion_proportion", list2_deletion_proportion)
    data_dir = output_dir / "data"
    model_dir = output_dir / "models" / "pubmedbert"
    point_eval_dir = output_dir / "point_evaluation"
    plot_dir = output_dir / "plots"
    summary_path = output_dir / "CRC_metrics_summary.json"
    for directory in [data_dir, model_dir, point_eval_dir, plot_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    full_df = load_full_terms_from_xlsx(INPUT_XLSX) if INPUT_XLSX.exists() else load_synthetic_full_terms(INPUT_XLSX)
    full_df = add_model_truth_columns(full_df)
    train_full, test_full = split_full_terms(full_df)
    all_model_df = build_synthetic_model_df(full_df)
    test_model_df = all_model_df[all_model_df["doc_id"].isin(test_full["doc_id"].unique())].copy().reset_index(drop=True)

    train_raw1, train_raw2, _ = simulate_two_lists(train_full, seed=SIM_SEED)
    test_raw1, test_raw2, _ = simulate_two_lists(test_full, seed=SIM_SEED + 1)
    train_original1, train_original2, train_ground_truth = prepare_annotations_and_ground_truth(train_raw1, train_raw2)
    test_original1, test_original2, test_ground_truth = prepare_annotations_and_ground_truth(test_raw1, test_raw2)
    train_list1, train_list2 = thin_annotation_lists(
        train_ground_truth,
        train_ground_truth,
        TRAIN_THINNING_SEED,
        list1_deletion_proportion,
        list2_deletion_proportion,
    )
    test_list1, test_list2 = thin_annotation_lists(
        test_ground_truth,
        test_ground_truth,
        TEST_THINNING_SEED,
        list1_deletion_proportion,
        list2_deletion_proportion,
    )
    train_visible = build_state_table_from_two_lists(train_list1, train_list2, key_cols=KEY_COLUMNS)

    config = TrainConfig(model_name="two_annotations_p_z_pubmedbert")
    simulation_config = p_z_configuration(list1_deletion_proportion, list2_deletion_proportion)
    p_z_config_path = output_dir / "p_z_configuration.json"
    p_z_config_path.write_text(json.dumps(simulation_config, indent=2, ensure_ascii=False), encoding="utf-8")
    q_function, train_summary = train_or_load_q(
        train_visible,
        config,
        data_dir,
        model_dir,
        simulation_config,
    )

    for name, table in {
        "train_original_list1.csv": train_original1,
        "train_original_list2.csv": train_original2,
        "train_ground_truth.csv": train_ground_truth,
        "train_list1.csv": train_list1,
        "train_list2.csv": train_list2,
        "test_original_list1.csv": test_original1,
        "test_original_list2.csv": test_original2,
        "test_ground_truth.csv": test_ground_truth,
        "test_list1.csv": test_list1,
        "test_list2.csv": test_list2,
        "test_model_predictions.csv": test_model_df,
    }.items():
        write_table(table, data_dir / name)

    truth, labeled_ground_truth = ground_truth_metrics(test_ground_truth, test_model_df)
    write_table(labeled_ground_truth, data_dir / "test_ground_truth_with_matches.csv")

    point_summary = evaluate_two_lists_with_model(
        list1_df=test_list1,
        list2_df=test_list2,
        model_df=test_model_df,
        output_dir=point_eval_dir,
        method="character",
        require_type_match=True,
        pred_total=len(test_model_df),
        q_function=q_function,
    )
    q_input = canonicalize_terms(test_ground_truth)
    q_pred = q_function.predict_dataframe(q_input.assign(state="11"))
    q_by_key = q_input.copy()
    q_by_key[["q1", "q2", "q12"]] = q_pred
    q_by_key = q_by_key.drop_duplicates(KEY_COLUMNS, keep="first")
    _, point_type_accuracy = evaluate_one_thinning(
        test_list1,
        test_list2,
        test_model_df,
        q_by_key,
        float(len(test_model_df)),
    )

    boot = run_bootstrap(
        test_ground_truth,
        test_model_df,
        q_by_key,
        float(len(test_model_df)),
        list1_deletion_proportion,
        list2_deletion_proportion,
        n_resamples,
    )
    precision_plot = plot_dir / "CRC_precision.png"
    recall_plot = plot_dir / "CRC_recall.png"
    type_accuracy_plot = plot_dir / "CRC_type_accuracy.png"
    plot_hist_two(boot["precision_corrected"], boot["precision_naive"], truth["precision_true"], "CRC precision (merged ground truth)", "Precision", precision_plot, "Merged ground truth")
    plot_hist_two(boot["recall_corrected"], boot["recall_naive"], truth["recall_true"], "CRC recall (merged ground truth)", "Recall", recall_plot, "Merged ground truth")
    plot_hist_two(boot["type_accuracy_corrected"], boot["type_accuracy_naive"], truth["type_accuracy_true"], "CRC type accuracy (merged ground truth)", "Type accuracy", type_accuracy_plot, "Merged ground truth")

    hidden_tp = float(test_full["matched_truth"].sum())
    hidden_phrase = float(test_full["phrase_match_truth"].sum())
    hidden_full_truth_reference = {
        "note": "Available only in this synthetic validation; not used by the merged-ground-truth estimator.",
        "n_terms": int(len(test_full)),
        "n_terms_missing_from_merged_ground_truth": int(len(test_full) - len(test_ground_truth)),
        "precision": hidden_tp / len(test_model_df),
        "recall": hidden_tp / len(test_full),
        "type_accuracy": hidden_tp / hidden_phrase,
    }

    summary = {
        "method": "two_annotation_union_ground_truth_sampled_into_two_lists_by_p_z",
        "matching_method": "character",
        "p_z": simulation_config,
        "thinning": {
            "list1_base_deletion_proportion": list1_deletion_proportion,
            "list2_base_deletion_proportion": list2_deletion_proportion,
            "test_list1_mean_p_z": float(
                test_ground_truth.apply(
                    p1_of_z,
                    axis=1,
                    base_deletion_proportion=list1_deletion_proportion,
                ).mean()
            ),
            "test_list2_mean_p_z": float(
                test_ground_truth.apply(
                    p2_of_z,
                    axis=1,
                    base_deletion_proportion=list2_deletion_proportion,
                ).mean()
            ),
            "test_list1_realized_deletion_proportion": float(1.0 - len(test_list1) / len(test_ground_truth)),
            "test_list2_realized_deletion_proportion": float(1.0 - len(test_list2) / len(test_ground_truth)),
            "train_seed": TRAIN_THINNING_SEED,
            "test_seed": TEST_THINNING_SEED,
        },
        "counts": {
            "train_original_list1": int(len(train_original1)),
            "train_original_list2": int(len(train_original2)),
            "train_ground_truth": int(len(train_ground_truth)),
            "train_simulated_list1": int(len(train_list1)),
            "train_simulated_list2": int(len(train_list2)),
            "test_original_list1": int(len(test_original1)),
            "test_original_list2": int(len(test_original2)),
            "test_ground_truth": int(len(test_ground_truth)),
            "test_simulated_list1": int(len(test_list1)),
            "test_simulated_list2": int(len(test_list2)),
            "test_point_visible": int(point_summary["n_visible_rows"]),
        },
        "truth": truth,
        "hidden_full_truth_reference": hidden_full_truth_reference,
        "point_estimate_from_evaluate_two_lists_with_model": point_summary["estimate"],
        "point_type_accuracy_from_matcher": point_type_accuracy,
        "bootstrap": distribution_summary(boot, n_resamples),
        "paths": {
            "test_original_list1": relpath(data_dir / "test_original_list1.csv"),
            "test_original_list2": relpath(data_dir / "test_original_list2.csv"),
            "test_ground_truth": relpath(data_dir / "test_ground_truth.csv"),
            "test_list1": relpath(data_dir / "test_list1.csv"),
            "test_list2": relpath(data_dir / "test_list2.csv"),
            "p_z_configuration": relpath(p_z_config_path),
            "test_model_predictions": relpath(data_dir / "test_model_predictions.csv"),
            "point_evaluation_summary": relpath(point_eval_dir / "evaluation_summary.json"),
            "precision_hist": relpath(precision_plot),
            "recall_hist": relpath(recall_plot),
            "type_accuracy_hist": relpath(type_accuracy_plot),
            "summary": relpath(summary_path),
        },
        "train_summary": train_summary,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list1-delete", type=float, default=LIST1_BASE_DELETION_PROPORTION)
    parser.add_argument("--list2-delete", type=float, default=LIST2_BASE_DELETION_PROPORTION)
    parser.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAP_RESAMPLES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.list1_delete, args.list2_delete, args.output_dir, args.bootstrap)
