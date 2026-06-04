# Two-List CRC Model Evaluation

This project evaluates a model's clinical term extraction performance when the available human annotations are incomplete. It combines two partially observed annotation lists into visible capture states, trains a q-function to estimate missing-annotation bias, and reports both naive and CRC-corrected precision and recall.

## What The Code Does

The workflow has three main stages:

1. Build a visible state table from two annotation lists.
2. Match model predictions against the visible human terms to create a `matched` label.
3. Train or load a q-function and estimate naive and CRC-corrected metrics.

The visible states are:

| State | Meaning |
| --- | --- |
| `10` | The term appears only in list 1. |
| `01` | The term appears only in list 2. |
| `11` | The term appears in both lists. |

Rows with state `00` are not visible in either list and are not included in the observed table.

## Repository Layout

| Path | Purpose |
| --- | --- |
| `evaluate_two_lists_model.py` | Main API for evaluating user-provided lists and model predictions. |
| `model_term_matching.py` | Utilities for character-based or AI-assisted term matching. |
| `build_two_lists_and_validate.py` | Lightweight validation script for reconstructing visible states from two simulated lists. |
| `run_list_state_simulation.py` | Full synthetic simulation, q-function training, bootstrap evaluation, and plot generation. |
| `simulation_bootstrap_validation.py` | Thin wrapper around `run_list_state_simulation.py`. |
| `crc_functions.py` | Helper functions for state-table construction, q-function training, plotting, and table output. |
| `synthetic_pipeline.py` | Synthetic data loading, sampling, bootstrap logic, and plotting helpers. |
| `data/` | Output folder for the lightweight two-list reconstruction validation. |
| `simulation_outputs/` | Output folder for full simulation data, models, metrics, and plots. |

## Requirements

Install the Python dependencies:

```bash
python3 -m pip install pandas numpy torch transformers matplotlib openpyxl
```

The full q-function training path uses a local PubMedBERT checkpoint through Hugging Face with `local_files_only=True`. The default path is set in `evaluate_two_lists_model.py`:

```text
~/.cache/huggingface/hub/models--microsoft--BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext/snapshots/e1354b7a3a09615f6aba48dfad4b7a613eef7062
```

If the checkpoint is not present, download PubMedBERT locally or update `PUBMEDBERT_PATH` / `TrainConfig.model_path` to point to an available local model directory.

## Quick Start

Run commands from the project folder:

```bash
cd /Users/kylewang/Desktop/GENIE/CRCmodel/CRCevaluation_20260422/list_state_extension
```

Validate two-list state reconstruction:

```bash
python3 build_two_lists_and_validate.py
```

This writes:

```text
data/simulated_list1.csv
data/simulated_list2.csv
data/expected_visible_from_simulation.csv
data/reconstructed_visible_states.csv
data/validation_report.json
```

A successful validation has:

```json
{
  "keys_equal": true,
  "state_equal": true,
  "matched_equal": true,
  "state_mismatch_count": 0
}
```

Run the full synthetic simulation:

```bash
python3 run_list_state_simulation.py
```

Equivalent wrapper:

```bash
python3 simulation_bootstrap_validation.py
```

The full simulation writes:

```text
simulation_outputs/simulation_summary.json
simulation_outputs/data/train_list1.csv
simulation_outputs/data/train_list2.csv
simulation_outputs/data/test_list1.csv
simulation_outputs/data/test_list2.csv
simulation_outputs/data/train_reconstructed_visible.csv
simulation_outputs/data/test_reconstructed_visible.csv
simulation_outputs/models/pubmedbert/q_function.pt
simulation_outputs/models/pubmedbert/train_summary.json
simulation_outputs/plots/precision_hist.png
simulation_outputs/plots/recall_hist.png
simulation_outputs/plots/summary_barplot.png
```

## Expected Simulation Output

The bundled simulation data produces approximately:

| Metric | Value |
| --- | ---: |
| Full terms | 1737 |
| Train reconstructed visible rows | 909 |
| Test reconstructed visible rows | 597 |
| Train state counts | `11=452`, `10=296`, `01=161` |
| Test state counts | `11=289`, `10=183`, `01=125` |
| Full-truth precision | 0.9455 |
| Full-truth recall | 0.6565 |
| Naive precision mean | 0.8285 |
| Naive recall mean | 0.6608 |
| CRC-corrected precision mean | 0.9232 |
| CRC-corrected recall mean | 0.6563 |
| Bootstrap resamples | 1000 |

The key result is that naive precision is biased downward under incomplete annotations. The CRC-corrected precision is closer to the full-truth precision, while recall remains close to the full-truth value.

The plot files summarize this visually:

| Plot | Description |
| --- | --- |
| `precision_hist.png` | Bootstrap distributions for naive and CRC-corrected precision, with full-truth precision marked. |
| `recall_hist.png` | Bootstrap distributions for naive and CRC-corrected recall, with full-truth recall marked. |
| `summary_barplot.png` | Bar plot comparing full truth, naive visible mean, and CRC-corrected mean. |

## Input Format For User Data

`list1_df` and `list2_df` must contain:

```text
doc_id, phrase, type, context
```

`model_df` must contain:

```text
doc_id, phrase, type
```

Recommended meaning of each field:

| Column | Meaning |
| --- | --- |
| `doc_id` | Document, note, or record identifier. |
| `phrase` | Extracted clinical term text. |
| `type` | Semantic type label or UMLS semantic type code. |
| `context` | Source text context for the extracted term. |

## Evaluate A Model With Two Lists

Use `evaluate_two_lists_with_model()` for the main evaluation path:

```python
import pandas as pd
from evaluate_two_lists_model import evaluate_two_lists_with_model

list1_df = pd.DataFrame([
    {
        "doc_id": 1,
        "phrase": "hypertension",
        "type": "Disease or Syndrome",
        "context": "The patient has hypertension.",
    }
])

list2_df = pd.DataFrame([
    {
        "doc_id": 1,
        "phrase": "hypertension",
        "type": "T047",
        "context": "The patient has hypertension.",
    }
])

model_df = pd.DataFrame([
    {
        "doc_id": 1,
        "phrase": "hypertension",
        "type": "Disease or Syndrome",
    }
])

summary = evaluate_two_lists_with_model(
    list1_df=list1_df,
    list2_df=list2_df,
    model_df=model_df,
    output_dir="evaluation_outputs",
    method="character",
)

print(summary)
```

The output directory contains:

```text
evaluation_outputs/data/q_training_visible_terms.csv
evaluation_outputs/data/evaluation_visible_terms.csv
evaluation_outputs/data/model_human_matches.csv
evaluation_outputs/models/q_function/q_function.pt
evaluation_outputs/models/q_function/train_summary.json
evaluation_outputs/estimate/estimate_two_list_pubmedbert.json
evaluation_outputs/evaluation_summary.json
```

`q_training_visible_terms.csv` contains the reconstructed two-list visible state table without `matched`; it is used to train q. `evaluation_visible_terms.csv` contains the same state table plus `matched`, which is created by comparing model predictions with visible human terms.

## Reuse An Existing Q-Function

If a q-function has already been trained, load it and pass it into the evaluator:

```python
from evaluate_two_lists_model import QFunction, evaluate_two_lists_with_model

q_function = QFunction.load("simulation_outputs/models/pubmedbert/q_function.pt")

summary = evaluate_two_lists_with_model(
    list1_df=list1_df,
    list2_df=list2_df,
    model_df=model_df,
    q_function=q_function,
    pred_total=len(model_df),
)
```

Pass `pred_total` when the denominator for precision is not exactly `len(model_df)`.

## Matching Methods

### Character Matching

`method="character"` is deterministic. It normalizes case, whitespace, simple punctuation, and common semantic type codes. For example, `T047` and `Disease or Syndrome` are treated as equivalent semantic types.

This method is conservative and does not infer synonyms such as `high blood pressure` and `hypertension`.

### AI Matching

Use `method="ai"` when synonym, abbreviation, or paraphrase matching is needed. Provide an `ai_matcher(prompt)` function that returns valid JSON:

```python
def ai_matcher(prompt: str) -> str:
    return '{"matches": [{"h_idx": 0, "g_idx": 0, "phrase_match": true, "type_match": true}]}'

summary = evaluate_two_lists_with_model(
    list1_df=list1_df,
    list2_df=list2_df,
    model_df=model_df,
    method="ai",
    ai_matcher=ai_matcher,
)
```

Expected JSON schema:

```json
{
  "matches": [
    {
      "h_idx": 0,
      "g_idx": 0,
      "phrase_match": true,
      "type_match": true
    }
  ]
}
```

Only pairs with `phrase_match=true` are used.

## Q-Function Details

The default q-function uses text inputs of the form:

```text
phrase [SEP] semantic type <type> [SEP] context <context>
```

It trains three binary heads:

| Head | Label |
| --- | --- |
| `q1` | `state` is `10` or `11`. |
| `q2` | `state` is `01` or `11`. |
| `q12` | `state` is `11`. |

Model predictions are not used to train q. They are used only to create the `matched` column for precision and recall estimation.

## Notes

- The project folder is designed to run independently from this directory.
- The full simulation expects the bundled Excel file `mimic_iii_synthetic_term_extraction_50_long_full_context-2 2.xlsx` or the default synthetic Excel filename handled by `synthetic_pipeline.py`.
- Full q-function training requires a local PubMedBERT checkpoint.
- For real evaluations, prefer the one-step API in `evaluate_two_lists_model.py`.
