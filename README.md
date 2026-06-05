# Two-List CRC Model Evaluation

This project evaluates clinical term extraction when human annotations are incomplete. It reconstructs visible capture states from two annotation lists, matches model predictions to visible human terms, trains a q-function for capture probabilities, and reports naive and CRC-corrected metrics.

The recommended synthetic validation entry point is:

```bash
python3 synthetic_bootstrap.py
```

This script simulates the same three inputs used in real evaluation:

```text
list1_df: terms captured by annotator/list 1
list2_df: terms captured by annotator/list 2
model_df: explicit synthetic model predictions
```

Ground truth metrics use hidden synthetic truth. Naive and CRC metrics are computed through the real matcher-based path from `list1_df`, `list2_df`, and `model_df`.

## Workflow

Two observed lists define the visible capture state of each term:

| State | Meaning |
| --- | --- |
| `10` | The term appears only in list 1. |
| `01` | The term appears only in list 2. |
| `11` | The term appears in both lists. |

Rows with state `00` are hidden from both lists and are not included in the observed table.

The evaluation pipeline is:

1. Build a visible state table from `list1_df` and `list2_df`.
2. Match visible human terms to `model_df` with character or AI matching.
3. Train or load a q-function that estimates `q1`, `q2`, and `q12`.
4. Compute naive and CRC-corrected precision, recall, and type accuracy.

## Repository Layout

| Path | Purpose |
| --- | --- |
| `evaluate_two_lists_model.py` | Main API for real user-provided lists and model predictions. |
| `synthetic_bootstrap.py` | Recommended synthetic validation: simulates `list1_df`, `list2_df`, and `model_df`, then evaluates through the real matcher-based path and plots CRC metrics. |
| `type_match_accuracy.py` | Older standalone type-accuracy bootstrap experiment. The recommended path is now `synthetic_bootstrap.py`. |
| `model_term_matching.py` | Standalone utilities for character-based or AI-assisted term matching. |
| `build_two_lists_and_validate.py` | Lightweight validation of two-list state reconstruction. Does not train PubMedBERT. |
| `run_list_state_simulation.py` | Older synthetic simulation that directly uses generated synthetic match labels rather than an explicit synthetic `model_df`. |
| `simulation_bootstrap_validation.py` | Wrapper around `run_list_state_simulation.py`. |
| `crc_functions.py` | Helper wrappers for state-table construction, q-function training, and table output. |
| `synthetic_pipeline.py` | Synthetic workbook loading, sampling probabilities, bootstrap helpers, and plotting helpers. |
| `mimic_iii_synthetic_term_extraction_50_long_full_context-2.xlsx` | Synthetic source workbook with extracted terms and source notes. It does not contain a `matched` column. |
| `sample_full_terms.csv` | Small fallback sample if the workbook is unavailable. |

## Requirements

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

or:

```bash
python3 -m pip install pandas numpy torch transformers matplotlib openpyxl
```

The q-function uses a local PubMedBERT checkpoint through Hugging Face with `local_files_only=True`. The default path is:

```text
~/.cache/huggingface/hub/models--microsoft--BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext/snapshots/e1354b7a3a09615f6aba48dfad4b7a613eef7062
```

If this checkpoint is missing, download PubMedBERT locally or set `PUBMEDBERT_PATH` / `TrainConfig.model_path` to an available local model directory.

## Quick Start

Run commands from the project folder:

```bash
cd /Users/kylewang/Desktop/CRCmodel_evaluation_20260604/CRCModel_evaluation
```

Validate two-list state reconstruction:

```bash
python3 build_two_lists_and_validate.py
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

Generate the recommended synthetic CRC plots:

```bash
python3 synthetic_bootstrap.py
```

This reads the bundled workbook and simulates:

| Simulated object | Output |
| --- | --- |
| `list1_df` | `simulation_outputs/real_entry/data/test_list1.csv` |
| `list2_df` | `simulation_outputs/real_entry/data/test_list2.csv` |
| `model_df` | `simulation_outputs/real_entry/data/test_model_predictions.csv` |
| hidden full truth | `simulation_outputs/real_entry/data/test_full_truth.csv` |

It then evaluates and bootstraps through matcher-generated labels. Outputs:

```text
simulation_outputs/real_entry/CRC_metrics_summary.json
simulation_outputs/real_entry/point_evaluation/evaluation_summary.json
simulation_outputs/real_entry/plots/CRC_precision.png
simulation_outputs/real_entry/plots/CRC_recall.png
simulation_outputs/real_entry/plots/CRC_type_accuracy.png
```

Plot meanings:

| Plot | Description |
| --- | --- |
| `CRC_precision.png` | Bootstrap distributions for naive and CRC-corrected precision, with hidden full-list truth marked. |
| `CRC_recall.png` | Bootstrap distributions for naive and CRC-corrected recall, with hidden full-list truth marked. |
| `CRC_type_accuracy.png` | Bootstrap distributions for naive and CRC-corrected type accuracy, with hidden full-list truth marked. |

Latest bundled-data run:

| Metric | Ground truth | CRC bootstrap mean | Naive bootstrap mean |
| --- | ---: | ---: | ---: |
| Precision | 0.8094 | 0.7899 | 0.7112 |
| Recall | 0.5502 | 0.5497 | 0.5554 |
| Type accuracy | 0.8571 | 0.8559 | 0.8596 |

## How The Synthetic Data Is Built

The workbook contains extracted terms and source notes. It does not contain model predictions or a `matched` column.

`synthetic_bootstrap.py` creates hidden synthetic truth and explicit model predictions:

1. For each full ground-truth term, simulate whether the model got the phrase right: `phrase_match_truth`.
2. Conditional on phrase match, simulate whether the semantic type is right: `type_match_truth`.
3. Define `matched_truth = phrase_match_truth * type_match_truth`.
4. Create one `model_df` row for each phrase-matched term. If type is wrong, the prediction keeps the phrase but uses a wrong semantic type.
5. Add false-positive predictions so precision has a realistic denominator.
6. Split terms by document into train/test.
7. Simulate `list1_df` and `list2_df` capture from the full terms.

Ground truth metrics use hidden columns from `test_full_truth.csv`. Naive and CRC metrics do not directly read those hidden columns; they are computed from matcher output between visible `list1_df` / `list2_df` terms and `model_df`.

## Metric Definitions

For hidden full truth:

```text
precision_true = sum(matched_truth) / len(model_df)
recall_true = sum(matched_truth) / number of full ground-truth terms
type_accuracy_true = sum(matched_truth) / sum(phrase_match_truth)
```

CRC uses the same capture weight for all three metrics:

```text
w_i = ((r1_i / q1_i) + (r2_i / q2_i) - (r12_i / q12_i)) / pi_hat_i
pi_hat_i = q12_i / (q1_i * q2_i)
```

Then:

```text
CRC precision = sum(w_i * matched_i) / len(model_df)
CRC recall = sum(w_i * matched_i) / sum(w_i)
CRC type accuracy = sum(w_i * phrase_match_i * type_match_i) / sum(w_i * phrase_match_i)
```

The naive versions use the same numerators and denominators without `w_i`, restricted to visible rows.

## Real Data Input Format

`list1_df` and `list2_df` must contain:

```text
doc_id, phrase, type, context
```

`model_df` must contain:

```text
doc_id, phrase, type
```

Recommended meanings:

| Column | Meaning |
| --- | --- |
| `doc_id` | Document, note, or record identifier. |
| `phrase` | Extracted clinical term text. |
| `type` | Semantic type label or UMLS semantic type code. |
| `context` | Source text context for the extracted term. |

## Evaluate Real Data

Use `evaluate_two_lists_with_model()`:

```python
from evaluate_two_lists_model import evaluate_two_lists_with_model

summary = evaluate_two_lists_with_model(
    list1_df=list1_df,
    list2_df=list2_df,
    model_df=model_df,
    output_dir="evaluation_outputs",
    method="character",
)

print(summary["estimate"])
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

To reuse a trained q-function:

```python
from evaluate_two_lists_model import QFunction, evaluate_two_lists_with_model

q_function = QFunction.load("simulation_outputs/real_entry/models/pubmedbert/q_function.pt")

summary = evaluate_two_lists_with_model(
    list1_df=list1_df,
    list2_df=list2_df,
    model_df=model_df,
    q_function=q_function,
    pred_total=len(model_df),
)
```

## Matching Methods

### Character Matching

`method="character"` is deterministic. It normalizes case, whitespace, simple punctuation, and common semantic type codes. For example, `T047` and `Disease or Syndrome` are treated as equivalent semantic types. It is conservative and does not infer synonyms such as `high blood pressure` and `hypertension`.

### AI Matching

`method="ai"` is for synonyms, abbreviations, acronyms, professional terms, and paraphrases. The built-in prompt emphasizes semantic clinical equivalence, including examples such as:

```text
HTN <-> hypertension
DM2 <-> type 2 diabetes mellitus
SOB <-> shortness of breath
myocardial infarction <-> heart attack
ceftriaxone <-> Rocephin
```

You provide an `ai_matcher(prompt)` function that sends the generated prompt to your AI API and returns valid JSON:

```python
from openai import OpenAI

client = OpenAI()

def ai_matcher(prompt: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "Return only valid JSON. No markdown."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )
    return response.choices[0].message.content
```

Expected JSON:

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

Only pairs with `phrase_match=true` are retained.

## Q-Function Details

The default q-function input is:

```text
phrase [SEP] semantic type <type> [SEP] context <context>
```

It trains three binary heads:

| Head | Label |
| --- | --- |
| `q1` | `state` is `10` or `11`. |
| `q2` | `state` is `01` or `11`. |
| `q12` | `state` is `11`. |

Model predictions are not used to train q. They are used only to create match labels for evaluation.
