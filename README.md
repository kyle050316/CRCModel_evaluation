# Two-List CRC Model Evaluation

This project is designed to evaluate clinical term extraction systems when human annotations are incomplete. It reconstructs observable capture states from two annotation lists, matches model predictions against visible human-annotated terms, trains a q-function to estimate capture probabilities, and reports both naive metrics and CRC-corrected metrics.

The workflow simulates the three inputs used in a real evaluation:

```text
list1_df: terms captured by annotator/list 1
list2_df: terms captured by annotator/list 2
model_df: explicit synthetic model predictions
```

Ground-truth metrics are computed using hidden synthetic truth. Naive and CRC metrics are computed from the two simulated incomplete annotation lists, using `list1_df`, `list2_df`, and the model output under evaluation (`model_df`).

## Workflow

The two observed lists define the visible capture state of each term:

| State | Meaning                          |
| ----- | -------------------------------- |
| `10`  | The term appears only in list 1. |
| `01`  | The term appears only in list 2. |
| `11`  | The term appears in both lists.  |

Rows with state `00` are hidden from both lists and therefore do not appear in the observed table.

The evaluation pipeline is:

1. Construct the visible state table from `list1_df` and `list2_df`.
2. Match visible human terms against `model_df` using either character matching or AI-assisted matching.
3. Train or load a q-function to estimate `q1`, `q2`, and `q12`.
4. Compute naive and CRC-corrected precision, recall, and type accuracy.

## Repository Structure

| Path                                                            | Purpose                                                                                                                                                         |
| --------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `evaluate_two_lists_model.py`                                   | Functions for evaluating real user-provided annotation lists and model predictions.                                                                             |
| `synthetic_bootstrap.py`                                        | Synthetic validation: simulates `list1_df`, `list2_df`, and `model_df`, evaluates them through the real matcher-based pipeline, and generates CRC metric plots. |
| `type_match_accuracy.py`                                        | Evaluates type extraction accuracy when the model has already extracted the correct phrase.                                                                     |
| `model_term_matching.py`                                        | Character-based or AI-assisted term matching utilities.                                                                                                         |
| `simulation_bootstrap_validation.py`                            | Draw plot based on bootstrap simulation `run_list_state_simulation.py`.                                                                                         |
| `crc_functions.py`                                              | Helper functions for state-table construction, q-function training, and output generation.                                                                      |
| `synthetic_pipeline.py`                                         | Synthetic workbook loading, sampling probabilities, bootstrap utilities, and plotting helpers.                                                                  |
| `mimic_iii_synthetic_term_extraction_50_long_full_context.xlsx` | A synthetic medical text and term extraction dataset generated with ChatGPT, containing 50 documents and more than 1,000 extracted terms.                       |
| `sample_full_terms.csv`                                         | Sample example dataset.                                                                                                                                         |

## Requirements

The q-function uses a local PubMedBERT checkpoint through Hugging Face with `local_files_only=True`. The default path is:

```text
~/.cache/huggingface/hub/models--microsoft--BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext/snapshots/e1354b7a3a09615f6aba48dfad4b7a613eef7062
```

If the checkpoint is missing, download PubMedBERT locally or set `PUBMEDBERT_PATH` / `TrainConfig.model_path` to a valid local model directory.

## Quick Start

Run commands from the project directory:

```bash
cd /Users/kylewang/Desktop/CRCmodel_evaluation_20260604/CRCModel_evaluation
```

Generate the recommended synthetic CRC plots:

```bash
python3 synthetic_bootstrap.py
```

This script reads the bundled workbook (`mimic_iii_synthetic_term_extraction_50_long_full_context.xlsx`) and simulates:

| Simulated Object  | Output                                                          |
| ----------------- | --------------------------------------------------------------- |
| `list1_df`        | `simulation_outputs/real_entry/data/test_list1.csv`             |
| `list2_df`        | `simulation_outputs/real_entry/data/test_list2.csv`             |
| `model_df`        | `simulation_outputs/real_entry/data/test_model_predictions.csv` |
| hidden full truth | `simulation_outputs/real_entry/data/test_full_truth.csv`        |

It then performs evaluation and bootstrap analysis using matcher-generated labels. Outputs include:

```text
simulation_outputs/real_entry/CRC_metrics_summary.json
simulation_outputs/real_entry/point_evaluation/evaluation_summary.json
simulation_outputs/real_entry/plots/CRC_precision.png
simulation_outputs/real_entry/plots/CRC_recall.png
simulation_outputs/real_entry/plots/CRC_type_accuracy.png
```

Plot descriptions:

| Plot                    | Description                                                                                                           |
| ----------------------- | --------------------------------------------------------------------------------------------------------------------- |
| `CRC_precision.png`     | Bootstrap distribution of naive precision and CRC-corrected precision, with hidden full-list truth indicated.         |
| `CRC_recall.png`        | Bootstrap distribution of naive recall and CRC-corrected recall, with hidden full-list truth indicated.               |
| `CRC_type_accuracy.png` | Bootstrap distribution of naive type accuracy and CRC-corrected type accuracy, with hidden full-list truth indicated. |

Most recent results using the bundled dataset:

| Metric        | Ground Truth | CRC Bootstrap Mean | Naive Bootstrap Mean |
| ------------- | ------------ | ------------------ | -------------------- |
| Precision     | 0.8094       | 0.7899             | 0.7112               |
| Recall        | 0.5502       | 0.5497             | 0.5554               |
| Type Accuracy | 0.8571       | 0.8559             | 0.8596               |

## How Synthetic Data Are Constructed

The workbook contains extracted terms and source notes. It does not contain model predictions or a `matched` column.

`synthetic_bootstrap.py` creates hidden synthetic truth and explicit model predictions:

1. For each full ground-truth term, simulate whether the model correctly extracts the phrase: `phrase_match_truth`.
2. Conditional on phrase extraction, simulate whether the semantic type is correct: `type_match_truth`.
3. Define `matched_truth = phrase_match_truth * type_match_truth`.
4. Create one row in `model_df` for each phrase-matched term. If the type is incorrect, the prediction retains the phrase but uses an incorrect semantic type.
5. Add false-positive predictions to provide a realistic precision denominator.
6. Split terms into training and test sets by document.
7. Simulate captures for `list1_df` and `list2_df` from the complete set of terms.

Ground-truth metrics use hidden columns in `test_full_truth.csv`. Naive and CRC metrics do not directly access these hidden columns; instead, they are computed from matcher outputs between visible `list1_df` / `list2_df` terms and `model_df`.

## Metric Definitions

For the hidden full truth:

```text
precision_true = sum(matched_truth) / len(model_df)
recall_true = sum(matched_truth) / number of full ground-truth terms
type_accuracy_true = sum(matched_truth) / sum(phrase_match_truth)
```

CRC uses the same capture weights for all three metrics:

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

The naive versions use the same numerators and denominators but without `w_i`, and are restricted to visible rows only.

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

| Column    | Meaning                                           |
| --------- | ------------------------------------------------- |
| `doc_id`  | Document, note, or record identifier.             |
| `phrase`  | Extracted clinical term text.                     |
| `type`    | Semantic type label or UMLS semantic type code.   |
| `context` | Source text context where the term was extracted. |

## Evaluating Real Data

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

To reuse a previously trained q-function:

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

`method="character"` is deterministic. It normalizes case, whitespace, simple punctuation, and common semantic type codes. For example, `T047` and `Disease or Syndrome` are treated as equivalent semantic types. This method is conservative and does not infer synonyms such as `high blood pressure` and `hypertension`.

### AI Matching

`method="ai"` is suitable for synonyms, abbreviations, acronyms, domain-specific terminology, and paraphrases. The built-in prompt emphasizes clinical semantic equivalence, for example:

```text
HTN <-> hypertension
DM2 <-> type 2 diabetes mellitus
```

You must provide an `ai_matcher(prompt)` function that sends the generated prompt to your AI API and returns valid JSON:

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

The default q-function input format is:

```text
phrase [SEP] semantic type <type> [SEP] context <context>
```

It trains three binary classification heads:

| Head  | Label                    |
| ----- | ------------------------ |
| `q1`  | `state` is `10` or `11`. |
| `q2`  | `state` is `01` or `11`. |
| `q12` | `state` is `11`.         |

Model predictions are not used to train the q-function. Model predictions are only used to create the matching labels required for evaluation.
