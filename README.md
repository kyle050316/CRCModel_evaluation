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
| `two_annotations_pseudo_truth.py`                               | Uses the union of two annotations as ground truth, thins each annotation with `p1(z)` / `p2(z)`, and compares naive and CRC estimates.                            |
| `type_match_accuracy.py`                                        | Evaluates type extraction accuracy when the model has already extracted the correct phrase.                                                                     |
| `model_term_matching.py`                                        | Character-based or AI-assisted term matching utilities.                                                                                                         |
| `simulation_bootstrap_validation.py`                            | Draw plot based on bootstrap simulation `run_list_state_simulation.py`.                                                                                         |
| `crc_functions.py`                                              | Helper functions for state-table construction, q-function training, and output generation.                                                                      |
| `synthetic_pipeline.py`                                         | Synthetic workbook loading, sampling probabilities, bootstrap utilities, and plotting helpers.                                                                  |
| `mimic_iii_synthetic_term_extraction_50_long_full_context-2.xlsx` | A synthetic medical text and term extraction dataset generated with ChatGPT, containing 50 documents and more than 1,000 extracted terms.                       |
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
cd /Users/kylewang/Desktop/CRCmodel_evaluation_20260813/CRCModel_evaluation_github
```

Generate the recommended synthetic CRC plots:

```bash
python3 synthetic_bootstrap.py
```

This script reads the bundled workbook (`mimic_iii_synthetic_term_extraction_50_long_full_context-2.xlsx`) and simulates:

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




## Simulating the Algorithm with ground truth annotation

When only one complete ground-truth annotation is available, treat it as a finite population of terms rather than as an observed list. Represent each term as

```text
z = (doc_id, phrase, type, context)
```

and assign two capture probabilities:

```text
p1(z) = P(z is selected into list1 | z)
p2(z) = P(z is selected into list2 | z)
```

Generate the two observed lists using independent Bernoulli draws conditional on `z`:

```python
r1 = Bernoulli(p1(z))
r2 = Bernoulli(p2(z))

list1_df = truth_df[r1 == 1][["doc_id", "phrase", "type", "context"]]
list2_df = truth_df[r2 == 1][["doc_id", "phrase", "type", "context"]]
```

This produces the four capture states `00`, `10`, `01`, and `11`. Terms in state `00` remain hidden, while `10`, `01`, and `11` are reconstructed by `build_state_table_from_two_lists()`.

Under conditional independence,

```text
p12(z) = p1(z) * p2(z).
```

In applications where the two lists may be dependent, this dependence should instead be represented by the learned `q12` head or by a simulation model that generates dependent captures.

### Simulation Procedure

1. Start with a complete ground-truth table containing `doc_id`, `phrase`, `type`, and `context`.
2. Define `p1(z)` and `p2(z)`. These probabilities may depend on term type, phrase difficulty, context length, document characteristics, or other known covariates.
3. Sample `r1` and `r2` for each ground-truth term to construct `list1_df` and `list2_df`.
4. Create or provide `model_df` independently. It must contain `doc_id`, `phrase`, and `type`, with each row representing a model prediction.
5. Pass the three observed tables to the standard evaluation pipeline:

```python
from evaluate_two_lists_model import evaluate_two_lists_with_model

summary = evaluate_two_lists_with_model(
    list1_df=list1_df,
    list2_df=list2_df,
    model_df=model_df,
    output_dir="simulation_outputs/one_ground_truth",
    method="character",
    require_type_match=True,
    pred_total=len(model_df),
)
```

The estimator does not use the hidden full ground truth. It reconstructs the visible capture states from `list1_df` and `list2_df`, matches the visible human terms to `model_df`, trains the q-function from the matched-free state table, and returns both naive and CRC-corrected estimates.

The original `truth_df` is retained only for simulation validation, such as comparing the estimated metrics with metrics computed directly from the complete ground truth.



### Comparison with Traditional Evaluation

The simulated data can be evaluated in three ways:

| Method | Input | Interpretation |
| --- | --- | --- |
| Direct full-ground-truth evaluation | `truth_df` + `model_df` | Reference value available because the simulation exposes the complete truth |
| Traditional/naive evaluation | Visible union of `list1_df` and `list2_df` + `model_df` | Uses only terms captured at least once and may therefore be biased under incomplete capture; corresponds to `naive_*` |
| Two-list CRC evaluation | `list1_df` + `list2_df` + `model_df` | Uses estimated `q1`, `q2`, and `q12` to account for terms missed by both lists; corresponds to `corrected_*` |

For a simulated benchmark, report the full-ground-truth, naive, and CRC-corrected values together. Repeat the Bernoulli sampling across multiple random seeds or bootstrap draws, and summarize bias, standard deviation, and coverage.

The CRC method is successful when its estimates are systematically closer to the direct full-ground-truth values than the naive baseline across the target metrics.

The repository already implements this pattern in `synthetic_bootstrap.py`: `simulate_two_lists()` samples terms using `sampling_probabilities(type)`, `make_evaluation_table()` combines the simulated lists with synthetic model predictions, and `run_bootstrap()` compares CRC-corrected and naive estimates against the hidden-truth metrics.

## Example: Two Annotations Without Complete Ground Truth

This example has five steps.

### 1. Input Two Annotations

Both inputs contain one row per mention:

```text
doc_id, phrase, type, context
```

In Python they are `annotation1_df` and `annotation2_df`.

### 2. Merge Them into Ground Truth

```python
from two_annotations_pseudo_truth import prepare_annotations_and_ground_truth

_, _, ground_truth = prepare_annotations_and_ground_truth(
    annotation1_df,
    annotation2_df,
)
```

The function uses `build_state_table_from_two_lists()` to normalize the four fields and perform a multiset union. It keeps duplicates inside each annotation and removes only overlap between annotations. If the same normalized row occurs `m` times in annotation 1 and `n` times in annotation 2, ground truth keeps `max(m, n)` rows. Different contexts remain different mentions.

### 3. Split Ground Truth into Two Sublists

For each term `z=(doc_id, phrase, type, context)`:

```text
pj(z) = clip(1 - base_deletion_j + type_adjustment[type(z)], 0.05, 0.95)
```

Defaults are list 1 deletion `18%` and list 2 deletion `23%`. Type adjustments are defined at the top of `two_annotations_pseudo_truth.py`.

```python
from two_annotations_pseudo_truth import thin_annotation_lists

sublist1, sublist2 = thin_annotation_lists(
    ground_truth,
    ground_truth,
    seed=20260523,
    list1_deletion_proportion=0.18,
    list2_deletion_proportion=0.23,
)
```

Every ground-truth term can enter either sublist. The two Bernoulli draws are independent conditional on `z`.

### 4. Run the Algorithm

```python
from evaluate_two_lists_model import evaluate_two_lists_with_model

summary = evaluate_two_lists_with_model(
    list1_df=sublist1,
    list2_df=sublist2,
    model_df=model_df,
    output_dir="evaluation_outputs/two_annotations",
    method="character",
    require_type_match=True,
    pred_total=len(model_df),
)
```

Run the complete example with 500 bootstrap repetitions:

```bash
python3 two_annotations_pseudo_truth.py --bootstrap 500
```

Main outputs are:

| Output | Meaning |
| --- | --- |
| `data/test_ground_truth.csv` | Merged ground truth. |
| `data/test_list1.csv`, `test_list2.csv` | Simulated sublists used by CRC. |
| `CRC_metrics_summary.json` | Counts and metric summaries. |
| `plots/CRC_*.png` | Bootstrap plots. |

### 5. Explain the Results

The default run uses 18%/23% base deletion and 500 bootstrap repetitions.

| Count | Value |
| --- | ---: |
| Input annotation 1 | 472 |
| Input annotation 2 | 414 |
| Merged ground truth | 597 |
| Simulated sublist 1 | 509 |
| Simulated sublist 2 | 492 |
| Visible sublist union | 576 |

The actual deletion rates were `14.74%` and `17.59%` after type adjustment.

| Metric | Ground truth | CRC mean (SD) | Naive mean (SD) |
| --- | ---: | ---: | ---: |
| Precision | 0.7152 | 0.7122 (0.0073) | 0.6981 (0.0058) |
| Recall | 0.5595 | 0.5590 (0.0039) | 0.5605 (0.0033) |
| Type accuracy | 0.8542 | 0.8536 (0.0037) | 0.8545 (0.0031) |

![Precision bootstrap](simulation_outputs/two_annotations_pseudo_truth_p_z/plots/CRC_precision.png)

CRC precision is `0.0030` below ground truth; naive precision is `0.0171` below it. CRC corrects most of the loss caused by terms missing from both sublists.

![Recall bootstrap](simulation_outputs/two_annotations_pseudo_truth_p_z/plots/CRC_recall.png)

Both recall estimates are close. CRC differs from ground truth by `-0.0005`; naive differs by `+0.0010`.

![Type-accuracy bootstrap](simulation_outputs/two_annotations_pseudo_truth_p_z/plots/CRC_type_accuracy.png)

Both type-accuracy estimates are close. Naive is slightly closer here. CRC has slightly larger variance because it uses inverse-probability weights.

Ground truth is used only for validation. The algorithm receives `sublist1`, `sublist2`, and `model_df`.






## Evaluation Metrics

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
