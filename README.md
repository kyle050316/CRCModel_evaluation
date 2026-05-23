# List-State Extension for CRC Evaluation

This project validates a list-to-state extension for colorectal cancer (CRC) evaluation experiments. It converts two independently sampled annotation lists into a visible state table and then uses that reconstructed table in a CRC-style simulation workflow.

## What This Project Does

The core workflow starts from two annotation lists:

- `list1`: terms identified by the first simulated annotator or extraction pass.
- `list2`: terms identified by the second simulated annotator or extraction pass.

The lists are merged by `doc_id`, `phrase`, `type`, and `context`. Each visible term receives a state label:

- `10`: present in list 1 only.
- `01`: present in list 2 only.
- `11`: present in both lists.

The reconstructed visible state table can then be used for downstream CRC correction and metric estimation.

## Main Scripts

### `build_two_lists_and_validate.py`

This script performs a focused validation of the list-to-state reconstruction.

It:

1. Loads synthetic full-term data.
2. Simulates two annotation lists using type-specific sampling probabilities.
3. Reconstructs visible states from the two lists.
4. Compares the reconstructed table against the expected simulated visible states.
5. Writes CSV outputs and a JSON validation report to `data/`.

Generated files include:

- `data/simulated_list1.csv`
- `data/simulated_list2.csv`
- `data/expected_visible_from_simulation.csv`
- `data/reconstructed_visible_states.csv`
- `data/validation_report.json`

The current validation report shows:

- `keys_equal = true`
- `state_equal = true`
- `matched_equal = true`
- `state_mismatch_count = 0`

### `run_list_state_simulation.py`

This script runs the full list-based simulation workflow.

It:

1. Reads the synthetic MIMIC-III term extraction workbook.
2. Builds normalized full-term records with phrase, semantic type, context, source category, and note text.
3. Simulates two lists for training and testing documents.
4. Reconstructs visible state tables from the simulated lists.
5. Trains a PubMedBERT-based q-function model on the reconstructed training table.
6. Runs bootstrap evaluation on the test data.
7. Compares naive visible estimates with CRC-corrected estimates.
8. Saves plots, model files, reconstructed data, and summary JSON files to `simulation_outputs/`.

Generated outputs include:

- `simulation_outputs/data/train_list1.csv`
- `simulation_outputs/data/train_list2.csv`
- `simulation_outputs/data/train_reconstructed_visible.csv`
- `simulation_outputs/data/test_list1.csv`
- `simulation_outputs/data/test_list2.csv`
- `simulation_outputs/data/test_reconstructed_visible.csv`
- `simulation_outputs/models/pubmedbert/q_function.pt`
- `simulation_outputs/models/pubmedbert/train_summary.json`
- `simulation_outputs/plots/precision_hist.png`
- `simulation_outputs/plots/recall_hist.png`
- `simulation_outputs/plots/summary_barplot.png`
- `simulation_outputs/simulation_summary.json`

## Important Dependency Note

The scripts in this directory import shared project utilities from the parent project directory:

- `crc_functions.py`
- `synthetic_pipeline.py`

When running this folder as part of the original project tree, no extra setup is needed. If this folder is cloned by itself, those shared modules must also be available on the Python path or copied into the same project structure.

## Input Data

The full simulation script expects the workbook:

```text
mimic_iii_synthetic_term_extraction_50_long_full_context-2 2.xlsx
```

This workbook is used to build the synthetic full-term table for the list-state simulation.

## How to Run

From the parent project directory:

```bash
python3 list_state_extension/build_two_lists_and_validate.py
```

To run the full simulation:

```bash
python3 list_state_extension/run_list_state_simulation.py
```

Or, from inside this directory:

```bash
python3 build_two_lists_and_validate.py
python3 run_list_state_simulation.py
```

## Python Dependencies

The scripts use:

- `pandas`
- `numpy`
- `torch`
- plotting libraries used by the shared CRC utilities
- PubMedBERT-related dependencies used by `train_q_from_excel`

The exact dependency set is inherited from the parent CRC evaluation project.

## Repository Contents

```text
.
├── README.md
├── build_two_lists_and_validate.py
├── run_list_state_simulation.py
├── mimic_iii_synthetic_term_extraction_50_long_full_context-2 2.xlsx
├── data/
└── simulation_outputs/
```

## Current Status

The focused reconstruction validation passes successfully, showing that the visible state table reconstructed from two lists matches the expected simulated visible states.
