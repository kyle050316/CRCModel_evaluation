# CRC Model Evaluation Minimal Demo

This repository contains a lightweight, self-contained demo for reconstructing visible states from two partially observed clinical term lists.

The bundled `sample_full_terms.csv` provides demo data. Running the script simulates two annotation lists, reconstructs the visible `10` / `01` / `11` state table, validates the reconstruction against the simulated truth, and writes generated outputs under `data/`.

## Files

- `build_two_lists_and_validate.py`: runnable demo entry point.
- `crc_functions.py`: lightweight helpers for table writing and two-list state reconstruction.
- `synthetic_pipeline.py`: demo data loading and deterministic sampling utilities.
- `sample_full_terms.csv`: bundled sample clinical-term data.
- `requirements.txt`: minimal Python dependencies.

## Run

```bash
pip install -r requirements.txt
python3 build_two_lists_and_validate.py
```

Expected output includes a JSON report like:

```json
{
  "keys_equal": true,
  "state_equal": true,
  "matched_equal": true
}
```

Generated files are written to `data/` and are intentionally ignored by Git.

## Data Input

The demo uses `sample_full_terms.csv` by default. The expected columns are:

```text
doc_id, phrase, type, context, source_category, note_text, matched
```

If you place the full workbook `mimic_iii_synthetic_term_extraction_50_long_full_context-2.xlsx` in the repository root, `synthetic_pipeline.py` can read that instead.

## State Meaning

| State | Meaning |
| --- | --- |
| `10` | Term appears in list 1 only |
| `01` | Term appears in list 2 only |
| `11` | Term appears in both lists |

Rows with state `00` are unobserved and are not included in the reconstructed visible table.
