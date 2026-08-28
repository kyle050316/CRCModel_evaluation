"""Run GENIE on the example notes and save predictions as model_df CSV.

GENIE's official output is a JSON list whose rows contain ``phrase`` and
``semantic_type``. This script converts those fields to the evaluation
interface ``doc_id, phrase, type``.
"""

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List

import pandas as pd

from run_list_state_simulation import INPUT_XLSX
from synthetic_pipeline import read_xlsx_sheet


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "genie_model_df.csv"
MODEL_NAME = "THUMedInfo/GENIE_en_8b"
PROMPT_TEMPLATE = "Human:\n{query}\n\n Assistant:"

# GENIE uses a coarser semantic-type vocabulary than this example annotation.
# Edit this small crosswalk if your ground-truth taxonomy uses other labels.
GENIE_TYPE_TO_PROJECT_TYPE = {
    "disease, syndrome or pathologic function": "diagnosis",
    "sign, symptom, or finding": "symptom",
    "chemical or drug": "medication",
    "therapeutic or preventive procedure": "treatment",
    "diagnostic procedure": "test",
    "laboratory or test result": "lab",
    "anatomical structure": "anatomy",
    "organism": "organism",
}


def load_example_notes(input_xlsx: Path, test_only: bool) -> pd.DataFrame:
    notes = read_xlsx_sheet(input_xlsx, "xl/worksheets/sheet2.xml")
    missing = [column for column in ["row_id", "text"] if column not in notes.columns]
    if missing:
        raise ValueError(f"Example note sheet is missing columns: {missing}")
    notes = notes[["row_id", "text"]].rename(columns={"row_id": "doc_id"})
    notes["doc_id"] = notes["doc_id"].astype(int)
    notes = notes.sort_values("doc_id").drop_duplicates("doc_id", keep="first").reset_index(drop=True)
    if test_only:
        split = math.ceil(len(notes) * 0.60)
        notes = notes.iloc[split:].reset_index(drop=True)
    return notes


def parse_genie_entities(generated_text: str, doc_id: int) -> List[Dict[str, object]]:
    text = generated_text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    entities = json.loads(text)
    if not isinstance(entities, list):
        raise ValueError(f"GENIE output for doc_id={doc_id} is not a JSON list")

    rows: List[Dict[str, object]] = []
    for entity in entities:
        if not isinstance(entity, dict):
            raise ValueError(f"GENIE output for doc_id={doc_id} contains a non-object row")
        phrase = str(entity.get("phrase", "") or "").strip()
        semantic_type = str(entity.get("semantic_type", "") or "").strip()
        if not phrase or not semantic_type:
            continue
        mapped_type = GENIE_TYPE_TO_PROJECT_TYPE.get(semantic_type.lower(), semantic_type)
        rows.append(
            {
                "doc_id": doc_id,
                "phrase": phrase,
                "type": mapped_type,
                "genie_semantic_type": semantic_type,
                "assertion_status": str(entity.get("assertion_status", "") or ""),
            }
        )
    return rows


def run(
    input_xlsx: Path,
    output_path: Path,
    test_only: bool,
    tensor_parallel_size: int,
    temperature: float,
    max_new_tokens: int,
) -> pd.DataFrame:
    try:
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise RuntimeError("vLLM is required for GENIE inference: pip install vllm") from exc

    notes = load_example_notes(input_xlsx, test_only=test_only)
    model = LLM(model=MODEL_NAME, tensor_parallel_size=tensor_parallel_size)
    sampling_params = SamplingParams(temperature=temperature, max_tokens=max_new_tokens)
    prompts = [PROMPT_TEMPLATE.format(query=text) for text in notes["text"].astype(str)]
    outputs = model.generate(prompts, sampling_params)

    rows: List[Dict[str, object]] = []
    for doc_id, output in zip(notes["doc_id"].astype(int), outputs):
        rows.extend(parse_genie_entities(output.outputs[0].text, int(doc_id)))

    model_df = pd.DataFrame(
        rows,
        columns=["doc_id", "phrase", "type", "genie_semantic_type", "assertion_status"],
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model_df.to_csv(output_path, index=False)
    print(f"Saved {len(model_df)} predictions from {len(notes)} notes to {output_path}")
    return model_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-xlsx", type=Path, default=INPUT_XLSX)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Run only the final 40%% of documents used as the current example test split.",
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        input_xlsx=args.input_xlsx,
        output_path=args.output,
        test_only=args.test_only,
        tensor_parallel_size=args.tensor_parallel_size,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
    )
