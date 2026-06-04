import json
import re
from typing import Callable, Dict, Iterable, List, Literal, Optional, Sequence, Tuple

import pandas as pd

try:
    from crc_functions import build_state_table_from_two_lists
except ImportError:
    def build_state_table_from_two_lists(
        list1_df: pd.DataFrame,
        list2_df: pd.DataFrame,
        key_cols: Sequence[str] = ("doc_id", "phrase", "type", "context"),
        context_col: str = "context",
        matched_col: Optional[str] = None,
    ) -> pd.DataFrame:
        for c in key_cols:
            if c not in list1_df.columns or c not in list2_df.columns:
                raise ValueError(f"Missing key column in list inputs: {c}")

        def canonicalize(df: pd.DataFrame) -> pd.DataFrame:
            out = df.copy()
            for c in key_cols:
                if c == "doc_id":
                    out[c] = out[c].astype(int)
                else:
                    out[c] = out[c].map(normalize_text)
            keep = list(dict.fromkeys(list(key_cols) + [context_col]))
            return out[keep].drop_duplicates(subset=list(key_cols), keep="first").reset_index(drop=True)

        a = canonicalize(list1_df)
        b = canonicalize(list2_df)
        merged = a.merge(b, on=list(key_cols), how="outer", suffixes=("_1", "_2"), indicator=True)
        merged["state"] = merged["_merge"].map({"left_only": "10", "right_only": "01", "both": "11"})
        if context_col in key_cols:
            merged["context"] = merged[context_col].fillna("")
        else:
            merged["context"] = merged[f"{context_col}_1"].where(
                merged[f"{context_col}_1"].astype(str).str.len() > 0,
                merged[f"{context_col}_2"],
            ).fillna("")
        out_cols = list(dict.fromkeys(list(key_cols) + ["state", "context"]))
        return merged[out_cols].sort_values(["doc_id", "phrase", "type"]).reset_index(drop=True)

try:
    from semantic_types import semantic_type_text
except ImportError:
    SEMANTIC_TYPE_TEXT = {
        "T033": "Finding",
        "T047": "Disease or Syndrome",
        "T103": "Chemical",
        "T121": "Pharmacologic Substance",
        "T184": "Sign or Symptom",
        "T200": "Clinical Drug",
    }

    def semantic_type_text(type_code: object) -> str:
        if isinstance(type_code, list) and type_code:
            type_code = type_code[0]
        type_code = str(type_code or "").strip()
        return SEMANTIC_TYPE_TEXT.get(type_code, type_code)


MatchMethod = Literal["character", "ai"]
AiMatcher = Callable[[str], str | Dict[str, object] | List[Dict[str, object]]]


AI_MATCH_PROMPT_TEMPLATE = """You are evaluating pairs of clinical/medical terms extracted from the SAME piece of text.

We have:
- text: the source text
- human: a list of human-annotated terms with {{"idx","phrases","type"}}
- model: a list of model-predicted terms with {{"idx","phrases","type"}}

Task:
Return the best one-to-one matches between human and model terms.

PHRASE:
- Compare the meaning of the two terms in THIS text.
- Normalize case and whitespace.
- Treat clear synonyms, abbreviations, and paraphrases as equal only when unambiguous in THIS text.
- phrase_match=true ONLY when the two terms refer to essentially the SAME clinical concept with the SAME level of specificity and scope.
- Do NOT match generic vs specific or supertype vs subtype.
  Examples where phrase_match=false:
  - "back pain" vs "low back pain"
  - "pneumonia" vs "aspiration pneumonia"
  - "diabetes" vs "type 2 diabetes mellitus"
- Allow clear synonyms / common lay vs professional terms.
  Examples where phrase_match=true:
  - "myocardial infarction" vs "heart attack"
  - "high blood pressure" vs "hypertension"
  - "SOB" vs "shortness of breath" when the text makes that abbreviation clear
- If unsure whether they are exactly the same concept, default to phrase_match=false.

TYPE:
- A coarse semantic category, such as a UMLS semantic type code or English label.
- Treat UMLS code and canonical English label as equivalent.
  Examples:
  - T033 <-> Finding
  - T047 <-> Disease or Syndrome
  - T184 <-> Sign or Symptom
  - T103 <-> Chemical
  - T121 <-> Pharmacologic Substance
  - T200 <-> Clinical Drug
- TYPE is only evaluated when phrase_match=true.
- type_match=true only if the semantic categories are equivalent or clinically compatible at the same coarse level.
- If phrase_match=false, type_match must be false.

One-to-one constraint:
- Each human term h_idx can be paired with AT MOST one model term g_idx.
- Each model term g_idx can be paired with AT MOST one human term h_idx.
- Only consider pairs within this same record.
- Prefer the highest-confidence exact concept match.
- Do not create a match merely because terms are related, adjacent, causal, or co-mentioned.

Output:
- Return ONLY valid JSON. No markdown, no commentary.
- Schema:
{{
  "matches": [
    {{
      "h_idx": 0,
      "g_idx": 2,
      "phrase_match": true,
      "type_match": true,
      "confidence": 0.0,
      "reason": "short reason"
    }}
  ]
}}
- Include only pairs where phrase_match=true.
- confidence must be between 0 and 1.

text:
{text}

human:
{human_json}

model:
{model_json}
"""


def normalize_text(value: object) -> str:
    return " ".join(str(value or "").strip().lower().split())


def normalize_phrase_for_character_match(value: object) -> str:
    text = normalize_text(value)
    text = re.sub(r"[^\w\s.+/-]", " ", text)
    return " ".join(text.split())


def canonical_type(value: object) -> str:
    if isinstance(value, list) and value:
        value = value[0]
    label = semantic_type_text(value)
    return normalize_text(label)


def term_phrase(term: Dict[str, object]) -> str:
    return str(term.get("phrases", term.get("phrase", "")) or "")


def term_type(term: Dict[str, object]) -> str:
    return str(term.get("type", "") or "")


def dataframe_to_terms(df: pd.DataFrame) -> List[Dict[str, object]]:
    terms: List[Dict[str, object]] = []
    for i, row in df.reset_index(drop=True).iterrows():
        terms.append(
            {
                "idx": int(row["idx"]) if "idx" in row and pd.notna(row["idx"]) else i,
                "phrases": str(row.get("phrases", row.get("phrase", "")) or ""),
                "type": str(row.get("type", "") or ""),
            }
        )
    return terms


def build_ai_match_prompt(text: str, human_terms: Sequence[Dict[str, object]], model_terms: Sequence[Dict[str, object]]) -> str:
    human_json = json.dumps(list(human_terms), ensure_ascii=False, indent=2)
    model_json = json.dumps(list(model_terms), ensure_ascii=False, indent=2)
    return AI_MATCH_PROMPT_TEMPLATE.format(text=text or "", human_json=human_json, model_json=model_json)


def character_match_terms(
    human_terms: Sequence[Dict[str, object]],
    model_terms: Sequence[Dict[str, object]],
    require_type_match: bool = True,
) -> List[Dict[str, object]]:
    matches: List[Dict[str, object]] = []
    used_model: set[int] = set()
    model_keys: Dict[Tuple[str, str], List[int]] = {}
    phrase_keys: Dict[str, List[int]] = {}

    for pos, term in enumerate(model_terms):
        phrase = normalize_phrase_for_character_match(term_phrase(term))
        typ = canonical_type(term_type(term))
        model_keys.setdefault((phrase, typ), []).append(pos)
        phrase_keys.setdefault(phrase, []).append(pos)

    for h_pos, h_term in enumerate(human_terms):
        h_phrase = normalize_phrase_for_character_match(term_phrase(h_term))
        h_type = canonical_type(term_type(h_term))
        candidates = model_keys.get((h_phrase, h_type), []) if require_type_match else phrase_keys.get(h_phrase, [])
        for g_pos in candidates:
            if g_pos in used_model:
                continue
            used_model.add(g_pos)
            g_term = model_terms[g_pos]
            type_match = canonical_type(term_type(g_term)) == h_type
            matches.append(
                {
                    "h_idx": int(h_term.get("idx", h_pos)),
                    "g_idx": int(g_term.get("idx", g_pos)),
                    "phrase_match": True,
                    "type_match": bool(type_match),
                    "confidence": 1.0,
                    "reason": "normalized exact phrase and type match" if type_match else "normalized exact phrase match",
                }
            )
            break
    return matches


def parse_ai_matches(payload: str | Dict[str, object] | List[Dict[str, object]]) -> List[Dict[str, object]]:
    if isinstance(payload, str):
        parsed = json.loads(payload)
    else:
        parsed = payload
    if isinstance(parsed, dict):
        parsed_matches = parsed.get("matches", [])
    else:
        parsed_matches = parsed
    if not isinstance(parsed_matches, list):
        raise ValueError("AI matcher output must be a JSON object with a 'matches' list or a list of match objects")

    matches: List[Dict[str, object]] = []
    used_h: set[int] = set()
    used_g: set[int] = set()
    for item in parsed_matches:
        if not isinstance(item, dict):
            continue
        if not item.get("phrase_match", False):
            continue
        h_idx = int(item["h_idx"])
        g_idx = int(item["g_idx"])
        if h_idx in used_h or g_idx in used_g:
            continue
        used_h.add(h_idx)
        used_g.add(g_idx)
        matches.append(
            {
                "h_idx": h_idx,
                "g_idx": g_idx,
                "phrase_match": True,
                "type_match": bool(item.get("type_match", False)),
                "confidence": float(item.get("confidence", 0.0)),
                "reason": str(item.get("reason", "")),
            }
        )
    return matches


def match_terms(
    text: str,
    human_terms: Sequence[Dict[str, object]],
    model_terms: Sequence[Dict[str, object]],
    method: MatchMethod = "character",
    ai_matcher: Optional[AiMatcher] = None,
    require_type_match: bool = True,
) -> List[Dict[str, object]]:
    if method == "character":
        return character_match_terms(human_terms, model_terms, require_type_match=require_type_match)
    if method == "ai":
        if ai_matcher is None:
            raise ValueError("ai_matcher is required when method='ai'")
        prompt = build_ai_match_prompt(text, human_terms, model_terms)
        return parse_ai_matches(ai_matcher(prompt))
    raise ValueError(f"Unknown match method: {method}")


def add_matched_from_model_predictions(
    human_visible_df: pd.DataFrame,
    model_df: pd.DataFrame,
    method: MatchMethod = "character",
    ai_matcher: Optional[AiMatcher] = None,
    require_type_match: bool = True,
    doc_col: str = "doc_id",
    context_col: str = "context",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    required_human = {doc_col, "phrase", "type"}
    required_model = {doc_col, "type"}
    missing_human = sorted(required_human - set(human_visible_df.columns))
    missing_model = sorted(required_model - set(model_df.columns))
    if "phrase" not in model_df.columns and "phrases" not in model_df.columns:
        missing_model.append("phrase or phrases")
    if missing_human:
        raise ValueError(f"human_visible_df missing columns: {missing_human}")
    if missing_model:
        raise ValueError(f"model_df missing columns: {missing_model}")

    out = human_visible_df.copy().reset_index(drop=True)
    model_df = model_df.copy()
    if "phrase" not in model_df.columns and "phrases" in model_df.columns:
        model_df["phrase"] = model_df["phrases"]
    out["_row_id"] = out.index
    out["matched"] = 0
    match_rows: List[Dict[str, object]] = []

    model_by_doc = {doc_id: group.reset_index(drop=True) for doc_id, group in model_df.groupby(doc_col, sort=False)}
    for doc_id, h_group in out.groupby(doc_col, sort=False):
        m_group = model_by_doc.get(doc_id)
        if m_group is None or m_group.empty:
            continue
        h_reset = h_group.reset_index(drop=True).copy()
        m_reset = m_group.reset_index(drop=True).copy()
        h_reset["idx"] = h_reset.index
        m_reset["idx"] = m_reset.index

        text = ""
        if context_col in h_reset.columns and len(h_reset) > 0:
            text = str(h_reset.iloc[0].get(context_col, "") or "")
        elif context_col in m_reset.columns and len(m_reset) > 0:
            text = str(m_reset.iloc[0].get(context_col, "") or "")

        human_terms = dataframe_to_terms(h_reset)
        model_terms = dataframe_to_terms(m_reset)
        matches = match_terms(
            text,
            human_terms,
            model_terms,
            method=method,
            ai_matcher=ai_matcher,
            require_type_match=require_type_match,
        )

        for match in matches:
            h_idx = int(match["h_idx"])
            g_idx = int(match["g_idx"])
            if h_idx < 0 or h_idx >= len(h_reset) or g_idx < 0 or g_idx >= len(m_reset):
                continue
            source_row = int(h_reset.iloc[h_idx]["_row_id"])
            out.loc[source_row, "matched"] = 1
            match_rows.append(
                {
                    "doc_id": doc_id,
                    "human_row_id": source_row,
                    "model_row_id": int(m_reset.index[g_idx]),
                    "human_phrase": h_reset.iloc[h_idx]["phrase"],
                    "model_phrase": m_reset.iloc[g_idx]["phrase"],
                    "human_type": h_reset.iloc[h_idx]["type"],
                    "model_type": m_reset.iloc[g_idx]["type"],
                    "phrase_match": bool(match.get("phrase_match", True)),
                    "type_match": bool(match.get("type_match", False)),
                    "confidence": float(match.get("confidence", 0.0)),
                    "reason": str(match.get("reason", "")),
                }
            )

    return out.drop(columns=["_row_id"]), pd.DataFrame(match_rows)


def build_evaluation_table_from_two_lists_and_model(
    list1_df: pd.DataFrame,
    list2_df: pd.DataFrame,
    model_df: pd.DataFrame,
    method: MatchMethod = "character",
    ai_matcher: Optional[AiMatcher] = None,
    require_type_match: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    human_visible = build_state_table_from_two_lists(
        list1_df,
        list2_df,
        key_cols=("doc_id", "phrase", "type", "context"),
        context_col="context",
        matched_col=None,
    )
    return add_matched_from_model_predictions(
        human_visible,
        model_df,
        method=method,
        ai_matcher=ai_matcher,
        require_type_match=require_type_match,
    )
