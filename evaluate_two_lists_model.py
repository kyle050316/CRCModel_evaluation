import json
import os
import re
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer


MatchMethod = Literal["character", "ai"]
AiMatcher = Callable[[str], str | Dict[str, object] | List[Dict[str, object]]]
PUBMEDBERT_PATH = os.environ.get(
    "PUBMEDBERT_PATH",
    str(Path.home() / ".cache/huggingface/hub/models--microsoft--BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext/snapshots/e1354b7a3a09615f6aba48dfad4b7a613eef7062"),
)
REQUIRED_COLUMNS = ["doc_id", "phrase", "type", "state", "context"]
EPS = 1e-6
MAX_ABS_WEIGHT = 1e6


def relative_to_cwd(path: str | os.PathLike) -> str:
    path_obj = Path(path).resolve()
    try:
        return path_obj.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return str(path_obj)


AI_MATCH_PROMPT_TEMPLATE = """You are evaluating pairs of clinical/medical terms extracted from the SAME piece of text.

We have:
- text: the source text
- human: a list of human-annotated terms with {{"idx","phrases","type"}}
- model: a list of model-predicted terms with {{"idx","phrases","type"}}

Task:
Return the best one-to-one matches between human and model terms.

PHRASE:
- Compare the clinical meaning of the two terms in THIS text, not just their surface characters.
- Normalize case and whitespace.
- Treat clear medical synonyms, professional-vs-lay terms, abbreviations, acronyms, spelling variants, and paraphrases as equal only when unambiguous in THIS text.
- Expand and interpret abbreviations using the local context when possible. For example, match an abbreviation to its full form only when the surrounding text supports that meaning.
- Prefer semantic equivalence over exact wording. Different strings can match if they denote the same clinical concept.
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
  - "HTN" vs "hypertension"
  - "DM2" vs "type 2 diabetes mellitus"
  - "SOB" vs "shortness of breath" when the text makes that abbreviation clear
  - "ceftriaxone" vs "Rocephin" when they refer to the same medication in context
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
- Prefer the best exact concept match.
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
      "type_match": true
    }}
  ]
}}
- Include only pairs where phrase_match=true.

text:
{text}

human:
{human_json}

model:
{model_json}
"""


SEMANTIC_TYPE_TEXT = {
    "T033": "Finding",
    "T047": "Disease or Syndrome",
    "T103": "Chemical",
    "T121": "Pharmacologic Substance",
    "T184": "Sign or Symptom",
    "T200": "Clinical Drug",
}


def normalize_text(value: object) -> str:
    return " ".join(str(value or "").strip().lower().split())


def normalize_phrase(value: object) -> str:
    text = normalize_text(value)
    text = re.sub(r"[^\w\s.+/-]", " ", text)
    return " ".join(text.split())


def canonical_type(value: object) -> str:
    if isinstance(value, list) and value:
        value = value[0]
    value = str(value or "").strip()
    return normalize_text(SEMANTIC_TYPE_TEXT.get(value, value))


def build_state_table_from_two_lists(
    list1_df: pd.DataFrame,
    list2_df: pd.DataFrame,
    key_cols: Sequence[str] = ("doc_id", "phrase", "type", "context"),
) -> pd.DataFrame:
    for col in key_cols:
        if col not in list1_df.columns or col not in list2_df.columns:
            raise ValueError(f"Both lists must contain key column: {col}")

    def canonicalize(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for col in key_cols:
            if col == "doc_id":
                out[col] = out[col].astype(int)
            elif col == "type":
                out[col] = out[col].map(canonical_type)
            else:
                out[col] = out[col].map(normalize_text)
        return out[list(key_cols)].drop_duplicates(subset=list(key_cols), keep="first").reset_index(drop=True)

    left = canonicalize(list1_df)
    right = canonicalize(list2_df)
    merged = left.merge(right, on=list(key_cols), how="outer", indicator=True)
    merged["state"] = merged["_merge"].map({"left_only": "10", "right_only": "01", "both": "11"}).astype(str)
    return merged[list(key_cols) + ["state"]].sort_values(["doc_id", "phrase", "type"]).reset_index(drop=True)


def dataframe_to_terms(df: pd.DataFrame) -> List[Dict[str, object]]:
    terms: List[Dict[str, object]] = []
    for i, row in df.reset_index(drop=True).iterrows():
        terms.append(
            {
                "idx": i,
                "phrases": str(row.get("phrases", row.get("phrase", "")) or ""),
                "type": str(row.get("type", "") or ""),
            }
        )
    return terms


def build_ai_match_prompt(text: str, human_terms: Sequence[Dict[str, object]], model_terms: Sequence[Dict[str, object]]) -> str:
    return AI_MATCH_PROMPT_TEMPLATE.format(
        text=text or "",
        human_json=json.dumps(list(human_terms), ensure_ascii=False, indent=2),
        model_json=json.dumps(list(model_terms), ensure_ascii=False, indent=2),
    )


def character_match_terms(
    human_terms: Sequence[Dict[str, object]],
    model_terms: Sequence[Dict[str, object]],
    require_type_match: bool = True,
) -> List[Dict[str, object]]:
    model_by_key: Dict[Tuple[str, str], List[int]] = {}
    model_by_phrase: Dict[str, List[int]] = {}
    for i, term in enumerate(model_terms):
        phrase = normalize_phrase(term.get("phrases", term.get("phrase", "")))
        typ = canonical_type(term.get("type", ""))
        model_by_key.setdefault((phrase, typ), []).append(i)
        model_by_phrase.setdefault(phrase, []).append(i)

    used_model: set[int] = set()
    matches: List[Dict[str, object]] = []
    for h_idx, term in enumerate(human_terms):
        phrase = normalize_phrase(term.get("phrases", term.get("phrase", "")))
        typ = canonical_type(term.get("type", ""))
        candidates = model_by_key.get((phrase, typ), []) if require_type_match else model_by_phrase.get(phrase, [])
        for g_idx in candidates:
            if g_idx in used_model:
                continue
            used_model.add(g_idx)
            type_match = canonical_type(model_terms[g_idx].get("type", "")) == typ
            matches.append(
                {
                    "h_idx": h_idx,
                    "g_idx": g_idx,
                    "phrase_match": True,
                    "type_match": type_match,
                    "confidence": 1.0,
                    "reason": "normalized exact phrase and type match" if type_match else "normalized exact phrase match",
                }
            )
            break
    return matches


def parse_ai_matches(payload: str | Dict[str, object] | List[Dict[str, object]]) -> List[Dict[str, object]]:
    parsed = json.loads(payload) if isinstance(payload, str) else payload
    raw_matches = parsed.get("matches", []) if isinstance(parsed, dict) else parsed
    if not isinstance(raw_matches, list):
        raise ValueError("AI matcher must return JSON with a 'matches' list")

    used_h: set[int] = set()
    used_g: set[int] = set()
    matches: List[Dict[str, object]] = []
    for item in raw_matches:
        if not isinstance(item, dict) or not item.get("phrase_match", False):
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
        return parse_ai_matches(ai_matcher(build_ai_match_prompt(text, human_terms, model_terms)))
    raise ValueError(f"Unknown match method: {method}")


def make_evaluation_table(
    list1_df: pd.DataFrame,
    list2_df: pd.DataFrame,
    model_df: pd.DataFrame,
    method: MatchMethod = "character",
    ai_matcher: Optional[AiMatcher] = None,
    require_type_match: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    visible = build_state_table_from_two_lists(list1_df, list2_df)
    model = model_df.copy()
    if "phrase" not in model.columns and "phrases" in model.columns:
        model["phrase"] = model["phrases"]
    for col in ["doc_id", "phrase", "type"]:
        if col not in model.columns:
            raise ValueError(f"model_df must contain column: {col}")

    visible = visible.reset_index(drop=True)
    visible["matched"] = 0
    match_rows: List[Dict[str, object]] = []
    model_by_doc = {doc_id: group.reset_index(drop=True) for doc_id, group in model.groupby("doc_id", sort=False)}

    for doc_id, human_group in visible.groupby("doc_id", sort=False):
        model_group = model_by_doc.get(doc_id)
        if model_group is None or model_group.empty:
            continue
        human_group = human_group.reset_index().rename(columns={"index": "_source_row"})
        text = str(human_group.iloc[0].get("context", "") or "")
        human_terms = dataframe_to_terms(human_group)
        model_terms = dataframe_to_terms(model_group)
        matches = match_terms(text, human_terms, model_terms, method, ai_matcher, require_type_match)
        for match in matches:
            h_idx = int(match["h_idx"])
            g_idx = int(match["g_idx"])
            if h_idx < 0 or h_idx >= len(human_group) or g_idx < 0 or g_idx >= len(model_group):
                continue
            source_row = int(human_group.iloc[h_idx]["_source_row"])
            visible.loc[source_row, "matched"] = 1
            match_row = {
                "doc_id": doc_id,
                "human_row": source_row,
                "model_row": int(model_group.index[g_idx]),
                "human_phrase": human_group.iloc[h_idx]["phrase"],
                "model_phrase": model_group.iloc[g_idx]["phrase"],
                "human_type": human_group.iloc[h_idx]["type"],
                "model_type": model_group.iloc[g_idx]["type"],
                "phrase_match": bool(match.get("phrase_match", True)),
                "type_match": bool(match.get("type_match", False)),
            }
            if "confidence" in match:
                match_row["confidence"] = float(match["confidence"])
            if "reason" in match:
                match_row["reason"] = str(match["reason"])
            match_rows.append(match_row)
    return visible, pd.DataFrame(match_rows)


@dataclass
class TrainConfig:
    model_name: str = "two_list_pubmedbert"
    model_path: str = PUBMEDBERT_PATH
    hidden_dim: int = 64
    dropout: float = 0.35
    lr: float = 1e-3
    weight_decay: float = 5e-4
    epochs: int = 40
    patience: int = 8
    batch_size_embed: int = 64
    batch_size_head: int = 64
    max_length: int = 96
    val_frac: float = 0.2
    seed: int = 2026
    device_name: str = "cpu"
    use_context: bool = True


class RowDataset(Dataset):
    def __init__(self, emb: np.ndarray, y: np.ndarray) -> None:
        self.emb = torch.tensor(emb, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return self.emb.shape[0]

    def __getitem__(self, idx: int):
        return self.emb[idx], self.y[idx]


class BinaryHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x)).squeeze(-1)


@dataclass
class QFunction:
    config: TrainConfig
    input_dim: int
    q1_state: Dict[str, torch.Tensor]
    q2_state: Dict[str, torch.Tensor]
    q12_state: Dict[str, torch.Tensor]

    def _head(self, state: Dict[str, torch.Tensor], device: torch.device) -> BinaryHead:
        head = BinaryHead(self.input_dim, self.config.hidden_dim, self.config.dropout).to(device)
        head.load_state_dict(state)
        head.eval()
        return head

    def predict_dataframe(self, df: pd.DataFrame) -> np.ndarray:
        device = torch.device(self.config.device_name)
        emb = embed_rows(df, self.config)
        heads = [self._head(s, device) for s in [self.q1_state, self.q2_state, self.q12_state]]
        preds = [predict_head(h, emb, self.config) for h in heads]
        q = np.stack(preds, axis=1).astype(np.float32)
        q[:, 0] = np.clip(q[:, 0], EPS, 1.0 - EPS)
        q[:, 1] = np.clip(q[:, 1], EPS, 1.0 - EPS)
        q[:, 2] = np.clip(q[:, 2], EPS, np.minimum(q[:, 0], q[:, 1]))
        return q

    def save(self, path: str | os.PathLike) -> None:
        payload = {
            "config": self.config.__dict__,
            "input_dim": self.input_dim,
            "q1_state": {k: v.cpu() for k, v in self.q1_state.items()},
            "q2_state": {k: v.cpu() for k, v in self.q2_state.items()},
            "q12_state": {k: v.cpu() for k, v in self.q12_state.items()},
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str | os.PathLike, device_name: str = "cpu") -> "QFunction":
        payload = torch.load(path, map_location="cpu")
        config = TrainConfig(**payload["config"])
        config.device_name = device_name
        return cls(
            config=config,
            input_dim=int(payload["input_dim"]),
            q1_state=payload["q1_state"],
            q2_state=payload["q2_state"],
            q12_state=payload["q12_state"],
        )


def write_table(df: pd.DataFrame, path: str | os.PathLike) -> None:
    path = str(path)
    os.makedirs(str(Path(path).parent), exist_ok=True)
    if path.endswith(".xlsx"):
        df.to_excel(path, index=False)
    else:
        df.to_csv(path, index=False)


def read_table(path: str | os.PathLike) -> pd.DataFrame:
    path = str(path)
    if path.endswith(".xlsx") or path.endswith(".xls"):
        df = pd.read_excel(path, dtype={"state": str, "type": str, "phrase": str, "context": str})
    else:
        df = pd.read_csv(path, dtype={"state": str, "type": str, "phrase": str, "context": str})
    if "state" in df.columns:
        df["state"] = df["state"].astype(str).str.strip().str.zfill(2)
    return df


def validate_table(df: pd.DataFrame, require_matched: bool = False) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if require_matched and "matched" not in df.columns:
        missing.append("matched")
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    bad = sorted(set(df["state"].astype(str)) - {"10", "01", "11"})
    if bad:
        raise ValueError(f"state must be one of 10, 01, 11; found {bad}")


def make_text_inputs(df: pd.DataFrame, use_context: bool = True) -> List[str]:
    phrase = df["phrase"].fillna("").astype(str).str.strip()
    typ = df["type"].fillna("").astype(str).str.strip()
    text = phrase + " [SEP] semantic type " + typ
    if use_context and "context" in df.columns:
        context = df["context"].fillna("").astype(str).str.strip()
        text = text + " [SEP] context " + context
    return text.tolist()


def embed_texts(texts: List[str], config: TrainConfig) -> np.ndarray:
    if not texts:
        return np.zeros((0, 768), dtype=np.float32)
    device = torch.device(config.device_name)
    tokenizer = AutoTokenizer.from_pretrained(config.model_path, local_files_only=True)
    encoder = AutoModel.from_pretrained(config.model_path, local_files_only=True).to(device)
    encoder.eval()
    chunks: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(texts), config.batch_size_embed):
            batch = texts[start:start + config.batch_size_embed]
            toks = tokenizer(batch, padding=True, truncation=True, max_length=config.max_length, return_tensors="pt")
            toks = {k: v.to(device) for k, v in toks.items()}
            hs = encoder(**toks).last_hidden_state
            mask = toks["attention_mask"].float()
            pooled = (hs * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1.0)
            chunks.append(pooled.cpu().numpy().astype(np.float32))
    return np.concatenate(chunks, axis=0)


def embed_rows(df: pd.DataFrame, config: TrainConfig) -> np.ndarray:
    return embed_texts(make_text_inputs(df, use_context=config.use_context), config)


def labels_from_state(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    state = df["state"].astype(str).values
    q1 = np.isin(state, ["10", "11"]).astype(np.float32)
    q2 = np.isin(state, ["01", "11"]).astype(np.float32)
    q12 = (state == "11").astype(np.float32)
    return q1, q2, q12


def split_indices(n: int, val_frac: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_val = int(n * val_frac)
    return idx[n_val:], idx[:n_val]


def eval_bce(head: BinaryHead, emb: np.ndarray, y: np.ndarray, config: TrainConfig) -> float:
    device = torch.device(config.device_name)
    loader = DataLoader(RowDataset(emb, y), batch_size=config.batch_size_head, shuffle=False)
    vals: List[float] = []
    loss_fn = nn.BCELoss()
    head.eval()
    with torch.no_grad():
        for x, yy in loader:
            vals.append(float(loss_fn(head(x.to(device)), yy.to(device)).item()))
    return float(np.mean(vals)) if vals else float("nan")


def predict_head(head: BinaryHead, emb: np.ndarray, config: TrainConfig) -> np.ndarray:
    device = torch.device(config.device_name)
    loader = DataLoader(RowDataset(emb, np.zeros(len(emb), dtype=np.float32)), batch_size=config.batch_size_head, shuffle=False)
    preds: List[np.ndarray] = []
    head.eval()
    with torch.no_grad():
        for x, _ in loader:
            preds.append(head(x.to(device)).cpu().numpy())
    return np.concatenate(preds, axis=0) if preds else np.zeros(0, dtype=np.float32)


def train_one_head(
    train_emb: np.ndarray,
    train_y: np.ndarray,
    val_emb: np.ndarray,
    val_y: np.ndarray,
    config: TrainConfig,
) -> Tuple[BinaryHead, Dict[str, object]]:
    torch.manual_seed(config.seed)
    device = torch.device(config.device_name)
    head = BinaryHead(train_emb.shape[1], config.hidden_dim, config.dropout).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    loss_fn = nn.BCELoss()
    loader = DataLoader(RowDataset(train_emb, train_y), batch_size=config.batch_size_head, shuffle=True)
    best_state = None
    best_val = float("inf")
    patience_left = config.patience
    history: List[Dict[str, float]] = []
    for epoch in range(1, config.epochs + 1):
        head.train()
        losses: List[float] = []
        for x, y in loader:
            opt.zero_grad()
            loss = loss_fn(head(x.to(device)), y.to(device))
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        val_bce = eval_bce(head, val_emb, val_y, config)
        history.append({"epoch": epoch, "train_bce": float(np.mean(losses)), "val_bce": val_bce})
        if val_bce < best_val - 1e-6:
            best_val = val_bce
            best_state = copy.deepcopy(head.state_dict())
            patience_left = config.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break
    if best_state is not None:
        head.load_state_dict(best_state)
    return head, {"best_val_bce": best_val, "history": history}


def train_q_from_table(
    input_path: str | os.PathLike,
    out_dir: str | os.PathLike,
    config: TrainConfig,
) -> Tuple[QFunction, Dict[str, object]]:
    os.makedirs(out_dir, exist_ok=True)
    df = read_table(input_path)
    validate_table(df, require_matched=False)
    emb = embed_rows(df, config)
    train_idx, val_idx = split_indices(len(df), config.val_frac, config.seed)
    y1, y2, y12 = labels_from_state(df)
    heads = []
    summaries: Dict[str, object] = {}
    for name, y in [("q1", y1), ("q2", y2), ("q12", y12)]:
        head, summary = train_one_head(emb[train_idx], y[train_idx], emb[val_idx], y[val_idx], config)
        heads.append(head)
        summaries[name] = summary
    qf = QFunction(
        config=config,
        input_dim=emb.shape[1],
        q1_state={k: v.detach().cpu() for k, v in heads[0].state_dict().items()},
        q2_state={k: v.detach().cpu() for k, v in heads[1].state_dict().items()},
        q12_state={k: v.detach().cpu() for k, v in heads[2].state_dict().items()},
    )
    qf.save(Path(out_dir) / "q_function.pt")
    summary_out = {
        "model_name": config.model_name,
        "model_path": config.model_path,
        "n_rows": int(len(df)),
        "n_train_rows": int(len(train_idx)),
        "n_val_rows": int(len(val_idx)),
        "config": config.__dict__,
        "heads": summaries,
        "q_function_path": relative_to_cwd(Path(out_dir) / "q_function.pt"),
    }
    with open(Path(out_dir) / "train_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_out, f, indent=2, ensure_ascii=False)
    return qf, summary_out


def estimate_precision_recall(
    input_path: str | os.PathLike,
    q_function: QFunction,
    pred_total: float,
    out_dir: Optional[str | os.PathLike] = None,
) -> Dict[str, float]:
    df = read_table(input_path)
    validate_table(df, require_matched=True)
    q = q_function.predict_dataframe(df)
    matched = df["matched"].astype(float).values
    states = df["state"].astype(str).values
    tp_corr = 0.0
    true_corr = 0.0
    tp_naive = float(matched.sum())
    true_naive = float(len(df))
    for i, state in enumerate(states):
        r1 = 1 if state in {"10", "11"} else 0
        r2 = 1 if state in {"01", "11"} else 0
        r12 = 1 if state == "11" else 0
        q1, q2, q12 = [float(v) for v in q[i]]
        pi_hat = min(MAX_ABS_WEIGHT, max(EPS, q12 / max(EPS, q1 * q2)))
        w = (r1 / q1) + (r2 / q2) - (r12 / q12)
        w = max(-MAX_ABS_WEIGHT, min(MAX_ABS_WEIGHT, w / pi_hat))
        tp_corr += matched[i] * w
        true_corr += w
    out = {
        "n_visible_rows": int(len(df)),
        "n_docs_with_visible_rows": int(df["doc_id"].nunique()) if "doc_id" in df.columns else None,
        "pred_total": float(pred_total),
        "corrected_tp_total": float(tp_corr),
        "corrected_true_terms_total": float(true_corr),
        "naive_tp_total": float(tp_naive),
        "naive_true_terms_total": float(true_naive),
        "corrected_precision": float(tp_corr / pred_total) if pred_total > 0 else 0.0,
        "corrected_recall": float(tp_corr / true_corr) if true_corr > 0 else 0.0,
        "naive_precision": float(tp_naive / pred_total) if pred_total > 0 else 0.0,
        "naive_recall": float(tp_naive / true_naive) if true_naive > 0 else 0.0,
    }
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        with open(Path(out_dir) / f"estimate_{q_function.config.model_name}.json", "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
    return out


def evaluate_two_lists_with_model(
    list1_df: pd.DataFrame,
    list2_df: pd.DataFrame,
    model_df: pd.DataFrame,
    output_dir: str | os.PathLike = "evaluation_outputs",
    method: MatchMethod = "character",
    ai_matcher: Optional[AiMatcher] = None,
    require_type_match: bool = True,
    pred_total: Optional[float] = None,
    q_function: object | None = None,
    train_config: object | None = None,
) -> Dict[str, object]:
    """
    Build a CRC evaluation table from two human lists plus one model-prediction table,
    then estimate naive and CRC-corrected precision/recall.

    The model predictions are used only to create matched for evaluation. The
    q-function is trained from the two-list reconstructed state table and does
    not use matched. If q_function is not provided, this function trains one
    from that matched-free visible state table. pred_total defaults to the
    number of model predictions.
    """
    out_dir = Path(output_dir)
    data_dir = out_dir / "data"
    model_dir = out_dir / "models" / "q_function"
    estimate_dir = out_dir / "estimate"
    data_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    estimate_dir.mkdir(parents=True, exist_ok=True)

    eval_df, matches = make_evaluation_table(
        list1_df,
        list2_df,
        model_df,
        method=method,
        ai_matcher=ai_matcher,
        require_type_match=require_type_match,
    )
    q_train_df = eval_df.drop(columns=["matched"]).copy()
    pred_total = float(len(model_df)) if pred_total is None else float(pred_total)

    q_train_path = data_dir / "q_training_visible_terms.csv"
    eval_path = data_dir / "evaluation_visible_terms.csv"
    match_path = data_dir / "model_human_matches.csv"
    write_table(q_train_df, q_train_path)
    write_table(eval_df, eval_path)
    write_table(matches, match_path)

    if q_function is None:
        if train_config is None:
            train_config = TrainConfig(
                model_name="two_list_pubmedbert",
                model_path=PUBMEDBERT_PATH,
                hidden_dim=64,
                dropout=0.35,
                lr=1e-3,
                weight_decay=5e-4,
                epochs=40,
                patience=8,
                batch_size_embed=64,
                batch_size_head=64,
                max_length=96,
                val_frac=0.2,
                seed=2026,
                device_name="cpu",
                use_context=True,
            )
        q_function, train_summary = train_q_from_table(q_train_path, model_dir, train_config)
    else:
        train_summary = {"source": "provided q_function"}

    estimate = estimate_precision_recall(eval_path, q_function, pred_total, estimate_dir)
    summary: Dict[str, object] = {
        "n_list1_rows": int(len(list1_df)),
        "n_list2_rows": int(len(list2_df)),
        "n_model_rows": int(len(model_df)),
        "n_visible_rows": int(len(eval_df)),
        "n_matches": int(len(matches)),
        "state_counts": {k: int(v) for k, v in eval_df["state"].value_counts().to_dict().items()},
        "pred_total": pred_total,
        "matching_method": method,
        "estimate": estimate,
        "paths": {
            "q_training_table": relative_to_cwd(q_train_path),
            "evaluation_table": relative_to_cwd(eval_path),
            "matches": relative_to_cwd(match_path),
            "q_function": relative_to_cwd(model_dir / "q_function.pt"),
            "estimate_dir": relative_to_cwd(estimate_dir),
        },
        "train_summary": train_summary,
    }
    with open(out_dir / "evaluation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary
