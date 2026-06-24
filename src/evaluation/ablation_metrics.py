from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd


FIELD_SPECS = {
    "Enzyme": {"kind": "str", "record_key": "enzyme_name"},
    "Substrate": {"kind": "str", "record_key": "substrate"},
    "Organism": {"kind": "str", "record_key": "organism"},
    "Km (μM)": {"kind": "num", "record_key": "Km_value"},
    "kcat (s⁻¹)": {"kind": "num", "record_key": "kcat_value"},
    "kcat/Km (M⁻¹s⁻¹)": {"kind": "num", "record_key": "kcat_Km_value"},
    "Degrad. %": {"kind": "num", "record_key": "degradation_efficiency"},
}

GOLD_DOI_COLUMN = "Source DOI"


def normalize_str(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = str(value).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def to_float(value: Any) -> float:
    if value is None or value == "":
        return math.nan
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace(",", "")
    try:
        return float(text)
    except ValueError:
        return math.nan


def has_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    return str(value).strip() != ""


def numeric_match(pred: Any, gold: Any, rel_tol: float = 0.05, abs_tol: float = 1e-12) -> bool:
    p = to_float(pred)
    g = to_float(gold)
    if math.isnan(p) and math.isnan(g):
        return True
    if math.isnan(p) or math.isnan(g):
        return False
    if abs(g) < abs_tol:
        return abs(p - g) <= abs_tol
    return abs(p - g) / abs(g) <= rel_tol


def value_match(pred: Any, gold: Any, kind: str) -> bool:
    if kind == "num":
        return numeric_match(pred, gold)
    return normalize_str(pred) == normalize_str(gold)


def prf1(tp: int, fp: int, fn: int) -> Dict[str, Any]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def load_gold_records(path: str) -> List[Dict[str, Any]]:
    gold = pd.read_excel(path, sheet_name=0)
    gold.columns = [str(c).strip() for c in gold.columns]
    return gold.to_dict(orient="records")


def group_gold_by_doi(gold_records: Iterable[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in gold_records:
        doi = normalize_doi(record.get(GOLD_DOI_COLUMN))
        if doi:
            grouped[doi].append(record)
    return grouped


def normalize_doi(value: Any) -> str:
    text = normalize_str(value)
    text = text.replace("https://doi.org/", "").replace("http://doi.org/", "")
    text = text.replace("doi:", "").strip()
    return text


def score_record_alignment(pred: Dict[str, Any], gold: Dict[str, Any]) -> float:
    score = 0.0
    if value_match(pred.get("enzyme_name"), gold.get("Enzyme"), "str"):
        score += 3.0
    if value_match(pred.get("substrate"), gold.get("Substrate"), "str"):
        score += 3.0
    if value_match(pred.get("organism"), gold.get("Organism"), "str"):
        score += 1.0
    for gold_field in ("Km (μM)", "kcat (s⁻¹)", "kcat/Km (M⁻¹s⁻¹)", "Degrad. %"):
        spec = FIELD_SPECS[gold_field]
        if has_value(gold.get(gold_field)) and numeric_match(pred.get(spec["record_key"]), gold.get(gold_field)):
            score += 1.0
    return score


def align_predictions_to_gold(
    predictions: List[Dict[str, Any]],
    gold_records: List[Dict[str, Any]],
) -> List[Tuple[Optional[Dict[str, Any]], Dict[str, Any]]]:
    """Greedy one-to-one alignment within one paper/DOI."""
    remaining = list(predictions)
    aligned: List[Tuple[Optional[Dict[str, Any]], Dict[str, Any]]] = []

    for gold in gold_records:
        best_idx = None
        best_score = 0.0
        for idx, pred in enumerate(remaining):
            score = score_record_alignment(pred, gold)
            if score > best_score:
                best_score = score
                best_idx = idx
        if best_idx is None or best_score < 3.0:
            aligned.append((None, gold))
        else:
            aligned.append((remaining.pop(best_idx), gold))
    return aligned


def evaluate_predictions_by_doi(
    predictions_by_doi: Dict[str, List[Dict[str, Any]]],
    gold_by_doi: Dict[str, List[Dict[str, Any]]],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    rows = []
    micro = {"tp": 0, "fp": 0, "fn": 0}

    for field, spec in FIELD_SPECS.items():
        tp = fp = fn = 0
        for doi, gold_records in gold_by_doi.items():
            predictions = predictions_by_doi.get(doi, [])
            aligned = align_predictions_to_gold(predictions, gold_records)
            matched_pred_ids = {id(pred) for pred, _ in aligned if pred is not None}

            for pred, gold in aligned:
                gold_has = has_value(gold.get(field))
                pred_value = pred.get(spec["record_key"]) if pred else None
                pred_has = has_value(pred_value)
                correct = pred_has and gold_has and value_match(pred_value, gold.get(field), spec["kind"])
                tp += 1 if correct else 0
                fp += 1 if pred_has and not correct else 0
                fn += 1 if gold_has and not correct else 0

            # Predictions not aligned to a gold row count as FP if they have this field.
            for pred in predictions:
                if id(pred) not in matched_pred_ids and has_value(pred.get(spec["record_key"])):
                    fp += 1

        metric = prf1(tp, fp, fn)
        micro["tp"] += tp
        micro["fp"] += fp
        micro["fn"] += fn
        rows.append({"field": field, **metric})

    return pd.DataFrame(rows), prf1(micro["tp"], micro["fp"], micro["fn"])


def records_by_doi_from_ablation_payload(payload: Dict[str, Any], config_name: str) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for paper in payload.get("papers", []):
        doi = normalize_doi(paper.get("doi"))
        if not doi:
            continue
        records = paper.get("configs", {}).get(config_name, {}).get("records", [])
        grouped[doi].extend(records)
    return grouped
