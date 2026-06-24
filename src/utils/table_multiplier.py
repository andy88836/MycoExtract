"""
Utilities for parsing table header multipliers such as x10^3 and 10³.

The parser is intentionally conservative: it returns a multiplier only when a
single unambiguous header multiplier is present.
"""

import re
from typing import Any, Dict, Optional, Tuple


SUPERSCRIPT_DIGITS = str.maketrans({
    "⁰": "0",
    "¹": "1",
    "²": "2",
    "³": "3",
    "⁴": "4",
    "⁵": "5",
    "⁶": "6",
    "⁷": "7",
    "⁸": "8",
    "⁹": "9",
    "⁻": "-",
})


def parse_table_header_multiplier(source_text: Optional[str]) -> Tuple[Optional[float], Optional[str], bool]:
    """
    Parse one unambiguous power-of-ten multiplier from header/source text.

    Returns:
        (multiplier, matched_text, ambiguous)
    """
    if not source_text:
        return None, None, False

    text = str(source_text)
    normalized = text.translate(SUPERSCRIPT_DIGITS)

    patterns = [
        ("explicit", r"(?:x|×|\*)\s*10\s*(?:\^)?\s*(-?\d+)"),
        ("caret", r"10\s*\^\s*(-?\d+)"),
        ("spaced", r"10\s+([+-]?\d{1,2})(?![\d.])"),
        ("signed", r"10\s*([+-])\s*(\d{1,2})(?![\d.])"),
        ("compact_unit", r"10([+-]?\d)(?=\s*[A-Za-zµμ])"),
        # Handles true superscript forms such as 10³.  Do not use the
        # translated text for plain "1031" because that is an ordinary value,
        # not a header multiplier.
        ("superscript", r"10\s*([⁰¹²³⁴⁵⁶⁷⁸⁹⁻]+)"),
    ]

    matches = []
    for kind, pattern in patterns:
        search_text = text if kind == "superscript" else normalized
        for match in re.finditer(pattern, search_text, flags=re.IGNORECASE):
            if kind == "signed":
                exponent_text = f"{match.group(1)}{match.group(2)}"
            else:
                exponent_text = match.group(1).translate(SUPERSCRIPT_DIGITS)
            exponent = int(exponent_text)
            raw = search_text[match.start():match.end()]
            matches.append((10 ** exponent, raw))

    # Deduplicate overlapping/equivalent matches.
    unique = []
    for multiplier, raw in matches:
        if not any(multiplier == m and raw == r for m, r in unique):
            unique.append((multiplier, raw))

    if not unique:
        return None, None, False

    distinct_multipliers = {m for m, _ in unique}
    if len(distinct_multipliers) > 1:
        return None, "; ".join(raw for _, raw in unique), True

    multiplier, raw = unique[0]
    return float(multiplier), raw, False


def apply_kinetic_unit_multiplier(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply a parsed table multiplier to kcat/Km-style values when source text
    clearly provides a header multiplier.
    """
    # Only parse multiplier from dedicated unit fields.  Do NOT include
    # notes/evidence_text/source_section — they may contain raw table text
    # (e.g. "1.75×10^4") that would trigger a false re-scale on records
    # whose values are already correct.
    source_text = " ".join(
        str(part)
        for part in [
            record.get("kinetic_unit_source_text"),
            record.get("kcat_Km_unit"),
        ]
        if part not in (None, "")
    )
    multiplier, matched_text, ambiguous = parse_table_header_multiplier(source_text)

    unit_multiplier, unit_matched_text, unit_ambiguous = parse_table_header_multiplier(record.get("kcat_Km_unit"))
    if not multiplier and unit_multiplier:
        multiplier = unit_multiplier
        matched_text = unit_matched_text
    if unit_ambiguous:
        ambiguous = True

    if record.get("_table_multiplier_applied"):
        # Multiplier was already applied upstream. Never re-scale — just
        # normalize the unit string to remove embedded power-of-ten markers.
        record["kcat_Km_unit"] = _normalized_kcat_km_unit(record.get("kcat_Km_unit"))
        return record

    # Upstream extractors/aggregation may already preserve both the normalized
    # value and the multiplier metadata. Treat this as already applied only if
    # the unit itself no longer contains a multiplier marker.
    if record.get("kinetic_unit_multiplier") not in (None, "") and not unit_multiplier:
        record["_table_multiplier_applied"] = True
        return record

    if ambiguous:
        record["human_review_required"] = True
        flags = record.get("error_flags") or []
        if "table_multiplier_scaling_error" not in flags:
            flags.append("table_multiplier_scaling_error")
        record["error_flags"] = flags
        if matched_text and not record.get("kinetic_unit_source_text"):
            record["kinetic_unit_source_text"] = matched_text
        return record

    if not multiplier or multiplier == 1:
        return record

    value = record.get("kcat_Km_value")
    if value in (None, ""):
        return record

    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        record["human_review_required"] = True
        flags = record.get("error_flags") or []
        if "table_multiplier_scaling_error" not in flags:
            flags.append("table_multiplier_scaling_error")
        record["error_flags"] = flags
        return record

    record["kcat_Km_value"] = numeric_value * multiplier
    record["kinetic_unit_multiplier"] = multiplier
    record["kinetic_unit_source_text"] = matched_text or str(source_text)
    record["kcat_Km_unit"] = _normalized_kcat_km_unit(record.get("kcat_Km_unit"))
    record["_table_multiplier_applied"] = True
    return record


def _normalized_kcat_km_unit(unit: Any) -> Any:
    """Drop an embedded power-of-ten multiplier from kcat/Km units."""
    if not unit:
        return unit
    text = str(unit)
    multiplier, _, ambiguous = parse_table_header_multiplier(text)
    if not multiplier or ambiguous:
        return unit
    lower = text.lower()
    if "min" in lower:
        return "M⁻¹ min⁻¹" if "mm" not in lower else "mM⁻¹ min⁻¹"
    if "s" in lower or "sec" in lower:
        return "M⁻¹ s⁻¹" if "mm" not in lower else "mM⁻¹ s⁻¹"
    return re.sub(r"(?:x|×|\*)?\s*10\s*(?:\^)?\s*[-+]?\d+\s*", "", text.translate(SUPERSCRIPT_DIGITS)).strip()
