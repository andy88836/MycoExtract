"""Row-level validation for measurement-context extraction records."""

import re
from typing import Any, Dict, List

from src.pipeline.post_processor import (
    apply_context_condition_mapping,
    infer_measurement_type,
    _has_value,
)
from src.utils.table_multiplier import parse_table_header_multiplier


class RowLevelValidator:
    """Adds error_flags and human_review_required without deleting rows."""

    AMBIGUOUS_ENZYME_ABBREVIATIONS = {"po", "pod", "rpod", "ppl", "ala"}
    BLOCKED_ENRICHMENT_STATES = {
        "commercial", "crude", "cell_free", "partially_purified",
        "immobilized composite"
    }

    @classmethod
    def validate_batch(cls, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [cls.validate_record(r) for r in records if isinstance(r, dict)]

    @classmethod
    def validate_record(cls, record: Dict[str, Any]) -> Dict[str, Any]:
        record = apply_context_condition_mapping(record)
        flags = list(record.get("error_flags") or [])

        measurement_type = infer_measurement_type(record)
        source_type = (
            record.get("source_type")
            or record.get("source_in_document", {}).get("source_type")
            or ""
        )

        has_kinetic = any(_has_value(record.get(f)) for f in ["Km_value", "kcat_value", "kcat_Km_value"])
        has_degradation = _has_value(record.get("degradation_efficiency"))

        if measurement_type == "kinetic" and has_degradation:
            cls._add(flags, "kinetic_degradation_condition_conflict")
            cls._add(flags, "metric_condition_mismatch")
        if measurement_type in {"degradation", "biotransformation", "application_matrix_degradation"} and has_kinetic:
            cls._add(flags, "kinetic_degradation_condition_conflict")
            cls._add(flags, "metric_condition_mismatch")

        if measurement_type == "kinetic":
            if _has_value(record.get("degradation_time_value")):
                cls._add(flags, "activity_assay_condition_misattribution")
            if _has_value(record.get("temperature_value")) and not _has_value(record.get("kinetic_temperature_value")):
                cls._add(flags, "metric_condition_mismatch")
            if _has_value(record.get("ph")) and not _has_value(record.get("kinetic_ph")):
                cls._add(flags, "metric_condition_mismatch")

        if measurement_type in {"degradation", "biotransformation", "application_matrix_degradation"}:
            if _has_value(record.get("temperature_value")) and not _has_value(record.get("degradation_temperature_value")):
                cls._add(flags, "metric_condition_mismatch")
            if _has_value(record.get("ph")) and not _has_value(record.get("degradation_ph")):
                cls._add(flags, "metric_condition_mismatch")
            if _has_value(record.get("reaction_time_value")) and not _has_value(record.get("degradation_time_value")):
                cls._add(flags, "metric_condition_mismatch")

        if measurement_type == "enzyme_activity_assay":
            if has_kinetic or has_degradation:
                cls._add(flags, "activity_assay_condition_misattribution")

        if measurement_type != "optimum_condition":
            optimum_text = " ".join(str(record.get(f) or "") for f in ["notes", "optimum_condition_target"]).lower()
            if (
                (_has_value(record.get("optimal_ph")) or _has_value(record.get("optimal_temperature_value"))
                 or _has_value(record.get("optimum_ph")) or _has_value(record.get("optimum_temperature_value")))
                and "optimum" not in optimum_text
                and "optimal" not in optimum_text
            ):
                cls._add(flags, "optimum_condition_over_assignment")

        if str(source_type).lower() == "review" or measurement_type == "review_background":
            cls._add(flags, "review_article_leakage")
            record["QC_Status"] = "exclude_review_article"
            record["Extraction_Allowed"] = False

        cls._validate_enrichment(record, flags)
        cls._validate_table_multiplier(record, flags)
        cls._validate_ocr_risk(record, flags)
        cls._validate_units_v8(record, flags)
        cls._validate_specific_activity_v8(record, flags)
        cls._validate_stability_v8(record, flags)
        cls._validate_v9_semantics(record, flags)

        if flags:
            record["error_flags"] = flags
            record["human_review_required"] = True
        else:
            record["error_flags"] = record.get("error_flags") or []

        return record

    # ---------- v8 validators ----------

    V8_STABILITY_METRICS = {
        "Tm", "T50", "half_life", "residual_activity",
        "storage_retention", "reuse_retention", "pH_residual_activity",
    }

    @classmethod
    def _validate_units_v8(cls, record: Dict[str, Any], flags: List[str]) -> None:
        """Catch the most common v7 errors:
        - kcat/Km value placed in Km_value (the H122A/L123A series)
        - Km units that look like rate constants
        - degradation_efficiency outside [0, 105]
        """
        km_unit = cls._normalize_unit(record.get("Km_unit"))
        if record.get("Km_value") and km_unit:
            if any(t in km_unit for t in ["m^-1", "m-1", "/m s", "/m/s", "s-1", "s^-1"]):
                cls._add(flags, "km_unit_looks_like_kcat_km")

        kcat_unit = cls._normalize_unit(record.get("kcat_unit"))
        if record.get("kcat_value") and kcat_unit:
            if not any(t in kcat_unit for t in ["s-1", "s^-1", "/s", "min-1", "min^-1", "/min", "h-1", "h^-1", "/h"]):
                cls._add(flags, "kcat_unit_unexpected")

        de = record.get("degradation_efficiency")
        if de not in (None, "", "null"):
            try:
                v = float(de)
                if v < 0 or v > 105:
                    cls._add(flags, "degradation_efficiency_out_of_range")
            except (ValueError, TypeError):
                cls._add(flags, "degradation_efficiency_not_numeric")

    @classmethod
    def _validate_specific_activity_v8(cls, record: Dict[str, Any], flags: List[str]) -> None:
        """Detect specific activity (U/mg, U/mL) being miscoded as degradation_efficiency.
        v7 had 4 such cases in the human review (degradation_efficiency = 100 when the
        real value was specific activity = 220 U/mg, 450 U/mg, etc.)."""
        text = " ".join(str(record.get(f) or "") for f in [
            "notes", "evidence_text", "Km_unit", "kcat_unit",
        ]).lower()
        sa_signals = ["specific activity", "u/mg", "u/ml", "iu/mg", "iu/ml", "units/mg"]
        if any(s in text for s in sa_signals):
            if record.get("degradation_efficiency") in (100, "100", 100.0):
                cls._add(flags, "specific_activity_miscoded_as_degradation_efficiency")
            elif record.get("degradation_efficiency"):
                cls._add(flags, "possible_specific_activity_in_degradation_field")

    @classmethod
    def _validate_stability_v8(cls, record: Dict[str, Any], flags: List[str]) -> None:
        """Stability rows must declare a metric type from the v8 enum."""
        if str(record.get("measurement_type") or "").lower() == "stability":
            smt = record.get("stability_metric_type")
            if smt and smt not in cls.V8_STABILITY_METRICS:
                cls._add(flags, "invalid_stability_metric_type")
            if not record.get("stability_value") and not record.get("stability_metric_type"):
                cls._add(flags, "stability_row_missing_value_and_type")

    @classmethod
    def _validate_v9_semantics(cls, record: Dict[str, Any], flags: List[str]) -> None:
        text = " ".join(str(record.get(f) or "") for f in [
            "notes", "evidence_text", "source_section",
        ]).lower()

        if record.get("degradation_efficiency") and not record.get("degradation_time_value"):
            cls._add(flags, "missing_time_for_degradation")

        if record.get("Km_value") and any(term in text for term in [
            "hill equation", "hill coefficient", " nh", "s50", "k'",
            "sigmoidal kinetics", "positive cooperativity", "composite constant k",
        ]):
            cls._add(flags, "hill_constant_not_michaelis_menten_km")

        de = record.get("degradation_efficiency")
        if de not in (None, "", "null"):
            try:
                is_hundred = abs(float(de) - 100.0) <= 1.0
            except (TypeError, ValueError):
                is_hundred = False
            if is_hundred and any(term in text for term in [
                "relative activity", "residual activity", "remaining activity",
                "normalized activity", "set as 100%", "control was set as 100%",
                "untreated control", "percent of control", "reference for other substrates",
            ]):
                cls._add(flags, "wrong_metric_type_relative_activity_baseline")
            if any(term in text for term in [
                "residual bioluminescence", "ecotoxicity", "cytotoxicity",
                "cell viability", "ldh", "ros", "dna damage", "inhibition rate",
                "tissue residue", "animal performance",
            ]):
                cls._add(flags, "wrong_metric_type_toxicity_endpoint")

        if record.get("substrate_concentration_value") and not record.get("substrate_concentration_unit"):
            cls._add(flags, "unit_context_ambiguity")

    @classmethod
    def _validate_enrichment(cls, record: Dict[str, Any], flags: List[str]) -> None:
        enzyme_state = str(record.get("enzyme_state") or "").lower()
        enzyme_name = str(record.get("enzyme_name") or record.get("reported_enzyme_name") or "")
        normalized_name = re.sub(r"[^A-Za-z0-9]+", " ", enzyme_name).strip().lower()
        tokens = normalized_name.split()
        ambiguous_name = normalized_name in cls.AMBIGUOUS_ENZYME_ABBREVIATIONS or any(
            t in cls.AMBIGUOUS_ENZYME_ABBREVIATIONS for t in tokens
        )

        has_enrichment = any(_has_value(record.get(f)) for f in ["uniprot_id", "gene_name", "pdb_id"])
        if enzyme_state in cls.BLOCKED_ENRICHMENT_STATES and has_enrichment:
            cls._add(flags, "enrichment_cascade_error")
            cls._add(flags, "unidentified_enzyme_over_resolution")
        if ambiguous_name and (_has_value(record.get("enzyme_full_name")) or has_enrichment):
            cls._add(flags, "enzyme_name_over_normalization")
            cls._add(flags, "enrichment_cascade_error")

        organism = str(record.get("organism") or "")
        if re.match(r"^[A-Za-z]\.\s+\S+", organism):
            cls._add(flags, "unit_context_ambiguity")

    @classmethod
    def _validate_table_multiplier(cls, record: Dict[str, Any], flags: List[str]) -> None:
        source = record.get("kinetic_unit_source_text") or record.get("kcat_Km_unit")
        multiplier, _, ambiguous = parse_table_header_multiplier(source)
        if ambiguous:
            cls._add(flags, "table_multiplier_scaling_error")
            return
        if multiplier and multiplier != 1 and not _has_value(record.get("kinetic_unit_multiplier")):
            cls._add(flags, "table_multiplier_scaling_error")

    @classmethod
    def _validate_ocr_risk(cls, record: Dict[str, Any], flags: List[str]) -> None:
        notes = str(record.get("notes") or "").lower()
        source_type = str(record.get("source_in_document", {}).get("source_type") or "").lower()
        risky_terms = ["ocr", "footnote", "small figure label", "image-only", "figure label"]
        if source_type == "figure" or any(term in notes for term in risky_terms):
            cls._add(flags, "ocr_footnote_recognition_error")

    @staticmethod
    def _add(flags: List[str], flag: str) -> None:
        if flag not in flags:
            flags.append(flag)

    @staticmethod
    def _normalize_unit(unit: Any) -> str:
        """Normalize common Unicode unit spellings before lightweight checks."""
        text = str(unit or "").strip().lower()
        if not text:
            return ""
        replacements = {
            "⁻": "-",
            "−": "-",
            "–": "-",
            "¹": "1",
            "²": "2",
            "³": "3",
            "µ": "u",
            "μ": "u",
            "·": "",
            " ": "",
        }
        for src, dst in replacements.items():
            text = text.replace(src, dst)
        # Convert compact exponent forms such as s-1/min-1 into the same
        # representation accepted by the existing validator.
        text = text.replace("^-", "-")
        return text
