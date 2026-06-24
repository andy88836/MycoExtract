"""v8: unit-level + specific-activity validators.

The H122A/L123A/Q202L mutation series in the human review showed multiple
records with kcat/Km values stuffed into Km_value. The 100%-degradation_efficiency
records were specific activity (220 U/mg, 450 U/mg) miscoded.
"""
from src.utils.row_level_validator import RowLevelValidator


def test_kcat_km_mistyped_as_km_flagged():
    """Km_value with rate-constant unit (M^-1 s^-1) is actually kcat/Km."""
    record = {
        "measurement_type": "kinetic",
        "Km_value": 2.24,
        "Km_unit": "M^-1 s^-1",   # actually a kcat/Km unit
    }
    validated = RowLevelValidator.validate_record(dict(record))
    assert "km_unit_looks_like_kcat_km" in (validated.get("error_flags") or [])


def test_km_with_concentration_unit_passes():
    record = {
        "measurement_type": "kinetic",
        "Km_value": 0.5,
        "Km_unit": "µM",
    }
    validated = RowLevelValidator.validate_record(dict(record))
    flags = validated.get("error_flags") or []
    assert "km_unit_looks_like_kcat_km" not in flags


def test_kcat_unit_unexpected_flagged():
    record = {
        "measurement_type": "kinetic",
        "kcat_value": 100,
        "kcat_unit": "%",  # not a rate unit
    }
    validated = RowLevelValidator.validate_record(dict(record))
    assert "kcat_unit_unexpected" in (validated.get("error_flags") or [])


def test_kcat_unit_per_second_passes():
    record = {
        "measurement_type": "kinetic",
        "kcat_value": 100,
        "kcat_unit": "s^-1",
    }
    validated = RowLevelValidator.validate_record(dict(record))
    flags = validated.get("error_flags") or []
    assert "kcat_unit_unexpected" not in flags


def test_kcat_unicode_superscript_unit_passes():
    record = {
        "measurement_type": "kinetic",
        "kcat_value": 27.1,
        "kcat_unit": "min⁻¹",
    }
    validated = RowLevelValidator.validate_record(dict(record))
    flags = validated.get("error_flags") or []
    assert "kcat_unit_unexpected" not in flags


def test_degradation_efficiency_out_of_range_flagged():
    record = {
        "measurement_type": "degradation",
        "degradation_efficiency": 250,
    }
    validated = RowLevelValidator.validate_record(dict(record))
    assert "degradation_efficiency_out_of_range" in (validated.get("error_flags") or [])


def test_specific_activity_miscoded_flagged():
    """The H122A/L123A series in human review: degradation_efficiency=100 when
    the real value was specific activity = 220 U/mg."""
    record = {
        "measurement_type": "degradation",
        "degradation_efficiency": 100,
        "notes": "specific activity 220 U/mg observed for variant H122A",
    }
    validated = RowLevelValidator.validate_record(dict(record))
    flags = validated.get("error_flags") or []
    assert "specific_activity_miscoded_as_degradation_efficiency" in flags


def test_specific_activity_with_evidence_text_flagged():
    record = {
        "measurement_type": "degradation",
        "degradation_efficiency": 100,
        "evidence_text": "The mutant showed 2-fold increase in specific activity (450 U/mg).",
    }
    validated = RowLevelValidator.validate_record(dict(record))
    flags = validated.get("error_flags") or []
    assert "specific_activity_miscoded_as_degradation_efficiency" in flags
