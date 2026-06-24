"""v8: stability schema validators."""
import pytest

from src.utils.row_level_validator import RowLevelValidator


VALID_STABILITY_RECORDS = [
    {
        "measurement_type": "stability",
        "stability_metric_type": "Tm",
        "stability_value": 71.13,
        "stability_unit": "°C",
    },
    {
        "measurement_type": "stability",
        "stability_metric_type": "T50",
        "stability_value": 65.0,
        "stability_unit": "°C",
    },
    {
        "measurement_type": "stability",
        "stability_metric_type": "residual_activity",
        "stability_value": 40,
        "stability_unit": "%",
        "stability_pretreatment_temperature_value": 40,
        "stability_pretreatment_time_value": 3,
        "stability_pretreatment_time_unit": "h",
    },
    {
        "measurement_type": "stability",
        "stability_metric_type": "reuse_retention",
        "stability_value": 80.7,
        "stability_unit": "%",
        "stability_cycles": 4,
    },
    {
        "measurement_type": "stability",
        "stability_metric_type": "storage_retention",
        "stability_value": 70.9,
        "stability_unit": "%",
        "stability_pretreatment_temperature_value": 4,
        "stability_pretreatment_time_value": 30,
        "stability_pretreatment_time_unit": "days",
    },
    {
        "measurement_type": "stability",
        "stability_metric_type": "half_life",
        "stability_value": 200,
        "stability_unit": "%_relative",
        "stability_pretreatment_temperature_value": 50,
    },
    {
        "measurement_type": "stability",
        "stability_metric_type": "pH_residual_activity",
        "stability_value": 80,
        "stability_unit": "%",
        "stability_pretreatment_ph": 4.0,
        "stability_pretreatment_time_value": 1,
        "stability_pretreatment_time_unit": "h",
    },
]


@pytest.mark.parametrize("record", VALID_STABILITY_RECORDS)
def test_valid_stability_metrics_accepted(record):
    validated = RowLevelValidator.validate_record(dict(record))
    flags = validated.get("error_flags") or []
    assert "invalid_stability_metric_type" not in flags, \
        f"{record['stability_metric_type']} should be valid: flags={flags}"


def test_invalid_stability_metric_type_flagged():
    record = {
        "measurement_type": "stability",
        "stability_metric_type": "thermal_stability",  # deprecated/invalid
        "stability_value": 80,
        "stability_unit": "%",
    }
    validated = RowLevelValidator.validate_record(dict(record))
    flags = validated.get("error_flags") or []
    assert "invalid_stability_metric_type" in flags


def test_stability_row_with_no_value_or_type_flagged():
    record = {
        "measurement_type": "stability",
        # both stability_value and stability_metric_type missing
    }
    validated = RowLevelValidator.validate_record(dict(record))
    flags = validated.get("error_flags") or []
    assert "stability_row_missing_value_and_type" in flags
