"""v8: substrate whitelist must reject non-mycotoxin substrates.

These were the 21+ substrates that leaked into primary_eligible in the v7
baseline. v8 must reject all of them via QualityConstraintFilter.
"""
import pytest

from src.utils.quality_constraints import QualityConstraintFilter


@pytest.fixture
def qc():
    return QualityConstraintFilter(
        require_sequence=False,
        require_mycotoxin=True,
        check_detoxification=False,
        strict_mode=True,
    )


# Real non-mycotoxin substrates from the human review FP set.
NON_MYCOTOXIN_SUBSTRATES = [
    "Benzo[a]pyrene", "ABTS", "7-pentoxyresorufin", "7-ethoxyresorufin",
    "7-Methoxyresorufin", "Coumarin", "Nifedipine",
    "1-aminocyclopropane-1-carboxylic acid", "D-cysteine",
    "HMF", "Pentoxyresorufin", "guaiacol", "hydrogen peroxide",
    "Aniline", "Styrene oxide", "Dibromomethane",
    "dichloromethane", "pentachlorophenol", "Penicillic acid",
    "monodehydroascorbate", "dehydroascorbate",
    "Synbiotic A", "BioPlus 2B", "Cylactin", "inulin",
    "turkey tissue", "intestinal content",
]


@pytest.mark.parametrize("substrate", NON_MYCOTOXIN_SUBSTRATES)
def test_non_mycotoxin_substrate_rejected(qc, substrate):
    record = {"substrate": substrate}
    ok, reason = qc._check_mycotoxin_substrate(record)
    assert not ok, f"{substrate!r} should be rejected but passed: {reason}"


# Real mycotoxin substrates that must pass.
MYCOTOXIN_SUBSTRATES = [
    "AFB1", "Aflatoxin B1", "Aflatoxin M1", "AFG1",
    "OTA", "Ochratoxin A", "Ochratoxin α",
    "DON", "Deoxynivalenol", "NIV", "T-2", "HT-2",
    "FB1", "Fumonisin B1", "ZEN", "Zearalenone",
    "Patulin", "Citrinin", "Sterigmatocystin",
    "Beauvericin", "Tenuazonic acid",
]


@pytest.mark.parametrize("substrate", MYCOTOXIN_SUBSTRATES)
def test_mycotoxin_substrate_accepted(qc, substrate):
    record = {"substrate": substrate}
    ok, reason = qc._check_mycotoxin_substrate(record)
    assert ok, f"{substrate!r} should be accepted but rejected: {reason}"


def test_empty_substrate_rejected(qc):
    ok, reason = qc._check_mycotoxin_substrate({"substrate": ""})
    assert not ok
    assert "substrate" in reason.lower()
