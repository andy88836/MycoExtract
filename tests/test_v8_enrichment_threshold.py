"""v8: enrichment auto-fill threshold + scoring rebalance.

v7 had threshold=0.9 but the score function topped out at 0.6 without gene_name,
so 75% of records (those without gene_name) could NEVER auto-fill.
v8 lowers threshold to 0.7 and rebalances weights so reviewed Swiss-Prot +
exact species + protein name match alone can reach 0.8.
"""
from src.utils.sequence_enricher import SequenceEnricher, SequenceCandidate


def test_default_threshold_is_07():
    e = SequenceEnricher(fetch_sequences=False, fetch_smiles=False)
    assert e.auto_fill_threshold == 0.7, (
        "v8 default auto_fill_threshold must be 0.7. v7 was 0.9 — unreachable "
        "without gene_name (75% of records have no gene_name)."
    )


def test_score_reachable_without_gene_name():
    """A reviewed Swiss-Prot entry with exact species + protein name match
    should score ≥0.7 even if the query has no gene_name."""
    e = SequenceEnricher(fetch_sequences=False, fetch_smiles=False)
    score = e._calculate_match_score(
        query_enzyme="laccase",
        query_organism="Bacillus subtilis",
        query_gene=None,
        result_protein="laccase",
        result_organism="Bacillus subtilis",
        result_gene="cotA",
        reviewed=True,
    )
    assert score >= 0.7, f"reviewed + exact species + name match should ≥0.7, got {score}"


def test_score_caps_at_one():
    e = SequenceEnricher(fetch_sequences=False, fetch_smiles=False)
    score = e._calculate_match_score(
        query_enzyme="ZenA",
        query_organism="Rhodococcus erythropolis",
        query_gene="zenA",
        result_protein="ZenA lactonase",
        result_organism="Rhodococcus erythropolis",
        result_gene="zenA",
        reviewed=True,
    )
    assert 0.9 <= score <= 1.0


def test_score_low_for_genus_only_match():
    """genus-only organism match should not pass the 0.7 threshold without
    additional signals."""
    e = SequenceEnricher(fetch_sequences=False, fetch_smiles=False)
    score = e._calculate_match_score(
        query_enzyme="unknown_enzyme",
        query_organism="Bacillus subtilis",
        query_gene=None,
        result_protein="hypothetical protein",
        result_organism="Bacillus cereus",  # same genus only
        result_gene=None,
        reviewed=False,
    )
    assert score < 0.7, f"weak match should fall below threshold, got {score}"


def test_organism_whitelist_expansion():
    """v8: SAFE_ORGANISM_EXPANSIONS dict must be activated."""
    from src.utils.sequence_enricher import SAFE_ORGANISM_EXPANSIONS
    assert "b. subtilis" in SAFE_ORGANISM_EXPANSIONS
    assert SAFE_ORGANISM_EXPANSIONS["b. subtilis"] == "Bacillus subtilis"
    assert "e. coli" in SAFE_ORGANISM_EXPANSIONS
    assert "f. graminearum" in SAFE_ORGANISM_EXPANSIONS

    e = SequenceEnricher(fetch_sequences=False, fetch_smiles=False)
    assert e._expand_organism_name("B. subtilis") == "Bacillus subtilis"
    assert e._expand_organism_name("E. coli") == "Escherichia coli"
    # Non-whitelisted abbreviation passes through unchanged
    assert e._expand_organism_name("X. unknownsp") == "X. unknownsp"


def test_pubchem_gated_on_whitelist():
    """v8: get_smiles_from_pubchem must reject non-mycotoxin substrates
    without making any HTTP call."""
    e = SequenceEnricher(fetch_sequences=False, fetch_smiles=True)
    # Should return None without any network call
    assert e.get_smiles_from_pubchem("ABTS") is None
    assert e.get_smiles_from_pubchem("Benzo[a]pyrene") is None
    assert e.get_smiles_from_pubchem("guaiacol") is None


def test_enrichment_status_top_level_only():
    """v8: enrichment metadata is flat (no nested _enrichment dict)."""
    e = SequenceEnricher(fetch_sequences=False, fetch_smiles=False)
    record = {}
    e._ensure_enrichment_metadata(record)
    assert "enrichment_status" in record
    assert record["enrichment_status"] == "not_attempted"
    # _enrichment dict should NOT be created in v8
    assert "_enrichment" not in record
