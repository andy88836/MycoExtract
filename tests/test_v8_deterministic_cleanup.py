import unittest

from scripts.run_all_papers_full_extraction import (
    apply_inferred_enzyme_guard,
    deterministic_final_cleanup,
    deterministic_record_cleanup,
    extract_commercial_enzyme_degradation_fallback,
    infer_paper_commercial_enzyme_context,
    infer_paper_allowed_scope_context,
    is_out_of_scope_enzyme_system,
)
from src.utils.quality_tier import QualityTierClassifier


classifier = QualityTierClassifier()


def _classify_with_preprocessing(row):
    """Replicate the preprocessing that apply_eligibility used to do, then classify."""
    from scripts.run_all_papers_full_extraction import normalize_auxiliary_fields
    out = normalize_auxiliary_fields(row)
    out = apply_inferred_enzyme_guard(out)
    classifier.classify_record(out)
    return out


class V8DeterministicCleanupTests(unittest.TestCase):
    # ------------------------------------------------------------------
    # Scope filtering helpers
    # ------------------------------------------------------------------
    def assertScopeAllowed(self, row):
        self.assertFalse(
            is_out_of_scope_enzyme_system(row),
            f"Record should be in scope: {row.get('reported_enzyme_name')}",
        )

    def assertScopeRejected(self, row):
        self.assertTrue(
            is_out_of_scope_enzyme_system(row),
            f"Record should be out of scope: {row.get('reported_enzyme_name')}",
        )

    # ------------------------------------------------------------------
    # Scope filtering tests
    # ------------------------------------------------------------------
    def test_keep_free_purified_enzyme(self):
        self.assertScopeAllowed({
            "measurement_type": "kinetic",
            "substrate": "Zearalenone",
            "reported_enzyme_name": "ZEN hydrolase",
            "notes": "purified enzyme was used for kinetic assay",
        })

    def test_keep_purified_recombinant_enzyme(self):
        self.assertScopeAllowed({
            "measurement_type": "kinetic",
            "substrate": "Aflatoxin B1",
            "reported_enzyme_name": "CotA-laccase E186A",
            "notes": "purified recombinant enzyme expressed in E. coli BL21",
        })

    def test_keep_identified_commercial_enzyme(self):
        self.assertScopeAllowed({
            "measurement_type": "degradation",
            "substrate": "Aflatoxin B1",
            "reported_enzyme_name": "commercial laccase",
            "notes": "commercial laccase was used",
        })

    def test_reject_unknown_commercial_preparation(self):
        """Generic 'commercial enzyme preparation' fails Rule 4 (enzyme identity)."""
        row = classifier.classify_record({
            "measurement_type": "degradation",
            "substrate": "Aflatoxin B1",
            "reported_enzyme_name": "commercial enzyme preparation",
            "notes": "commercial enzyme preparation was used",
        })
        self.assertEqual(row["quality_tier"], "Rejected")
        self.assertIn("rule_4_enzyme_identity", row["hard_rule_failures"])

    def test_keep_commercial_named_hydrolase(self):
        self.assertScopeAllowed({
            "measurement_type": "kinetic",
            "substrate": "Zearalenone",
            "reported_enzyme_name": "ZEN hydrolase",
            "notes": "commercial ZEN hydrolase from a preparation company",
        })

    def test_document_allowed_context_keeps_variant_only_record(self):
        context = infer_paper_allowed_scope_context(
            "The recombinant CotA-laccase variants were expressed in E. coli BL21 "
            "and purified before kinetic assays."
        )
        self.assertScopeAllowed({
            "measurement_type": "kinetic",
            "substrate": "Aflatoxin B1",
            "reported_enzyme_name": "E186A",
            "Km_value": 0.109,
            "paper_scope_allowed_context": context,
        })

    def test_reject_cell_free_lysate(self):
        self.assertScopeRejected({
            "measurement_type": "degradation",
            "substrate": "Aflatoxin B1",
            "reported_enzyme_name": "lipase",
            "notes": "P. putida cell-free lysate degraded AFB1",
        })

    def test_reject_culture_supernatant(self):
        self.assertScopeRejected({
            "measurement_type": "degradation",
            "substrate": "Aflatoxin B1",
            "reported_enzyme_name": "laccase",
            "notes": "culture supernatant degraded AFB1",
        })

    def test_reject_whole_cell(self):
        self.assertScopeRejected({
            "measurement_type": "degradation",
            "substrate": "Zearalenone",
            "reported_enzyme_name": "enzyme system",
            "notes": "whole-cell biotransformation of ZEN",
        })

    def test_reject_biological_culture_without_identified_enzyme(self):
        self.assertScopeRejected({
            "measurement_type": "degradation",
            "substrate": "Aflatoxin B1",
            "reported_enzyme_name": "lipase",
            "notes": "Degradation observed within 24 h of incubation with P. putida culture.",
        })

    def test_reject_immobilized_enzyme(self):
        self.assertScopeRejected({
            "measurement_type": "degradation",
            "substrate": "Aflatoxin B1",
            "reported_enzyme_name": "laccase",
            "notes": "immobilized enzyme degraded AFB1",
        })

    def test_reject_enzyme_nanocomplex(self):
        self.assertScopeRejected({
            "measurement_type": "kinetic",
            "substrate": "Zearalenone",
            "reported_enzyme_name": "laccase nanocomplex",
            "notes": "enzyme nanocomplex / fiber material modified by enzyme",
        })

    def test_do_not_reject_not_immobilized(self):
        self.assertScopeAllowed({
            "measurement_type": "degradation",
            "substrate": "Aflatoxin B1",
            "reported_enzyme_name": "purified laccase",
            "notes": "free enzyme, not immobilized",
        })

    def test_ground_truth_scope_expectations(self):
        examples = [
            {
                "label": "catalysts table1 free His6-OPH kept",
                "row": {
                    "measurement_type": "kinetic",
                    "reported_enzyme_name": "His6-OPH",
                    "substrate": "Patulin",
                    "Km_value": 10.9,
                    "notes": "purified His6-OPH free enzyme kinetic assay",
                },
                "allowed": True,
            },
            {
                "label": "catalysts table5 material complex excluded",
                "row": {
                    "measurement_type": "kinetic",
                    "reported_enzyme_name": "His6-OPH/PLE50",
                    "substrate": "Sterigmatocystin",
                    "Km_value": 0.27,
                    "notes": "enzyme nanocomplex / fiber material apparent kinetics",
                },
                "allowed": False,
            },
            {
                "label": "PIIS cell-free lysate excluded",
                "row": {
                    "measurement_type": "degradation",
                    "reported_enzyme_name": "lipase",
                    "substrate": "Aflatoxin B1",
                    "degradation_efficiency": 85,
                    "notes": "P. putida cell-free lysate degraded AFB1",
                },
                "allowed": False,
            },
            {
                "label": "IJMS purified recombinant kept",
                "row": {
                    "measurement_type": "kinetic",
                    "reported_enzyme_name": "CotA-laccase E186A",
                    "substrate": "Aflatoxin B1",
                    "Km_value": 0.109,
                    "notes": "purified recombinant enzyme expressed in E. coli BL21",
                },
                "allowed": True,
            },
            {
                "label": "Toxins17 ZPH1101 kept",
                "row": {
                    "measurement_type": "degradation",
                    "reported_enzyme_name": "ZPH1101",
                    "substrate": "Zearalenone",
                    "degradation_efficiency": 95,
                    "notes": "purified recombinant phosphotransferase ZPH1101",
                },
                "allowed": True,
            },
        ]
        for case in examples:
            rejected = is_out_of_scope_enzyme_system(case["row"])
            self.assertEqual(not rejected, case["allowed"], case["label"])

    # ------------------------------------------------------------------
    # Classifier + preprocessing tests
    # ------------------------------------------------------------------
    def test_inferred_enzyme_not_filled_for_strain_culture(self):
        """apply_inferred_enzyme_guard clears enzyme_name for strain/culture records."""
        row = {
            "organism": "Pseudomonas putida",
            "strain": "MTCC 2445",
            "measurement_type": "degradation",
            "reported_enzyme_name": "lipase",
            "enzyme_name": "lipase",
            "gene_name": "lipase",
            "substrate": "Aflatoxin B1",
            "degradation_efficiency": 85,
            "degradation_efficiency_unit": "%",
            "notes": "HgCl2 inhibition suggests lipase involvement; PCR confirmed lipase gene.",
        }
        # The inferred enzyme guard clears enzyme fields for lysate/strain records
        guarded = apply_inferred_enzyme_guard(row)
        self.assertEqual(guarded.get("enzyme_name"), "")
        self.assertEqual(guarded.get("reported_enzyme_name"), "")
        self.assertEqual(guarded.get("gene_name"), "")
        self.assertEqual(guarded.get("identified_enzyme"), "False")
        self.assertEqual(guarded.get("putative_enzyme"), "True")
        self.assertEqual(guarded.get("degradation_efficiency"), 85)

        # After guard clears enzyme identity, classifier should reject (Rule 4 fails)
        classifier.classify_record(guarded)
        self.assertEqual(guarded["quality_tier"], "Rejected")
        self.assertIn("rule_4_enzyme_identity", guarded["hard_rule_failures"])

    def test_commercial_lipase_kept_as_enzyme(self):
        """A clearly identified commercial enzyme passes classification."""
        row = _classify_with_preprocessing({
            "reported_enzyme_name": "Triacylglycerol lipase from Pseudomonas sp., Sigma-Aldrich",
            "enzyme_name": "triacylglycerol lipase",
            "measurement_type": "degradation",
            "substrate": "Aflatoxin B1",
            "degradation_efficiency": 90,
            "degradation_efficiency_unit": "%",
            "notes": "Commercial lipase enzyme from Sigma-Aldrich was used directly.",
        })
        self.assertNotEqual(row.get("enzyme_name"), "")
        self.assertIn(row["quality_tier"], ("Gold", "Silver", "Bronze"))

    def test_commercial_name_context_adds_vendor(self):
        context = infer_paper_commercial_enzyme_context(
            "Lipase enzyme (Triacylglycerol lipase from Pseudomonas sp) was procured from Sigma-Aldrich."
        )
        row = _classify_with_preprocessing({
            "reported_enzyme_name": "Triacylglycerol lipase from Pseudomonas sp.",
            "enzyme_name": "lipase",
            "measurement_type": "degradation",
            "substrate": "Aflatoxin B1",
            "degradation_efficiency": 90,
            "degradation_efficiency_unit": "%",
            "notes": "Commercial lipase enzyme was used directly.",
            "paper_commercial_enzyme_context": context,
        })
        # Commercial enzyme with degradation data should pass classification
        self.assertIn(row["quality_tier"], ("Gold", "Silver", "Bronze"))

    def test_paper_commercial_context_does_not_protect_lysate_record(self):
        """Lysate records are still rejected even with commercial context."""
        context = infer_paper_commercial_enzyme_context(
            "Lipase enzyme (Triacylglycerol lipase from Pseudomonas sp) was procured from Sigma-Aldrich."
        )
        row = _classify_with_preprocessing({
            "organism": "Pseudomonas putida",
            "strain": "MTCC 2445",
            "reported_enzyme_name": "lipase",
            "enzyme_name": "lipase",
            "measurement_type": "degradation",
            "substrate": "Aflatoxin B1",
            "degradation_efficiency": 85,
            "degradation_efficiency_unit": "%",
            "notes": "Cell-free lysate used. Activity abolished by HgCl2 lipase inhibitor.",
            "paper_commercial_enzyme_context": context,
        })
        # Inferred enzyme guard clears enzyme identity for lysate
        self.assertEqual(row.get("enzyme_name"), "")
        self.assertEqual(row.get("identified_enzyme"), "False")
        self.assertEqual(row["quality_tier"], "Rejected")

    def test_commercial_enzyme_with_inhibitor_note_not_treated_as_inferred(self):
        context = infer_paper_commercial_enzyme_context(
            "Lipase enzyme (Triacylglycerol lipase from Pseudomonas sp) was procured from Sigma-Aldrich."
        )
        row = _classify_with_preprocessing({
            "organism": "Pseudomonas sp.",
            "reported_enzyme_name": "Triacylglycerol lipase from Pseudomonas sp.",
            "enzyme_name": "lipase",
            "measurement_type": "degradation",
            "substrate": "Aflatoxin B1",
            "degradation_efficiency": 90,
            "degradation_efficiency_unit": "%",
            "degradation_time_value": 1,
            "degradation_time_unit": "h",
            "degradation_temperature_value": 37,
            "degradation_temperature_unit": "°C",
            "notes": "Commercial enzyme used as validation; degradation observed at 60 min. Activity abolished by HgCl2 lipase inhibitor.",
            "paper_commercial_enzyme_context": context,
        })
        self.assertEqual(row.get("enzyme_name"), "lipase")
        self.assertIn(row["quality_tier"], ("Gold", "Silver", "Bronze"))

    def test_commercial_enzyme_degradation_fallback_extracts_direct_afb1_record(self):
        text = """
        Lipase enzyme (Triacylglycerol lipase from Pseudomonas sp) was procured from Sigma-Aldrich.
        The tubes were incubated for different time intervals (10, 30 and 60 min), at 37 °C.
        pNPP activity was measured separately with Km = 0.62 mM and Vmax = 405.70 μmol min-1.
        Degradation of AFB1 by commercial lipase enzyme showed that only about 10% reduction
        was noted in 10 min but more than 90% was observed in 60 min.
        """
        records = extract_commercial_enzyme_degradation_fallback(text, [])
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record.get("reported_enzyme_name"), "Triacylglycerol lipase from Pseudomonas sp, Sigma-Aldrich")
        self.assertEqual(record.get("substrate"), "Aflatoxin B1")
        self.assertEqual(record.get("degradation_efficiency"), 90.0)
        self.assertEqual(record.get("degradation_time_value"), 60.0)
        self.assertEqual(record.get("degradation_time_unit"), "min")
        self.assertEqual(record.get("degradation_temperature_value"), 37.0)
        self.assertFalse(record.get("Km_value"))
        self.assertFalse(record.get("kcat_value"))

    def test_no_uniprot_enrichment_for_inferred_enzyme(self):
        row = apply_inferred_enzyme_guard({
            "organism": "Pseudomonas putida",
            "strain": "MTCC 2445",
            "measurement_type": "degradation",
            "reported_enzyme_name": "lipase",
            "enzyme_name": "lipase",
            "uniprot_id": "P00000",
            "sequence": "MSEQUENCE",
            "substrate": "Aflatoxin B1",
            "degradation_efficiency": 85,
            "notes": "HgCl2 inhibition suggests lipase involvement; PCR confirmed lipase gene.",
        })
        self.assertEqual(row.get("uniprot_id"), "")
        self.assertEqual(row.get("sequence"), "")
        self.assertEqual(row.get("enrichment_status"), "blocked_inferred_enzyme")

    def test_bulk_lysate_named_enzyme_without_direct_evidence_cleared(self):
        """Lysate records with inferred enzymes are rejected by the classifier."""
        row = _classify_with_preprocessing({
            "measurement_type": "degradation",
            "reported_enzyme_name": "lipase",
            "enzyme_name": "Lipase",
            "substrate": "Aflatoxin B1",
            "degradation_efficiency": 100,
            "degradation_efficiency_unit": "%",
            "notes": "Lysate was pre-incubated at 70°C for 24 h before measuring AFB1 degradation.",
        })
        self.assertEqual(row.get("enzyme_name"), "")
        self.assertEqual(row["quality_tier"], "Rejected")

    # ------------------------------------------------------------------
    # deterministic_record_cleanup tests (unchanged behavior)
    # ------------------------------------------------------------------
    def test_ijms_dedup(self):
        rows = []
        for prefix in ("", "CotA-laccase "):
            rows.extend([
                {
                    "_pdf_stem": "ijms-25-06455",
                    "measurement_type": "kinetic",
                    "reported_enzyme_name": f"{prefix}WT".strip(),
                    "enzyme_name": "CotA-laccase",
                    "substrate": "Aflatoxin B1",
                    "Km_value": 0.191,
                    "kcat_value": 0.072,
                    "kcat_Km_value": 0.377,
                    "kinetic_temperature_value": 37 if prefix else "",
                    "kinetic_time_value": 8 if prefix else "",
                    "notes": "Table 2 kinetic parameters",
                },
                {
                    "_pdf_stem": "ijms-25-06455",
                    "measurement_type": "kinetic",
                    "reported_enzyme_name": f"{prefix}E186A".strip(),
                    "enzyme_name": "CotA-laccase",
                    "mutations": "E186A",
                    "substrate": "AFB1",
                    "Km_value": 0.109,
                    "kcat_value": 0.073,
                    "kcat_Km_value": 0.67,
                    "notes": "Table 2 kinetic parameters",
                },
                {
                    "_pdf_stem": "ijms-25-06455",
                    "measurement_type": "kinetic",
                    "reported_enzyme_name": f"{prefix}E186R".strip(),
                    "enzyme_name": "CotA-laccase",
                    "mutations": "E186R",
                    "substrate": "Aflatoxin B1",
                    "Km_value": 0.177,
                    "kcat_value": 0.211,
                    "kcat_Km_value": 1.192,
                    "notes": "Table 2 kinetic parameters with fuller evidence text" if prefix else "Table 2",
                },
            ])
        cleaned = deterministic_final_cleanup(rows)
        kinetic = [r for r in cleaned if r.get("measurement_type") == "kinetic"]
        self.assertEqual(len(kinetic), 3)
        self.assertEqual({r.get("mutations") or r.get("reported_enzyme_name") for r in kinetic}, {"WT", "E186A", "E186R"})

    def test_purified_recombinant_not_immobilized(self):
        row = deterministic_record_cleanup({
            "reported_enzyme_name": "CotA-laccase E186A",
            "enzyme_system_type": "immobilized_enzyme",
            "enzyme_state": "immobilized",
            "notes": "Purified recombinant enzyme expressed in E. coli BL21.",
        })
        self.assertEqual(row["enzyme_state"], "free")
        self.assertEqual(row["enzyme_system_type"], "purified_recombinant_enzyme")

    def test_thermodynamic_kcat_removed(self):
        row = deterministic_record_cleanup({
            "measurement_type": "kinetic",
            "substrate": "Zearalenone",
            "Km_value": 11476,
            "Km_unit": "μmol/kg",
            "kcat_value": 0.2679,
            "kcat_unit": "s^-1",
            "source_section": "Table 3",
            "notes": "Thermodynamic temperature-dependent kcat row; activation energy analysis.",
        })
        self.assertEqual(row["Km_value"], 11476)
        self.assertFalse(row.get("kcat_value"))
        self.assertFalse(row.get("kcat_unit"))
        self.assertIn("thermodynamic_kcat_removed_from_primary", row.get("error_flags"))

    def test_pretreatment_temperature_remap(self):
        row = deterministic_record_cleanup({
            "measurement_type": "degradation",
            "substrate": "Aflatoxin B1",
            "degradation_efficiency": 100,
            "degradation_temperature_value": 70,
            "degradation_temperature_unit": "°C",
            "notes": "Cell-free lysate was pre-incubated for thermal stability before assay; retained activity measured.",
        })
        self.assertFalse(row.get("degradation_temperature_value"))
        self.assertEqual(row.get("stability_temperature_value"), 70)
        self.assertEqual(row.get("stability_temperature_unit"), "°C")
        self.assertIn("stability_pretreatment_context", row.get("error_flags"))

    def test_degradation_record_clears_kinetic_fields(self):
        row = deterministic_record_cleanup({
            "measurement_type": "degradation",
            "reported_enzyme_name": "lipase",
            "substrate": "Aflatoxin B1",
            "degradation_efficiency": 85,
            "Km_value": 0.62,
            "Km_unit": "mM",
            "notes": "some auxiliary assay",
        })
        self.assertFalse(row.get("Km_value"))
        self.assertFalse(row.get("Km_unit"))
        self.assertEqual(row.get("degradation_efficiency"), 85)
        self.assertIn(
            "Removed kinetic fields from degradation record",
            row.get("notes"),
        )

    def test_kinetic_non_mycotoxin_substrate_rejected_by_classifier(self):
        """Non-mycotoxin substrates pass deterministic_final_cleanup but fail classifier Rule 1."""
        rows = deterministic_final_cleanup([{
            "measurement_type": "kinetic",
            "reported_enzyme_name": "lipase",
            "substrate": "p-nitrophenyl palmitate",
            "Km_value": 0.62,
            "Km_unit": "mM",
        }])
        # deterministic_final_cleanup no longer filters by substrate
        self.assertEqual(len(rows), 1)
        # But classifier rejects it
        classifier.classify_record(rows[0])
        self.assertEqual(rows[0]["quality_tier"], "Rejected")
        self.assertIn("rule_1_mycotoxin_substrate", rows[0]["hard_rule_failures"])
        # Verify the whitelist check directly
        self.assertFalse(classifier._is_mycotoxin_substrate("p-nitrophenyl palmitate"))
        self.assertTrue(classifier._is_mycotoxin_substrate("Aflatoxin B1"))

    def test_teacher_cannot_add_kinetic_to_degradation(self):
        row = deterministic_record_cleanup({
            "measurement_type": "degradation",
            "substrate": "Aflatoxin B1",
            "degradation_efficiency": 85,
            "Km_value": 0.62,
            "Km_unit": "mM",
            "source_channel": "text",
            "source_model": "teacher_harmonized",
            "notes": "teacher added Km after harmonization",
        })
        self.assertFalse(row.get("Km_value"))
        self.assertFalse(row.get("Km_unit"))
        self.assertEqual(row.get("degradation_efficiency"), 85)

    def test_same_paper_afb1_kinetic_and_degradation_are_separate(self):
        rows = deterministic_final_cleanup([
            {
                "_pdf_stem": "paper1",
                "measurement_type": "kinetic",
                "reported_enzyme_name": "enzyme A",
                "substrate": "Aflatoxin B1",
                "Km_value": 0.5,
                "Km_unit": "mM",
            },
            {
                "_pdf_stem": "paper1",
                "measurement_type": "degradation",
                "reported_enzyme_name": "enzyme A",
                "substrate": "Aflatoxin B1",
                "degradation_efficiency": 85,
                "degradation_efficiency_unit": "%",
            },
        ])
        self.assertEqual(len(rows), 2)
        self.assertEqual({r.get("measurement_type") for r in rows}, {"kinetic", "degradation"})

    def test_multiplier_warning_cleanup(self):
        row = deterministic_record_cleanup({
            "measurement_type": "kinetic",
            "reported_enzyme_name": "His6-OPH",
            "substrate": "Patulin",
            "kcat_Km_value": 2480,
            "kcat_Km_unit": "M^-1 min^-1",
            "kinetic_unit_multiplier": 1000,
            "kinetic_unit_source_text": "10^3 M^-1 min^-1",
            "evidence_text": "Patulin: Vmax/(E0×Km)=2.48 ×10^3 M−1 min−1",
            "notes": "kcat_Km_unit has multiplier scaling ambiguity: table header shows 10^3 M−1 min−1 but formatting is unclear.",
            "error_flags": ["table_multiplier_scaling_error", "metric_condition_mismatch"],
            "human_review_required": True,
        })
        self.assertNotIn("table_multiplier_scaling_error", row.get("error_flags"))
        self.assertIn("metric_condition_mismatch", row.get("error_flags"))
        self.assertNotIn("scaling ambiguity", row.get("notes").lower())
        self.assertIn("Multiplier applied from table header", row.get("notes"))

    def test_sample_specific_complex_table_rows_are_preserved_without_paper_specific_conditions(self):
        rows = []
        for enzyme, substrate in [
            ("Thermolysin/PLE50", "Zearalenone"),
            ("Thermolysin/PLE50", "Ochratoxin A"),
            ("His6-OPH/PLE50", "Sterigmatocystin"),
        ]:
            for sample in (1, 2, 3):
                rows.append({
                    "measurement_type": "kinetic",
                    "reported_enzyme_name": enzyme,
                    "substrate": substrate,
                    "Km_value": sample,
                    "kcat_value": sample + 1,
                    "kcat_Km_value": sample + 2,
                    "source_channel": "table_image_rescue",
                    "error_flags": ["table_image_rescue"],
                    "notes": f"Deterministic complex-table fallback from table unknown; sample #{sample}",
                })
        cleaned = deterministic_final_cleanup(rows)
        self.assertEqual(len(cleaned), 9)
        for row in cleaned:
            self.assertFalse(row.get("kinetic_temperature_value"))
            self.assertFalse(row.get("kinetic_ph"))
            self.assertFalse(row.get("kinetic_time_value"))

    def test_catalysts_table1_unit_equivalent_text_duplicate_removed(self):
        rows = [
            {
                "_pdf_stem": "catalysts-12-01095",
                "measurement_type": "kinetic",
                "reported_enzyme_name": "His6-OPH",
                "substrate": "Patulin",
                "Km_value": 10.9,
                "kcat_value": 27.1,
                "kcat_Km_value": 2.48,
                "kinetic_unit_multiplier": 1000,
                "kinetic_unit_source_text": "kcat/Km (x10^3 M^-1 min^-1)",
                "locked_candidate": True,
                "source_channel": "parsed_table",
            },
            {
                "_pdf_stem": "catalysts-12-01095",
                "measurement_type": "kinetic",
                "reported_enzyme_name": "His6-OPH",
                "substrate": "Patulin",
                "Km_value": 10.9,
                "kcat_value": 0.452,
                "kcat_Km_value": 41.4,
                "source_channel": "text",
                "source_model": "teacher_harmonized",
            },
        ]
        cleaned = deterministic_final_cleanup(rows)
        kinetic = [r for r in cleaned if r.get("measurement_type") == "kinetic"]
        self.assertEqual(len(kinetic), 1)
        self.assertEqual(float(kinetic[0]["kcat_Km_value"]), 2480.0)
        self.assertEqual(float(kinetic[0]["kcat_value"]), 27.1)

    def test_sample_specific_table_rows_cover_text_summary(self):
        rows = [
            {
                "_pdf_stem": "catalysts-12-01095",
                "measurement_type": "kinetic",
                "reported_enzyme_name": "His6-OPH/PLE50",
                "enzyme_name": "His6-OPH/PLE50",
                "substrate": "Sterigmatocystin",
                "Km_value": 0.27,
                "kcat_value": 189,
                "kcat_Km_value": 705000,
                "source_channel": "table_image_rescue",
                "locked_candidate": True,
                "notes": "sample #1",
            },
            {
                "_pdf_stem": "catalysts-12-01095",
                "measurement_type": "kinetic",
                "reported_enzyme_name": "His6-OPH/PLE50",
                "enzyme_name": "OPH",
                "substrate": "Sterigmatocystin",
                "Km_value": 0.25,
                "kcat_value": 3.75,
                "kcat_Km_value": 15000,
                "source_channel": "text",
                "source_model": "teacher_harmonized",
                "notes": "text summary without sample label",
            },
        ]
        cleaned = deterministic_final_cleanup(rows)
        self.assertEqual(len(cleaned), 1)
        self.assertEqual(cleaned[0]["Km_value"], 0.27)


if __name__ == "__main__":
    unittest.main()
