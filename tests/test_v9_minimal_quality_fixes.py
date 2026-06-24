import unittest

from scripts.run_all_papers_full_extraction import (
    V9_SCHEMA_COLUMNS,
    apply_primary_hard_gate,
    apply_inferred_enzyme_guard,
    canonical_substrate_name,
    deterministic_final_cleanup,
    deterministic_record_cleanup,
    fill_missing_degradation_time_from_peer_records,
    final_deduplicate_records,
    project_to_v9,
)
from src.pipeline.post_processor import FULL_SCHEMA
from src.utils.quality_tier import QualityTierClassifier


classifier = QualityTierClassifier()


class V9MinimalQualityFixesTests(unittest.TestCase):
    def test_d207q_variant_not_inferred_strain_culture(self):
        row = apply_inferred_enzyme_guard({
            "measurement_type": "kinetic",
            "reported_enzyme_name": "D207Q",
            "enzyme_name": "laccase",
            "mutations": "D207Q",
            "substrate": "Aflatoxin B1",
            "Km_value": 0.12,
            "Km_unit": "mM",
            "notes": "Kinetic parameters of variants; inhibitor evidence discussed elsewhere.",
        })
        self.assertEqual(row["enzyme_name"], "laccase")
        self.assertEqual(row["reported_enzyme_name"], "D207Q")
        self.assertNotEqual(row.get("enrichment_status"), "blocked_inferred_enzyme")
        classified = classifier.classify_record(dict(row))
        self.assertNotEqual(classified["quality_tier"], "Rejected")

    def test_extracellular_enzymes_not_valid_enzyme_name(self):
        row = classifier.classify_record({
            "measurement_type": "degradation",
            "reported_enzyme_name": "extracellular enzymes",
            "substrate": "Zearalenone",
            "degradation_efficiency": 80,
            "degradation_efficiency_unit": "%",
        })
        self.assertEqual(row["quality_tier"], "Rejected")
        self.assertIn("rule_4_enzyme_identity", row["hard_rule_failures"])

    def test_unidentified_biological_material_rejected(self):
        row = classifier.classify_record({
            "measurement_type": "degradation",
            "reported_enzyme_name": "OTA-hydrolytic enzyme",
            "substrate": "Ochratoxin A",
            "degradation_efficiency": 70,
            "degradation_efficiency_unit": "%",
            "notes": "Pleurotus ostreatus mushroom powder produced OTalpha during digestion.",
        })
        self.assertEqual(row["quality_tier"], "Rejected")
        self.assertEqual(row["QC_Status"], "unidentified_biological_system_not_enzyme_record")

    def test_p450_bioactivation_rejected(self):
        row = classifier.classify_record({
            "measurement_type": "kinetic",
            "enzyme_name": "CYP3A4",
            "substrate": "Aflatoxin B1",
            "Km_value": 15,
            "Km_unit": "uM",
            "products": "AFBO",
            "notes": "Human liver microsomes and cDNA-expressed CYP3A4 formed aflatoxin epoxide DNA adducts.",
        })
        self.assertEqual(row["quality_tier"], "Rejected")
        self.assertIn("rule_2_enzymatic_process", row["hard_rule_failures"])
        self.assertEqual(row["QC_Status"], "metabolic_activation_or_toxic_biotransformation")

    def test_p450_afq1_metabolism_rejected(self):
        row = classifier.classify_record({
            "measurement_type": "kinetic",
            "reported_enzyme_name": "3A4/OR+b5",
            "enzyme_name": "P450 3A4",
            "substrate": "Aflatoxin B1",
            "Km_value": 39.77,
            "Km_unit": "uM",
            "products": "AFQ1",
            "notes": "Recombinant enzyme expressed in baculovirus system with cytochrome b5. Kinetic parameters for AFQ1 formation.",
        })
        self.assertEqual(row["quality_tier"], "Rejected")
        self.assertIn("rule_2_enzymatic_process", row["hard_rule_failures"])
        self.assertEqual(row["QC_Status"], "metabolic_activation_or_toxic_biotransformation")

    def test_nanomaterial_catalyst_not_biological_enzyme_rejected(self):
        row = classifier.classify_record(deterministic_record_cleanup({
            "measurement_type": "degradation",
            "reported_enzyme_name": "MF@CRHHT",
            "enzyme_name": "MF@CRHHT",
            "substrate": "Aflatoxin B1",
            "degradation_efficiency": 91.1,
            "degradation_efficiency_unit": "%",
            "notes": "Catalytic system is a superparamagnetic Mn-Fe biocomposite, not a biological enzyme.",
            "evidence_text": "MF@CRHHT [0.4 g/L], PDS [1.0 mmol/L], Efficiency 91.10%, This work.",
        }))
        self.assertEqual(row["quality_tier"], "Rejected")
        self.assertIn("rule_2_enzymatic_process", row["hard_rule_failures"])

    def test_porous_carbon_material_rejected(self):
        row = classifier.classify_record(deterministic_record_cleanup({
            "measurement_type": "degradation",
            "reported_enzyme_name": "Fe/N-PC",
            "enzyme_name": "Fe/N-PC",
            "substrate": "Aflatoxin B1",
            "degradation_efficiency": 99.88,
            "degradation_efficiency_unit": "%",
            "notes": "Catalytic system is a porous carbon material, not a biological enzyme.",
        }))
        self.assertEqual(row["quality_tier"], "Rejected")
        self.assertIn("rule_2_enzymatic_process", row["hard_rule_failures"])

    def test_afbo_not_valid_substrate_alias(self):
        row = classifier.classify_record({
            "measurement_type": "kinetic",
            "enzyme_name": "GSTA2X",
            "substrate": "AFBO",
            "Km_value": 10,
            "Km_unit": "uM",
        })
        self.assertEqual(row["quality_tier"], "Rejected")
        self.assertIn("rule_1_mycotoxin_substrate", row["hard_rule_failures"])
        self.assertFalse(classifier._is_mycotoxin_substrate("AFBO"))

    def test_hill_k_not_km(self):
        row = deterministic_record_cleanup({
            "measurement_type": "kinetic",
            "enzyme_name": "CYP3A4",
            "substrate": "Aflatoxin B1",
            "Km_value": 139,
            "Km_unit": "uM",
            "notes": "Hill equation S50 and nH were reported; this is a composite constant K.",
        })
        self.assertFalse(row.get("Km_value"))
        self.assertFalse(row.get("Km_unit"))
        self.assertIn("hill_constant_not_michaelis_menten_km", row.get("error_flags"))

    def test_relative_activity_100_not_degradation_efficiency(self):
        row = deterministic_record_cleanup({
            "measurement_type": "degradation",
            "enzyme_name": "laccase",
            "substrate": "Aflatoxin B1",
            "degradation_efficiency": 100,
            "degradation_efficiency_unit": "%",
            "notes": "Relative activity was set as 100% for the untreated control.",
        })
        self.assertFalse(row.get("degradation_efficiency"))
        self.assertFalse(row.get("degradation_efficiency_unit"))
        self.assertIn("wrong_metric_type_relative_activity_baseline", row.get("error_flags"))

    def test_degradation_time_exported(self):
        row = deterministic_record_cleanup({
            "measurement_type": "degradation",
            "enzyme_name": "commercial laccase",
            "substrate": "Aflatoxin B1",
            "degradation_efficiency": 85,
            "degradation_efficiency_unit": "%",
            "notes": "AFB1 degradation reached 85% after 24 h.",
        })
        self.assertEqual(row["degradation_time_value"], 24)
        self.assertEqual(row["degradation_time_unit"], "h")

    def test_substrate_concentration_value_unit_split(self):
        row = deterministic_record_cleanup({
            "measurement_type": "degradation",
            "enzyme_name": "laccase",
            "substrate": "Aflatoxin B1",
            "substrate_concentration": "0.4 \u00b5g/mL",
            "degradation_efficiency": 50,
            "degradation_efficiency_unit": "%",
        })
        self.assertEqual(row["substrate_concentration_value"], 0.4)
        self.assertEqual(row["substrate_concentration_unit"], "\u00b5g/mL")

    def test_mediator_fields_minimal(self):
        row = deterministic_record_cleanup({
            "measurement_type": "degradation",
            "enzyme_name": "CotA",
            "substrate": "Aflatoxin B1",
            "degradation_efficiency": 75,
            "degradation_efficiency_unit": "%",
            "notes": "ABTS mediator 1 mM in the AFB1 degradation reaction mixture.",
        })
        self.assertEqual(row["mediator_name"].lower(), "abts")
        self.assertEqual(row["mediator_concentration"], 1)
        self.assertEqual(row["mediator_concentration_unit"], "mM")

    def test_table_title_abts_not_mediator_for_afb1_kinetic_row(self):
        row = deterministic_record_cleanup({
            "measurement_type": "kinetic",
            "reported_enzyme_name": "D207Q",
            "enzyme_name": "Laccase",
            "mutations": "D207Q",
            "substrate": "Aflatoxin B1",
            "Km_value": 8.06,
            "Km_unit": "\u00b5M",
            "kcat_value": 2.37e-05,
            "kcat_unit": "s^-1",
            "mediator_name": "ABTS",
            "evidence_text": (
                "Table 2 Kinetic constants of wild-type versus variants on AFB1 and ABTS. "
                "AFB1 row: Km, kcat, kcat/Km. D207Q values: Km=8.06."
            ),
            "notes": "Kinetic parameters for AFB1 degradation measured at pH 7.0 and 45\u00b0C.",
        })
        self.assertFalse(row.get("mediator_name"))
        self.assertIn("mediator_removed_not_in_current_reaction_context", row.get("error_flags", ""))

    def test_mediator_percent_not_concentration(self):
        row = deterministic_record_cleanup({
            "measurement_type": "degradation",
            "enzyme_name": "CotA",
            "substrate": "Zearalenone",
            "degradation_efficiency": 96,
            "degradation_efficiency_unit": "%",
            "notes": "CotA-ABTS system achieved 96% ZEN degradation.",
        })
        self.assertEqual(row["mediator_name"].lower(), "abts")
        self.assertFalse(row.get("mediator_concentration"))
        self.assertFalse(row.get("mediator_concentration_unit"))

    def test_mediator_concentration_requires_local_evidence(self):
        row = deterministic_record_cleanup({
            "measurement_type": "degradation",
            "enzyme_name": "CotA",
            "substrate": "Zearalenone",
            "degradation_efficiency": 96,
            "degradation_efficiency_unit": "%",
            "mediator_name": "ABTS",
            "mediator_concentration": 96,
            "mediator_concentration_unit": "mM",
            "notes": "CotA-ABTS system achieved 96% ZEN degradation.",
        })
        self.assertEqual(row["mediator_name"].lower(), "abts")
        self.assertFalse(row.get("mediator_concentration"))
        self.assertFalse(row.get("mediator_concentration_unit"))
        self.assertIn("mediator_concentration_requires_local_evidence", row.get("error_flags", ""))

    def test_no_mediator_context_clears_mediator_fields(self):
        row = deterministic_record_cleanup({
            "measurement_type": "degradation",
            "enzyme_name": "CotA",
            "substrate": "Zearalenone",
            "degradation_efficiency": 96,
            "degradation_efficiency_unit": "%",
            "mediator_name": "ABTS",
            "mediator_concentration": 96,
            "mediator_concentration_unit": "%",
            "notes": "Direct oxidation by CotA alone (no mediator).",
        })
        self.assertFalse(row.get("mediator_name"))
        self.assertFalse(row.get("mediator_concentration"))
        self.assertFalse(row.get("mediator_concentration_unit"))

    def test_t2_toxin_canonical_dedup(self):
        rows = deterministic_final_cleanup([
            {
                "_pdf_stem": "paper",
                "measurement_type": "kinetic",
                "reported_enzyme_name": "TRI101",
                "substrate": "T2",
                "Km_value": 15,
                "Km_unit": "uM",
                "kcat_value": 12,
                "kcat_unit": "s-1",
            },
            {
                "_pdf_stem": "paper",
                "measurement_type": "kinetic",
                "reported_enzyme_name": "TRI101",
                "substrate": "T-2 toxin",
                "Km_value": 15,
                "Km_unit": "uM",
                "kcat_value": 12,
                "kcat_unit": "s-1",
                "source_channel": "parsed_table",
                "locked_candidate": "true",
                "evidence_text": "Gold table row",
            },
        ])
        self.assertEqual(len(rows), 1)
        self.assertEqual(canonical_substrate_name(rows[0]["substrate"]), "t-2 toxin")
        self.assertIn("Gold table row", rows[0]["evidence_text"])

    def test_don_table_text_dedup_prefers_gold_table_row(self):
        rows = deterministic_final_cleanup([
            {
                "_pdf_stem": "paper",
                "measurement_type": "kinetic",
                "reported_enzyme_name": "FUMD",
                "mutations": "WT",
                "substrate": "DON",
                "Km_value": 2.0,
                "Km_unit": "uM",
                "kcat_Km_value": 4.7,
                "kcat_Km_unit": "M-1 s-1",
                "kinetic_unit_multiplier": 1000000,
                "notes": "Text row reports 4.7 x 10^6.",
            },
            {
                "_pdf_stem": "paper",
                "measurement_type": "kinetic",
                "reported_enzyme_name": "FUMD",
                "substrate": "Deoxynivalenol",
                "Km_value": 2.0,
                "Km_unit": "uM",
                "kcat_Km_value": 4700000,
                "kcat_Km_unit": "M-1 s-1",
                "source_channel": "parsed_table",
                "locked_candidate": "true",
                "evidence_text": "Gold normalized table row",
            },
        ])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["locked_candidate"], "true")
        self.assertEqual(float(rows[0]["kcat_Km_value"]), 4700000)

    def test_observed_don_multiplier_notes_dedup_prefers_gold_table_row(self):
        # After fixing the double-scaling bug, multipliers are no longer parsed
        # from notes/evidence_text fields.  The Silver record (4.7) and Gold
        # record (4700000) have different kcat_Km values and are NOT deduped.
        # In the full pipeline, remove_silver_when_gold_exists() removes the
        # Silver record downstream.
        rows = deterministic_final_cleanup([
            {
                "pdf_file": "10.1128_aem.01738-10.pdf",
                "measurement_type": "kinetic",
                "reported_enzyme_name": "Fps2TRI101",
                "enzyme_name": "TRI101",
                "substrate": "Deoxynivalenol",
                "Km_value": 41,
                "Km_unit": "uM",
                "kcat_value": 195,
                "kcat_unit": "s-1",
                "kcat_Km_value": 4.7,
                "kcat_Km_unit": "M^-1 s^-1",
                "notes": "kcat/Km value 4.7 × 10^6 M^-1 s^-1; multiplier not captured.",
                "quality_tier": "Silver",
            },
            {
                "pdf_file": "10.1128_aem.01738-10.pdf",
                "measurement_type": "kinetic",
                "reported_enzyme_name": "Fps2TRI101",
                "enzyme_name": "TRI101",
                "substrate": "Deoxynivalenol",
                "Km_value": 41,
                "Km_unit": "uM",
                "kcat_value": 195,
                "kcat_unit": "s-1",
                "kcat_Km_value": 4700000,
                "kcat_Km_unit": "M^-1 s^-1",
                "quality_tier": "Gold",
                "source_channel": "parsed_table",
                "locked_candidate": "true",
                "evidence_text": "Gold normalized table row",
            },
        ])
        gold_rows = [r for r in rows if r.get("quality_tier") == "Gold"]
        self.assertEqual(len(gold_rows), 1)
        self.assertEqual(gold_rows[0]["locked_candidate"], "true")
        self.assertEqual(float(gold_rows[0]["kcat_Km_value"]), 4700000)

    def test_gold_over_silver_dedup_after_csv_projection(self):
        # After fixing the double-scaling bug, multipliers are no longer parsed
        # from notes fields.  The Silver record (4.7) and Gold record (4700000)
        # have different kcat_Km values and are NOT deduped here.  In the full
        # pipeline, remove_silver_when_gold_exists() handles this case.
        rows = deterministic_final_cleanup([
            {
                "pdf_file": "10.1128_aem.01738-10.pdf",
                "measurement_type": "kinetic",
                "reported_enzyme_name": "Fps2TRI101",
                "enzyme_name": "TRI101",
                "substrate": "Deoxynivalenol",
                "Km_value": 41,
                "Km_unit": "uM",
                "kcat_value": 195,
                "kcat_unit": "s-1",
                "kcat_Km_value": 4.7,
                "kcat_Km_unit": "M^-1 s^-1",
                "quality_tier": "Silver",
                "notes": "kcat/Km value 4.7 × 10^6 M^-1 s^-1; multiplier not captured in numeric value.",
            },
            {
                "pdf_file": "10.1128_aem.01738-10.pdf",
                "measurement_type": "kinetic",
                "reported_enzyme_name": "Fps2TRI101",
                "enzyme_name": "TRI101",
                "substrate": "Deoxynivalenol",
                "Km_value": 41,
                "Km_unit": "uM",
                "kcat_value": 195,
                "kcat_unit": "s-1",
                "kcat_Km_value": 4700000,
                "kcat_Km_unit": "M^-1 s^-1",
                "quality_tier": "Gold",
                "notes": "Purified recombinant enzyme expressed in E. coli. Kinetic assay performed at 25°C.",
            },
        ])
        gold_rows = [r for r in rows if r.get("quality_tier") == "Gold"]
        self.assertEqual(len(gold_rows), 1)
        self.assertEqual(gold_rows[0]["quality_tier"], "Gold")
        self.assertEqual(float(gold_rows[0]["kcat_Km_value"]), 4700000)

    def test_literature_comparison_rejected(self):
        row = classifier.classify_record(deterministic_record_cleanup({
            "measurement_type": "kinetic",
            "enzyme_name": "CotA",
            "substrate": "Aflatoxin B1",
            "Km_value": 0.5,
            "Km_unit": "uM",
            "source_section": "Table comparing previous study values from other researchers.",
        }))
        self.assertEqual(row["quality_tier"], "Rejected")
        self.assertIn("rule_5_original_experiment", row["hard_rule_failures"])
        self.assertEqual(row["QC_Status"], "literature_comparison_not_current_experiment")

    def test_current_study_kinetic_not_rejected_by_nearby_literature_context(self):
        cleaned = deterministic_record_cleanup({
            "measurement_type": "kinetic",
            "reported_enzyme_name": "CotA laccase",
            "enzyme_name": "CotA laccase",
            "substrate": "Zearalenone",
            "Km_value": 90.43,
            "Km_unit": "\u00b5g/mL",
            "kcat_value": 0.11,
            "kcat_unit": "s^-1",
            "source_section": "Fig. 3D current study Michaelis-Menten kinetic experiment.",
            "notes": "Kinetic parameters determined at 37\u00b0C, pH 8.0. A separate literature comparison table appears elsewhere.",
        })
        self.assertNotEqual(cleaned.get("QC_Status"), "literature_comparison_not_current_experiment")
        row = classifier.classify_record(cleaned)
        self.assertNotEqual(row["quality_tier"], "Rejected")
        self.assertNotIn("rule_5_original_experiment", row.get("hard_rule_failures", ""))

    def test_reference_column_prior_work_rejected(self):
        row = classifier.classify_record(deterministic_record_cleanup({
            "measurement_type": "degradation",
            "reported_enzyme_name": "Commercial laccase Ery4",
            "enzyme_name": "laccase",
            "substrate": "Zearalenone",
            "degradation_efficiency": 82,
            "degradation_efficiency_unit": "%",
            "evidence_text": "<table><tr><td>Enzyme</td><td>Degradation rates</td><td>Reference</td></tr>"
                             "<tr><td>Commercial laccase Ery4</td><td>82%</td><td>Banu et al. (2014)</td></tr></table>",
        }))
        self.assertEqual(row["quality_tier"], "Rejected")
        self.assertIn("literature_comparison_source", row["error_flags"])
        self.assertEqual(row["QC_Status"], "literature_comparison_not_current_experiment")

    def test_commercial_signal_does_not_override_crude_compound_immobilized_system(self):
        row = deterministic_record_cleanup({
            "measurement_type": "kinetic",
            "reported_enzyme_name": "compound enzymes (laccase + Aspergillus niger crude enzyme)",
            "enzyme_name": "Compound enzymes (laccase + Aspergillus niger crude enzyme)",
            "substrate": "Aflatoxin B1",
            "Km_value": 5.5e-05,
            "Km_unit": "mol/L",
            "notes": (
                "Laccase was purchased from Sigma-Aldrich, but kinetic parameters were obtained "
                "from an immobilized compound enzyme system containing laccase and Aspergillus niger crude enzyme."
            ),
        })
        self.assertNotEqual(row.get("enzyme_system_type"), "clearly_identified_commercial_enzyme")
        classified = classifier.classify_record(row)
        self.assertEqual(classified["quality_tier"], "Rejected")
        self.assertIn("rule_2_enzymatic_process", classified["hard_rule_failures"])

    def test_immobilized_microsphere_system_rejected(self):
        row = classifier.classify_record(deterministic_record_cleanup({
            "measurement_type": "degradation",
            "reported_enzyme_name": "ZLHY6 enzyme",
            "enzyme_name": "ZLHY6",
            "substrate": "Zearalenone",
            "degradation_efficiency": 93.18,
            "degradation_efficiency_unit": "%",
            "enzyme_state": "immobilized",
            "notes": "SA/Mt./EZ microspheres are sodium alginate/montmorillonite immobilized ZLHY6 enzyme microspheres.",
        }))
        self.assertEqual(row["quality_tier"], "Rejected")
        self.assertIn("rule_2_enzymatic_process", row["hard_rule_failures"])

    def test_specific_activity_rate_not_degradation_efficiency(self):
        row = classifier.classify_record(deterministic_record_cleanup({
            "measurement_type": "degradation",
            "reported_enzyme_name": "TaGST-02",
            "enzyme_name": "TaGST-02",
            "substrate": "Deoxynivalenol",
            "degradation_efficiency": 0.00117,
            "degradation_efficiency_unit": "%",
            "evidence_text": (
                "Table 2: Apparent specific activities. TaGST-02 with DON: "
                "1.17e-5 µmol min⁻¹ mg⁻¹."
            ),
            "notes": (
                "Specific activity converted to degradation efficiency by assuming "
                "1 µmol min⁻¹ mg⁻¹ = 100% conversion per minute per mg enzyme."
            ),
        }))
        self.assertFalse(row.get("degradation_efficiency"))
        self.assertIn("wrong_metric_type_specific_activity_rate", row.get("error_flags", ""))
        self.assertEqual(row["quality_tier"], "Rejected")
        self.assertIn("rule_3_quantitative_metric", row["hard_rule_failures"])
        self.assertIn("GST-mediated glutathione conjugation", row["notes"])
        self.assertNotIn("AFBO-GSH", row["notes"])

    def test_rejected_biocatalyst_names_are_cleared(self):
        row = deterministic_record_cleanup({
            "measurement_type": "degradation",
            "reported_enzyme_name": "OTA-hydrolytic enzyme",
            "enzyme_name": "OTA-hydrolytic enzyme",
            "substrate": "Ochratoxin A",
            "degradation_efficiency": 100,
            "degradation_efficiency_unit": "%",
            "notes": "Enzyme inferred from OTalpha detection in Pleurotus ostreatus powder.",
        })
        self.assertFalse(row.get("reported_enzyme_name"))
        self.assertFalse(row.get("enzyme_name"))
        self.assertEqual(row.get("identified_enzyme"), "False")
        self.assertEqual(row.get("putative_enzyme"), "True")

    def test_stability_activity_percent_not_degradation_efficiency(self):
        row = classifier.classify_record(deterministic_record_cleanup({
            "measurement_type": "degradation",
            "reported_enzyme_name": "ZEN degrading enzyme",
            "enzyme_name": "ZHD101",
            "gene_name": "ZHD101",
            "substrate": "Zearalenone",
            "degradation_efficiency": 75,
            "degradation_efficiency_unit": "%",
            "evidence_text": (
                "Table 2: Comparison of thermal stability of ZEN degrading enzymes. "
                "Remained activities: Less than 25% activity after incubation at 45 °C for 10 min."
            ),
            "source_section": "table_2",
        }))
        self.assertFalse(row.get("degradation_efficiency"))
        self.assertIn("wrong_metric_type_activity_or_stability", row.get("error_flags", ""))
        self.assertEqual(row["quality_tier"], "Rejected")
        self.assertIn("rule_3_quantitative_metric", row["hard_rule_failures"])

    def test_hydrolysis_efficiency_degradation_retained(self):
        row = classifier.classify_record(deterministic_record_cleanup({
            "measurement_type": "degradation",
            "reported_enzyme_name": "ZEN degrading enzyme M2",
            "enzyme_name": "ZHD101",
            "gene_name": "ZHD101",
            "mutations": "M2",
            "is_recombinant": True,
            "enzyme_state": "free",
            "substrate": "Zearalenone",
            "degradation_efficiency": 75.6,
            "degradation_efficiency_unit": "%",
            "evidence_text": (
                "Table 3 Hydrolysis efficiencies of ZEN degrading enzymes under simulated pig stomach "
                "acidic condition for 30 min. pH 4.2, M2, Degradation rate (%) 75.6"
            ),
            "source_section": "table_3",
        }))
        self.assertEqual(float(row["degradation_efficiency"]), 75.6)
        self.assertNotEqual(row["quality_tier"], "Rejected")

    def test_absurd_kcat_km_repaired_from_local_table_row(self):
        row = deterministic_record_cleanup({
            "measurement_type": "kinetic",
            "reported_enzyme_name": "WT",
            "enzyme_name": "ZEN degrading enzyme",
            "mutations": "WT",
            "substrate": "Zearalenone",
            "Km_value": 283.61,
            "Km_unit": "µM",
            "kcat_value": 0.292,
            "kcat_unit": "s^-1",
            "kcat_Km_value": 1.031e34,
            "kcat_Km_unit": "s^-1 M^-1",
            "evidence_text": (
                "Table 4 Kinetic parameters of WT, M1 or M2. ZEN degrading enzyme, "
                "Km (µM), a (1), Ka/Km (s1M−1). WT: 283.61 ± 17.24, "
                "0.292 ± 0.0026, 1031.00 ± 22.36."
            ),
        })
        self.assertEqual(float(row["kcat_Km_value"]), 1031.0)
        self.assertIn("absurd_kcat_km_repaired_from_evidence", row.get("error_flags", ""))

    def test_duplicate_text_identity_restores_locked_table_generic_enzyme(self):
        rows = deterministic_final_cleanup([
            {
                "pdf_file": "paper.pdf",
                "measurement_type": "kinetic",
                "reported_enzyme_name": "WT",
                "enzyme_name": "ZEN degrading enzyme",
                "mutations": "WT",
                "substrate": "Zearalenone",
                "Km_value": 283.61,
                "Km_unit": "µM",
                "kcat_value": 0.292,
                "kcat_unit": "s^-1",
                "kcat_Km_value": 1031,
                "kcat_Km_unit": "s^-1 M^-1",
                "source_channel": "parsed_table",
                "locked_candidate": "true",
            },
            {
                "pdf_file": "paper.pdf",
                "measurement_type": "kinetic",
                "reported_enzyme_name": "ZEN degrading enzyme WT",
                "enzyme_name": "ZHD101",
                "gene_name": "ZHD101",
                "mutations": "WT",
                "is_recombinant": True,
                "enzyme_state": "free",
                "substrate": "Zearalenone",
                "Km_value": 283.61,
                "Km_unit": "µM",
                "kcat_value": 0.292,
                "kcat_unit": "s^-1",
                "kcat_Km_value": 1031,
                "kcat_Km_unit": "s^-1 M^-1",
                "source_channel": "text",
            },
        ])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["enzyme_name"], "ZHD101")
        self.assertEqual(rows[0]["gene_name"], "ZHD101")
        self.assertEqual(rows[0].get("identified_enzyme"), "True")
        self.assertNotEqual(classifier.classify_record(dict(rows[0]))["quality_tier"], "Rejected")

    def test_no_unrequested_export_fields(self):
        projected = project_to_v9([{}])[0]
        for forbidden in [
            "reaction_time_value", "reaction_time_unit", "reaction_time_scope",
            "kinetic_model", "Hill_K_value", "Hill_K_unit", "Hill_coefficient_n",
            "mediator_role", "enzyme_loading_value", "enzyme_loading_unit",
            "substrate_concentration_raw", "mediator_concentration_raw",
        ]:
            self.assertNotIn(forbidden, projected)
        for required in [
            "primary_dataset_allowed", "record_scope", "rejection_reason",
            "degradation_time_value", "degradation_time_unit",
            "substrate_concentration_value", "substrate_concentration_unit",
            "mediator_name", "mediator_concentration", "mediator_concentration_unit",
            "raw_substrate", "canonical_substrate_name",
        ]:
            self.assertIn(required, V9_SCHEMA_COLUMNS)

    def _classified_gate(self, row):
        return apply_primary_hard_gate(classifier.classify_record(deterministic_record_cleanup(row)))

    def test_plant_extract_cannot_be_primary(self):
        row = self._classified_gate({
            "measurement_type": "degradation",
            "reported_enzyme_name": "ginger juice",
            "substrate": "Aflatoxin B1",
            "degradation_efficiency": 80,
            "degradation_efficiency_unit": "%",
            "notes": "5% ginger juice crude plant extract reduced aflatoxin.",
        })
        self.assertEqual(row["primary_dataset_allowed"], "False")
        self.assertNotEqual(row["record_scope"], "primary_enzyme_record")

    def test_whole_cell_strain_cannot_be_primary(self):
        row = self._classified_gate({
            "measurement_type": "degradation",
            "organism": "Desulfitobacterium sp.",
            "strain": "PGC-3-9",
            "substrate": "Deoxynivalenol",
            "degradation_efficiency": 92,
            "degradation_efficiency_unit": "%",
            "notes": "Whole-cell bacterial strain de-epoxidation under anaerobic condition.",
        })
        self.assertEqual(row["primary_dataset_allowed"], "False")
        self.assertEqual(row["record_scope"], "rejected_out_of_scope")

    def test_cytosol_microsome_cannot_be_primary(self):
        row = self._classified_gate({
            "measurement_type": "kinetic",
            "reported_enzyme_name": "GST activity",
            "substrate": "Aflatoxin B1",
            "Km_value": 12,
            "Km_unit": "µM",
            "notes": "Poultry hepatic cytosolic fraction apparent kinetic parameters; no further enzyme purification was done.",
        })
        self.assertEqual(row["primary_dataset_allowed"], "False")
        self.assertEqual(row["record_scope"], "rejected_out_of_scope")

    def test_inferred_activity_cannot_be_primary_enzyme_name(self):
        row = self._classified_gate({
            "measurement_type": "degradation",
            "reported_enzyme_name": "glucosyltransferase/reductase activity",
            "substrate": "Zearalenone",
            "degradation_efficiency": 75,
            "degradation_efficiency_unit": "%",
            "notes": "Whole-cell/cell membrane biotransformation; product pattern suggests glucosyltransferase/reductase activity.",
        })
        self.assertEqual(row["primary_dataset_allowed"], "False")
        self.assertFalse(row.get("enzyme_name"))
        self.assertFalse(row.get("reported_enzyme_name"))

    def test_purified_recombinant_enzyme_can_be_primary(self):
        row = self._classified_gate({
            "measurement_type": "kinetic",
            "reported_enzyme_name": "Os79",
            "enzyme_name": "Os79 UGT",
            "gene_name": "Os79",
            "enzyme_system_type": "purified_recombinant_enzyme",
            "enzyme_state": "free",
            "substrate": "Deoxynivalenol",
            "Km_value": 50,
            "Km_unit": "µM",
            "kcat_value": 1.5,
            "kcat_unit": "s^-1",
            "evidence_text": "Purified recombinant Os79 kinetic parameters for DON.",
        })
        self.assertEqual(row["primary_dataset_allowed"], "True")
        self.assertEqual(row["record_scope"], "primary_enzyme_record")

    def test_mediator_vs_no_mediator_not_merged(self):
        rows = final_deduplicate_records([
            {
                "pdf_file": "p.pdf",
                "measurement_type": "degradation",
                "reported_enzyme_name": "BsCotA",
                "enzyme_name": "BsCotA laccase",
                "substrate": "Aflatoxin B1",
                "degradation_efficiency": 20,
                "degradation_efficiency_unit": "%",
                "mediator_name": "",
                "notes": "No mediator direct oxidation.",
            },
            {
                "pdf_file": "p.pdf",
                "measurement_type": "degradation",
                "reported_enzyme_name": "BsCotA",
                "enzyme_name": "BsCotA laccase",
                "substrate": "Aflatoxin B1",
                "degradation_efficiency": 100,
                "degradation_efficiency_unit": "%",
                "mediator_name": "methyl syringate",
                "notes": "With methyl syringate mediator.",
            },
        ])
        self.assertEqual(len(rows), 2)

    def test_recovery_cannot_be_direct_degradation(self):
        row = self._classified_gate({
            "measurement_type": "degradation",
            "reported_enzyme_name": "OTA hydrolase",
            "enzyme_name": "OTA hydrolase",
            "enzyme_system_type": "purified_recombinant_enzyme",
            "substrate": "Ochratoxin A",
            "degradation_efficiency": 41,
            "degradation_efficiency_unit": "%",
            "evidence_text": "OTA recovery was 41% after treatment.",
        })
        self.assertEqual(row["primary_dataset_allowed"], "False")
        self.assertIn("metric_semantic_mismatch", row["rejection_reason"])

    def test_auxiliary_kinetic_not_primary(self):
        row = self._classified_gate({
            "measurement_type": "kinetic",
            "reported_enzyme_name": "Lipase",
            "enzyme_name": "lipase",
            "enzyme_system_type": "clearly_identified_commercial_enzyme",
            "substrate": "pNPP",
            "Km_value": 0.62,
            "Km_unit": "mM",
            "evidence_text": "pNPP hydrolysis assay.",
        })
        self.assertEqual(row["primary_dataset_allowed"], "False")
        self.assertIn("rule_1_mycotoxin_substrate", row.get("hard_rule_failures", ""))

    def test_final_primary_export_contains_only_allowed_rows(self):
        rows = [
            self._classified_gate({
                "source_record_id": "ok",
                "measurement_type": "kinetic",
                "reported_enzyme_name": "rPOD2",
                "enzyme_name": "rPOD2 peroxidase",
                "enzyme_system_type": "purified_recombinant_enzyme",
                "substrate": "Aflatoxin B1",
                "Km_value": 1.2,
                "Km_unit": "µM",
                "evidence_text": "Purified recombinant rPOD2 kinetic assay.",
            }),
            self._classified_gate({
                "source_record_id": "bad",
                "measurement_type": "degradation",
                "reported_enzyme_name": "extracellular enzymes",
                "substrate": "Aflatoxin B1",
                "degradation_efficiency": 88,
                "degradation_efficiency_unit": "%",
                "notes": "Aspergillus niger culture supernatant.",
            }),
        ]
        primary = [r for r in rows if r.get("primary_dataset_allowed") == "True"]
        self.assertEqual(len(primary), 1)
        self.assertEqual(primary[0]["source_record_id"], "ok")

    def test_gst_afbo_conjugation_rejected_from_primary(self):
        row = self._classified_gate({
            "measurement_type": "kinetic",
            "reported_enzyme_name": "GST M1-1",
            "enzyme_name": "GST M1-1",
            "substrate": "AFB1 exo-8,9-epoxide",
            "Km_value": 30,
            "Km_unit": "µM",
            "kcat_value": 0.055,
            "kcat_unit": "s^-1",
            "notes": "GST-mediated GSH conjugation of AFBO; metabolic detoxification, not ordinary degradation.",
        })
        self.assertEqual(row["primary_dataset_allowed"], "False")
        self.assertEqual(row["record_scope"], "rejected_out_of_scope")
        self.assertIn("metabolic_activity_not_primary_degrading_enzyme", row["rejection_reason"])

    def test_os79_named_ugt_can_be_primary_without_external_id(self):
        row = self._classified_gate({
            "measurement_type": "kinetic",
            "reported_enzyme_name": "Os79",
            "enzyme_name": "Os79",
            "organism": "Oryza sativa",
            "substrate": "HT-2 toxin",
            "Km_value": 22,
            "Km_unit": "µM",
            "kcat_value": 0.85,
            "kcat_unit": "s^-1",
            "kcat_Km_value": 38600,
            "kcat_Km_unit": "M^-1 s^-1",
            "notes": "Coupled continuous enzymatic assay for purified Os79 UGT.",
        })
        self.assertEqual(row["primary_dataset_allowed"], "True")

    def test_rpod_recombinant_peroxidase_can_be_primary_without_external_id(self):
        row = self._classified_gate({
            "measurement_type": "degradation",
            "reported_enzyme_name": "rPOD2",
            "enzyme_name": "rPOD2",
            "substrate": "Aflatoxin B1",
            "degradation_efficiency": 90,
            "degradation_efficiency_unit": "%",
            "degradation_time_value": 12,
            "degradation_time_unit": "h",
            "notes": "Recombinant peroxidase rPOD2 degraded AFB1 in beer matrix.",
        })
        self.assertEqual(row["primary_dataset_allowed"], "True")

    def test_candida_whole_cell_inferred_activity_rejected(self):
        row = self._classified_gate({
            "measurement_type": "degradation",
            "reported_enzyme_name": "glucosyltransferase/reductase activities",
            "enzyme_name": "glucosyltransferase/reductase activities",
            "organism": "Candida parapsilosis",
            "strain": "ATCC 7330",
            "substrate": "Zearalenone",
            "degradation_efficiency": 80,
            "degradation_efficiency_unit": "%",
            "notes": "Whole-cell/cell-membrane biotransformation; activities inferred from product pattern.",
        })
        self.assertEqual(row["primary_dataset_allowed"], "False")
        self.assertFalse(row.get("enzyme_name"))

    def test_candida_activity_system_label_rejected_even_when_gold_candidate(self):
        row = self._classified_gate({
            "measurement_type": "degradation",
            "reported_enzyme_name": "glucosyltransferase and reductase activities of Candida parapsilosis ATCC 7330",
            "enzyme_name": "glucosyltransferase/reductase system",
            "organism": "Candida parapsilosis",
            "strain": "ATCC 7330",
            "substrate": "Zearalenone",
            "degradation_efficiency": 80,
            "degradation_efficiency_unit": "%",
            "degradation_time_value": 24,
            "degradation_time_unit": "h",
            "notes": "Candida whole-cell/cell membrane associated intracellular activities inferred from product pattern.",
        })
        self.assertEqual(row["primary_dataset_allowed"], "False")
        self.assertIn("inferred_activity_label_not_primary_enzyme", row.get("rejection_reason", ""))
        self.assertFalse(row.get("enzyme_name"))

    def test_kcat_km_repaired_from_kcat_and_um_km(self):
        row = self._classified_gate({
            "measurement_type": "kinetic",
            "reported_enzyme_name": "Q202E",
            "enzyme_name": "Os79",
            "mutations": "Q202E",
            "substrate": "Deoxynivalenol",
            "Km_value": 1072,
            "Km_unit": "µM",
            "kcat_value": 2.4,
            "kcat_unit": "s^-1",
            "kcat_Km_value": 22.4,
            "kcat_Km_unit": "M^-1 s^-1",
            "notes": "Purified recombinant Os79 UGT kinetic assay.",
        })
        self.assertAlmostEqual(float(row["kcat_Km_value"]), 2240, delta=1)
        self.assertIn("kcat_km_repaired_from_kcat_and_km", row.get("error_flags", ""))

    def test_export_time_normalized_to_hours(self):
        row = self._classified_gate({
            "measurement_type": "degradation",
            "reported_enzyme_name": "commercial peroxidase",
            "enzyme_name": "peroxidase",
            "enzyme_system_type": "clearly_identified_commercial_enzyme",
            "substrate": "Aflatoxin B1",
            "degradation_efficiency": 97,
            "degradation_efficiency_unit": "%",
            "degradation_time_value": 480,
            "degradation_time_unit": "min",
            "notes": "Commercial peroxidase degraded AFB1.",
        })
        self.assertEqual(row["primary_dataset_allowed"], "True")
        self.assertEqual(row["degradation_time_unit"], "h")
        self.assertAlmostEqual(float(row["degradation_time_value"]), 8.0)

    def test_reported_range_midpoint_not_exported_as_exact(self):
        row = self._classified_gate({
            "measurement_type": "degradation",
            "reported_enzyme_name": "commercial peroxidase",
            "enzyme_name": "peroxidase",
            "enzyme_system_type": "clearly_identified_commercial_enzyme",
            "substrate": "Aflatoxin B1",
            "degradation_efficiency": 97,
            "degradation_efficiency_unit": "%",
            "degradation_time_value": 8,
            "degradation_time_unit": "h",
            "degradation_temperature_value": 35,
            "degradation_temperature_unit": "°C",
            "degradation_ph": 7.5,
            "notes": "Reaction conditions were pH 7.0-8.0 and 30-40 °C.",
        })
        self.assertEqual(row["primary_dataset_allowed"], "True")
        self.assertFalse(row.get("degradation_temperature_value"))
        self.assertFalse(row.get("degradation_ph"))

    def test_organism_as_enzyme_name_removed(self):
        row = self._classified_gate({
            "measurement_type": "degradation",
            "reported_enzyme_name": "Aspergillus niger",
            "enzyme_name": "Aspergillus niger",
            "organism": "Aspergillus niger",
            "strain": "ND-1",
            "substrate": "Aflatoxin B1",
            "degradation_efficiency": 58.2,
            "degradation_efficiency_unit": "%",
            "notes": "Culture supernatant from Aspergillus niger ND-1; enzyme not purified.",
        })
        self.assertEqual(row["primary_dataset_allowed"], "False")
        self.assertFalse(row.get("enzyme_name"))
        self.assertEqual(row.get("reported_biocatalyst"), "Aspergillus niger")

    def test_phfb1_7_alias_dedup_key(self):
        self.assertEqual(canonical_substrate_name("pHFB1_7"), canonical_substrate_name("pHFB1-7"))

    def test_matrix_context_prevents_false_merge(self):
        rows = final_deduplicate_records([
            {
                "pdf_file": "p.pdf",
                "measurement_type": "degradation",
                "reported_enzyme_name": "FE2",
                "enzyme_name": "FE2",
                "substrate": "Fumonisin B1",
                "matrix": "corn steep liquid",
                "degradation_efficiency": 96.3,
                "degradation_efficiency_unit": "%",
            },
            {
                "pdf_file": "p.pdf",
                "measurement_type": "degradation",
                "reported_enzyme_name": "FE2",
                "enzyme_name": "FE2",
                "substrate": "FB1",
                "matrix": "solid residues",
                "degradation_efficiency": 95.5,
                "degradation_efficiency_unit": "%",
            },
        ])
        self.assertEqual(len(rows), 2)

    def test_equivalent_degradation_time_can_merge(self):
        rows = final_deduplicate_records([
            {
                "pdf_file": "p.pdf",
                "measurement_type": "degradation",
                "reported_enzyme_name": "commercial peroxidase",
                "enzyme_name": "peroxidase",
                "substrate": "Fumonisin B1",
                "degradation_efficiency": 80,
                "degradation_efficiency_unit": "%",
                "degradation_time_value": 8,
                "degradation_time_unit": "h",
            },
            {
                "pdf_file": "p.pdf",
                "measurement_type": "degradation",
                "reported_enzyme_name": "commercial peroxidase",
                "enzyme_name": "peroxidase",
                "substrate": "FB1",
                "degradation_efficiency": 80,
                "degradation_efficiency_unit": "%",
                "degradation_time_value": 480,
                "degradation_time_unit": "min",
            },
        ])
        self.assertEqual(len(rows), 1)

    def test_wild_type_and_named_enzyme_duplicate_merge(self):
        rows = final_deduplicate_records([
            {
                "pdf_file": "p.pdf",
                "measurement_type": "kinetic",
                "reported_enzyme_name": "wild-type",
                "enzyme_name": "Os79",
                "substrate": "HT-2 toxin",
                "Km_value": 22,
                "Km_unit": "µM",
                "kcat_value": 0.85,
                "kcat_unit": "s^-1",
            },
            {
                "pdf_file": "p.pdf",
                "measurement_type": "kinetic",
                "reported_enzyme_name": "Os79",
                "enzyme_name": "Os79",
                "substrate": "HT-2 toxin",
                "Km_value": 22,
                "Km_unit": "µM",
                "kcat_value": 0.85,
                "kcat_unit": "s^-1",
            },
        ])
        self.assertEqual(len(rows), 1)

    def test_matrix_alias_lager_beer_and_beer_merge(self):
        rows = final_deduplicate_records([
            {
                "pdf_file": "p.pdf",
                "measurement_type": "degradation",
                "reported_enzyme_name": "commercial peroxidase",
                "enzyme_name": "peroxidase",
                "substrate": "AFB1",
                "matrix": "lager beer",
                "degradation_efficiency": 24,
                "degradation_efficiency_unit": "%",
                "degradation_time_value": 8,
                "degradation_time_unit": "h",
            },
            {
                "pdf_file": "p.pdf",
                "measurement_type": "degradation",
                "reported_enzyme_name": "commercial peroxidase",
                "enzyme_name": "peroxidase",
                "substrate": "Aflatoxin B1",
                "matrix": "beer",
                "degradation_efficiency": 24,
                "degradation_efficiency_unit": "%",
                "degradation_time_value": 480,
                "degradation_time_unit": "min",
            },
        ])
        self.assertEqual(len(rows), 1)

    def test_matrix_alias_uht_milk_and_milk_merge(self):
        rows = final_deduplicate_records([
            {
                "pdf_file": "p.pdf",
                "measurement_type": "degradation",
                "reported_enzyme_name": "commercial peroxidase",
                "enzyme_name": "peroxidase",
                "substrate": "AFM1",
                "matrix": "UHT milk",
                "degradation_efficiency": 65,
                "degradation_efficiency_unit": "%",
                "degradation_time_value": 8,
                "degradation_time_unit": "h",
            },
            {
                "pdf_file": "p.pdf",
                "measurement_type": "degradation",
                "reported_enzyme_name": "commercial peroxidase",
                "enzyme_name": "peroxidase",
                "substrate": "Aflatoxin M1",
                "matrix": "milk",
                "degradation_efficiency": 65,
                "degradation_efficiency_unit": "%",
                "degradation_time_value": 480,
                "degradation_time_unit": "min",
            },
        ])
        self.assertEqual(len(rows), 1)

    def test_severe_missing_time_blocks_primary(self):
        row = self._classified_gate({
            "measurement_type": "degradation",
            "reported_enzyme_name": "commercial laccase",
            "enzyme_name": "laccase",
            "enzyme_system_type": "clearly_identified_commercial_enzyme",
            "substrate": "Aflatoxin B1",
            "degradation_efficiency": 80,
            "degradation_efficiency_unit": "%",
            "error_flags": "missing_time_for_degradation",
        })
        self.assertEqual(row["primary_dataset_allowed"], "False")
        self.assertIn("missing_time_for_degradation", row["rejection_reason"])

    def test_fill_missing_time_from_equivalent_peer(self):
        rows = fill_missing_degradation_time_from_peer_records([
            {
                "pdf_file": "p.pdf",
                "measurement_type": "degradation",
                "reported_enzyme_name": "commercial laccase",
                "enzyme_name": "laccase",
                "substrate": "Aflatoxin B1",
                "matrix": "beer",
                "degradation_efficiency": 80,
                "degradation_efficiency_unit": "%",
                "error_flags": "missing_time_for_degradation",
            },
            {
                "pdf_file": "p.pdf",
                "measurement_type": "degradation",
                "reported_enzyme_name": "commercial laccase",
                "enzyme_name": "laccase",
                "substrate": "AFB1",
                "matrix": "lager beer",
                "degradation_efficiency": 80,
                "degradation_efficiency_unit": "%",
                "degradation_time_value": 8,
                "degradation_time_unit": "h",
            },
        ])
        self.assertEqual(rows[0]["degradation_time_value"], 8)
        self.assertNotIn("missing_time_for_degradation", rows[0].get("error_flags", ""))

    def test_trichothecene_derivative_whitelist(self):
        for substrate in ["4-ANIV", "4-acetylnivalenol", "Fusarenon-X", "DAS", "IsoT", "4,15-diANIV"]:
            row = self._classified_gate({
                "measurement_type": "kinetic",
                "reported_enzyme_name": "Os79",
                "enzyme_name": "Os79",
                "substrate": substrate,
                "Km_value": 10,
                "Km_unit": "µM",
                "notes": "Purified Os79 enzyme kinetic assay.",
            })
            self.assertNotIn("non_mycotoxin_kinetic_substrate", row.get("rejection_reason", ""))

    def test_wet_milling_incomplete_marked_not_generated(self):
        row = deterministic_record_cleanup({
            "measurement_type": "degradation",
            "reported_enzyme_name": "FE2",
            "enzyme_name": "FE2",
            "substrate": "Fumonisin B1",
            "matrix": "corn flour",
            "degradation_efficiency": 96.3,
            "degradation_efficiency_unit": "%",
            "degradation_time_value": 24,
            "degradation_time_unit": "h",
            "notes": "Wet-milling experiment with corn steep liquid and solid residues contexts.",
        })
        self.assertIn("matrix_context_incomplete", row.get("error_flags", ""))

    def test_matrix_context_incomplete_blocks_primary(self):
        row = self._classified_gate({
            "measurement_type": "degradation",
            "reported_enzyme_name": "FE2",
            "enzyme_name": "FE2",
            "enzyme_system_type": "purified_recombinant_enzyme",
            "substrate": "Fumonisin B1",
            "matrix": "corn flour",
            "degradation_efficiency": 96.3,
            "degradation_efficiency_unit": "%",
            "degradation_time_value": 24,
            "degradation_time_unit": "h",
            "quality_tier": "Gold",
            "error_flags": "matrix_context_incomplete",
            "notes": "Wet-milling experiment mentions corn steep liquid and solid residues.",
        })
        self.assertEqual(row["primary_dataset_allowed"], "False")
        self.assertIn("matrix_context_incomplete", row["rejection_reason"])

    def test_existing_degradation_time_clears_stale_missing_time_flag(self):
        row = deterministic_record_cleanup({
            "measurement_type": "degradation",
            "reported_enzyme_name": "commercial laccase",
            "enzyme_name": "laccase",
            "substrate": "Aflatoxin B1",
            "degradation_efficiency": 80,
            "degradation_efficiency_unit": "%",
            "degradation_time_value": 24,
            "degradation_time_unit": "h",
            "error_flags": "missing_time_for_degradation",
        })
        self.assertNotIn("missing_time_for_degradation", row.get("error_flags", ""))


if __name__ == "__main__":
    unittest.main()
