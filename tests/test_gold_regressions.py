import unittest
import asyncio

from src.agents.aggregation_agent import AggregationAgent
from src.extractors.paper_level_extractor import PaperLevelMultiModelExtractor
from src.pipeline.enhanced_pipeline import EnhancedExtractionPipeline
from src.pipeline.paper_level_prechecker import PaperLevelPrechecker
from src.pipeline.post_processor import RecordMerger, infer_measurement_type, normalize_records_batch
from src.utils.row_level_validator import RowLevelValidator
from src.utils.sequence_enricher import SequenceEnricher, SequenceCandidate
from src.utils.table_multiplier import apply_kinetic_unit_multiplier
try:
    from src.utils.zero_record_rescue import deterministic_table_fallback, has_rescue_keyword_evidence
except ModuleNotFoundError:
    deterministic_table_fallback = None
    has_rescue_keyword_evidence = None


class GoldDatasetRegressionTests(unittest.TestCase):
    class _DummyLLM:
        model_name = "dummy"

        def chat(self, *args, **kwargs):
            return "[]"

    def test_record_merger_does_not_merge_kinetic_and_degradation_contexts(self):
        records = normalize_records_batch([
            {
                "enzyme_name": "CotA",
                "substrate": "Aflatoxin B1",
                "Km_value": 191,
                "temperature_value": 37,
                "ph": 8,
            },
            {
                "enzyme_name": "CotA",
                "substrate": "Aflatoxin B1",
                "degradation_efficiency": 0.51,
                "reaction_time_value": 12,
                "reaction_time_unit": "h",
                "temperature_value": 37,
                "ph": 8,
            },
        ])

        merged = RecordMerger().merge_records(records)

        self.assertEqual(len(merged), 2)
        self.assertEqual({r["measurement_type"] for r in merged}, {"kinetic", "degradation"})

    def test_sequence_enrichment_blocks_commercial_and_abbreviation_cases(self):
        enricher = SequenceEnricher(fetch_sequences=False, fetch_smiles=False)
        rows, _ = enricher.enrich_records([
            {"enzyme_name": "PO", "enzyme_state": "commercial", "substrate": "Ochratoxin A"},
            {"enzyme_name": "PPL", "enzyme_state": "commercial", "substrate": "Ochratoxin A"},
            {"enzyme_name": "ALA", "enzyme_state": "commercial", "substrate": "Zearalenone"},
        ])

        for row in rows:
            self.assertIsNone(row.get("uniprot_id"))
            self.assertEqual(row["enrichment_status"], "blocked_due_to_commercial_or_crude_source")

    def test_sequence_enrichment_does_not_overwrite_main_fields_with_candidate(self):
        enricher = SequenceEnricher(fetch_sequences=False, fetch_smiles=False)
        enricher._query_uniprot = lambda *args, **kwargs: [
            SequenceCandidate("P12345", "Candidate protein", "Candidate organism", "geneX", 100, True, 0.99)
        ]

        rows, _ = enricher.enrich_records([
            {"enzyme_name": "Clearase", "organism": "Full organism", "substrate": "Aflatoxin B1"}
        ])
        row = rows[0]

        self.assertIsNone(row.get("uniprot_id"))
        self.assertIsNone(row.get("gene_name"))
        self.assertEqual(row["candidate_uniprot_id"], "P12345")
        self.assertEqual(row["enrichment_status"], "matched_exact_reported_name_and_organism")

    def test_organism_abbreviation_blocks_enrichment(self):
        enricher = SequenceEnricher(fetch_sequences=False, fetch_smiles=False)
        rows, _ = enricher.enrich_records([
            {"enzyme_name": "TRI101", "organism": "F. graminearum", "substrate": "Deoxynivalenol"}
        ])

        self.assertEqual(rows[0]["enrichment_status"], "blocked_due_to_abbreviation_ambiguity")
        self.assertEqual(rows[0]["organism"], "F. graminearum")

    def test_table_multiplier_scaling_examples(self):
        examples = [
            (2.48, 2480.0),
            (0.25, 250.0),
            (2.68, 2680.0),
            (15.8, 15800.0),
        ]

        for raw, expected in examples:
            record = {
                "kcat_Km_value": raw,
                "kinetic_unit_source_text": "Vmax/(E0 x Km) (x10^3 M^-1 min^-1)",
            }
            scaled = apply_kinetic_unit_multiplier(record)
            self.assertEqual(scaled["kcat_Km_value"], expected)
            self.assertEqual(scaled["kinetic_unit_multiplier"], 1000.0)

        already_scaled = {
            "kcat_Km_value": 2480.0,
            "kinetic_unit_multiplier": 1000.0,
            "kinetic_unit_source_text": "x10^3 M^-1 min^-1",
        }
        self.assertEqual(apply_kinetic_unit_multiplier(already_scaled)["kcat_Km_value"], 2480.0)

        upstream_marked_applied_but_value_is_raw = {
            "kcat_Km_value": 2.48,
            "kcat_Km_unit": "10³ M⁻¹ min⁻¹",
            "kinetic_unit_multiplier": 1000,
            "kinetic_unit_source_text": "Table header: 10³ M⁻¹ min⁻¹",
            "_table_multiplier_applied": True,
        }
        scaled = apply_kinetic_unit_multiplier(upstream_marked_applied_but_value_is_raw)
        # When _table_multiplier_applied is set, trust it — never re-scale.
        # This prevents double-scaling bugs (e.g. 10.1021_acs.biochem.7b01007).
        self.assertEqual(scaled["kcat_Km_value"], 2.48)
        self.assertEqual(scaled["kcat_Km_unit"], "M⁻¹ min⁻¹")

        upstream_marked_applied_and_value_is_scaled = {
            "kcat_Km_value": 2480.0,
            "kcat_Km_unit": "10³ M⁻¹ min⁻¹",
            "kinetic_unit_multiplier": 1000,
            "_table_multiplier_applied": True,
        }
        self.assertEqual(
            apply_kinetic_unit_multiplier(upstream_marked_applied_and_value_is_scaled)["kcat_Km_value"],
            2480.0,
        )

        upstream_scaled_small_value = {
            "kcat_Km_value": 250.0,
            "kcat_Km_unit": "10³ M⁻¹ min⁻¹",
            "kinetic_unit_multiplier": 1000,
            "_table_multiplier_applied": True,
        }
        self.assertEqual(
            apply_kinetic_unit_multiplier(upstream_scaled_small_value)["kcat_Km_value"],
            250.0,
        )

    def test_text_extraction_helpers_do_not_reference_missing_block_variable(self):
        extractor = PaperLevelMultiModelExtractor(
            kimi_client=None,
            deepseek_client=None,
            glm47_client=None,
            glm46v_client=None,
            aggregation_client=None,
            text_prompt_template="",
            table_prompt_template="",
            figure_prompt_template="",
        )
        model = self._DummyLLM()

        full_text_records = asyncio.run(
            extractor._extract_full_paper_text(model, "This article has no extractable records.", "dummy")
        )
        block_text_records = asyncio.run(
            extractor._extract_text_block(model, "No table record here.", 1, "dummy")
        )

        self.assertEqual(full_text_records, [])
        self.assertEqual(block_text_records, [])

    def test_table_text_only_missing_fields_does_not_raise_name_error(self):
        extractor = PaperLevelMultiModelExtractor(
            kimi_client=None,
            deepseek_client=self._DummyLLM(),
            glm47_client=None,
            glm46v_client=None,
            aggregation_client=None,
            text_prompt_template="",
            table_prompt_template="",
            figure_prompt_template="",
        )

        records = asyncio.run(extractor._extract_table_text_only({"type": "table"}, 1))
        self.assertEqual(records, [])

    def test_aggregation_parser_accepts_json_with_surrounding_text(self):
        agent = AggregationAgent(llm_client=None)
        response = "Here are the records:\n```json\n[{\"enzyme_name\":\"His6-OPH\",\"substrate\":\"Patulin\"}]\n```\nPlease verify."
        parsed = agent._parse_response(response)
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0]["substrate"], "Patulin")

    def test_ambiguous_power_ten_header_routes_to_vision(self):
        extractor = PaperLevelMultiModelExtractor(
            kimi_client=None,
            deepseek_client=None,
            glm47_client=None,
            glm46v_client=None,
            aggregation_client=None,
            text_prompt_template="",
            table_prompt_template="",
            figure_prompt_template="",
        )
        block = {
            "table_body": (
                "<table><tr><td>Mycotoxin</td><td>Vmax /Eo, min-1</td>"
                "<td>Km, mM</td><td>Vmax/(Eo x Km),10 M-1 min-1</td></tr>"
                "<tr><td>Patulin</td><td>27.1</td><td>10.9</td><td>2.48</td></tr></table>"
            )
        }
        use_text_only, reason = extractor._should_use_text_only_extraction(block)
        self.assertFalse(use_text_only)
        self.assertEqual(reason, "ambiguous_power_ten_multiplier_header")

        fallback_records = extractor._extract_simple_kinetic_table_from_html(block, 1)
        self.assertEqual(len(fallback_records), 1)
        self.assertEqual(fallback_records[0]["substrate"], "Patulin")
        # Multiplier is NOT applied during extraction — stored as metadata
        # for apply_kinetic_unit_multiplier to handle during normalization.
        self.assertEqual(fallback_records[0]["kcat_Km_value"], 2.48)
        self.assertEqual(fallback_records[0]["kinetic_unit_multiplier"], 1000.0)

        non_mycotoxin_block = {
            "table_body": (
                "<table><tr><td>Sample</td><td>Vmax /Eo, sec-1</td>"
                "<td>Km, uM</td><td>Vmax/(Eo x Km),105 M-1 sec-1</td></tr>"
                "<tr><td>Paraoxon</td><td>142</td><td>324</td><td>4.37</td></tr></table>"
            )
        }
        self.assertEqual(extractor._extract_simple_kinetic_table_from_html(non_mycotoxin_block, 2), [])

    def test_review_article_excluded_from_primary_dataset(self):
        prechecker = PaperLevelPrechecker()
        document_type = prechecker.classify_document_type(
            "Review: recent advances in mycotoxin detoxification enzymes\n"
            "This article summarizes previous studies and reported enzymes."
        )
        self.assertEqual(document_type, "review")

        records = [
            {"enzyme_name": "ZENA", "source_type": "review", "measurement_type": "review_background"},
            {"enzyme_name": "CotA", "source_type": "original", "measurement_type": "kinetic"},
        ]
        filtered = EnhancedExtractionPipeline._filter_primary_dataset_records(object(), records)
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["enzyme_name"], "CotA")

    def test_row_validator_flags_over_normalized_abbreviations_and_review_leakage(self):
        ppl = RowLevelValidator.validate_record({
            "enzyme_name": "PPL",
            "enzyme_full_name": "Periplakin",
            "uniprot_id": "F1RK90",
            "enzyme_state": "commercial",
            "substrate": "Ochratoxin A",
            "Km_value": 330,
        })
        self.assertIn("enzyme_name_over_normalization", ppl["error_flags"])
        self.assertIn("enrichment_cascade_error", ppl["error_flags"])

        review = RowLevelValidator.validate_record({
            "enzyme_name": "ZHD101",
            "substrate": "Zearalenone",
            "kcat_Km_value": 14800,
            "source_type": "review",
        })
        self.assertIn("review_article_leakage", review["error_flags"])
        self.assertEqual(review["QC_Status"], "exclude_review_article")

    def test_conflicting_source_statements_infer_ambiguous_measurement_type(self):
        record = {
            "enzyme_name": "rPOD2",
            "substrate": "Aflatoxin M1",
            "degradation_efficiency": 73.9,
            "notes": "conflicting statements: abstract optimum pH differs from optimized degradation pH",
        }

        self.assertEqual(infer_measurement_type(record), "ambiguous_conflicting_source")

    def test_zero_record_rescue_kinetic_trichothecene_fixture(self):
        if deterministic_table_fallback is None or has_rescue_keyword_evidence is None:
            self.skipTest("zero_record_rescue module removed in v8")
        text = "kinetic analysis of trichothecene mycotoxins using Os79 with Km, kcat, DON"
        self.assertTrue(has_rescue_keyword_evidence(text))
        records = deterministic_table_fallback([
            {
                "type": "table",
                "table_caption": ["Steady-state kinetic constants for Os79 with trichothecene substrates"],
                "table_body": (
                    "<table><tr><td>enzyme</td><td>substrate</td><td>kcat (s-1)</td>"
                    "<td>Km (uM)</td><td>kcat/Km (s-1 M-1)</td></tr>"
                    "<tr><td>Os79</td><td>DON</td><td>1.07</td><td>61</td><td>1.75 x 10^4</td></tr></table>"
                ),
            }
        ])
        self.assertGreater(len(records), 0)
        self.assertEqual(records[0]["reported_enzyme_name"], "Os79")
        self.assertEqual(records[0]["substrate"], "Deoxynivalenol")

    def test_zero_record_rescue_tri101_transformation_fixture(self):
        if deterministic_table_fallback is None:
            self.skipTest("zero_record_rescue module removed in v8")
        records = deterministic_table_fallback([
            {
                "type": "table",
                "table_caption": ["TRI101/TRI201 acetyltransferase modification of DON to 3ADON; catalytic efficiency"],
                "table_body": (
                    "<table><tr><td>enzyme</td><td>substrate</td><td>product</td>"
                    "<td>kcat/Km (M-1 s-1)</td></tr>"
                    "<tr><td>TRI101</td><td>DON</td><td>3ADON</td><td>12000</td></tr></table>"
                ),
            }
        ])
        self.assertGreater(len(records), 0)
        self.assertEqual(records[0]["reported_enzyme_name"], "TRI101")
        self.assertEqual(records[0]["measurement_type"], "kinetic")

    def test_zero_record_rescue_microsomal_fraction_fixture(self):
        if deterministic_table_fallback is None:
            self.skipTest("zero_record_rescue module removed in v8")
        records = deterministic_table_fallback([
            {
                "type": "table",
                "table_caption": ["Kinetic parameters for ZEA biotransformation by microsomal fraction: pH and cofactor effects"],
                "table_body": (
                    "<table><tr><td>enzyme</td><td>substrate</td><td>product</td>"
                    "<td>Vmax (pmol/mg/min)</td><td>Km (uM)</td></tr>"
                    "<tr><td>microsomal fraction</td><td>ZEA</td><td>α-ZOL</td><td>953</td><td>769</td></tr></table>"
                ),
            }
        ])
        self.assertGreater(len(records), 0)
        self.assertEqual(records[0]["enzyme_system_type"], "subcellular_fraction")
        self.assertFalse(records[0]["identified_enzyme"])
        self.assertEqual(records[0]["enrichment_status"], "blocked_due_to_unidentified_enzyme")

    def test_zero_record_rescue_adh3_ota_fixture(self):
        if deterministic_table_fallback is None:
            self.skipTest("zero_record_rescue module removed in v8")
        records = deterministic_table_fallback([
            {
                "type": "table",
                "table_caption": ["ADH3 OTA hydrolase kinetic/degradation record"],
                "table_body": (
                    "<table><tr><td>enzyme</td><td>substrate</td><td>kcat/Km (M-1 s-1)</td>"
                    "<td>conversion (%)</td></tr>"
                    "<tr><td>ADH3</td><td>OTA</td><td>35000</td><td>100</td></tr></table>"
                ),
            }
        ])
        self.assertGreater(len(records), 0)
        self.assertEqual(records[0]["reported_enzyme_name"], "ADH3")
        self.assertEqual(records[0]["substrate"], "Ochratoxin A")


if __name__ == "__main__":
    unittest.main()
