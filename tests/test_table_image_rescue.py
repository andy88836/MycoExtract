import tempfile
import unittest
from pathlib import Path

from src.extractors.paper_level_extractor import PANDAS_AVAILABLE, PaperLevelMultiModelExtractor


TABLE5_HTML = (
    '<table><tr><td>Enzyme Nanocomplex</td><td>Sample</td><td>Vmax /E0, min−1</td>'
    '<td>Km, mM</td><td>Vmax/(E0 × Km), 105 M−1 min−1</td></tr>'
    '<tr><td colspan="5">Zearalenone</td></tr>'
    '<tr><td rowspan="3">Thermolysin/PLE50</td><td>#1</td><td>1197 ± 175</td><td>3.55 ± 0.25</td><td>3.37 ± 0.73</td></tr>'
    '<tr><td>#2 #3</td><td>1071 ± 105</td><td>3.48 ± 0.40</td><td>3.08 ± 0.65</td></tr>'
    '<tr><td></td><td>1141 ± 78</td><td>3.45 ± 0.27</td><td>3.31 ± 0.49</td></tr>'
    '<tr><td colspan="5">Ochratoxin A</td></tr>'
    '<tr><td rowspan="3">Thermolysin/PLE50</td><td></td><td>507 ± 113</td><td>1.21 ± 0.20</td><td>4.19 ± 1.64</td></tr>'
    '<tr><td></td><td>537 ± 126</td><td>1.23 ± 0.15</td><td>4.38 ± 1.58</td></tr>'
    '<tr><td></td><td>536 ± 109</td><td>1.22 ± 0.11</td><td>4.40 ± 1.29</td></tr>'
    '<tr><td colspan="5">Sterigmatocystin</td></tr>'
    '<tr><td rowspan="3">His6-OPH/PLE50</td><td>#1</td><td>189 ± 14</td><td>0.27 ± 0.03</td><td>7.05 ± 1.31</td></tr>'
    '<tr><td>#2</td><td>261 ± 27</td><td>0.26 ± 0.04</td><td>9.98 ± 2.56</td></tr>'
    '<tr><td>#3</td><td>216 ± 6</td><td>0.22 ± 0.01</td><td>9.72 ± 0.79</td></tr></table>'
)

FREE_ENZYME_COMPLEX_HTML = TABLE5_HTML.replace("Enzyme Nanocomplex", "Enzyme")


class TableImageRescueTests(unittest.TestCase):
    def setUp(self):
        self.extractor = PaperLevelMultiModelExtractor(
            None, None, None, None, None, "", "", ""
        )
        self.block = {
            "type": "table",
            "img_path": "images/table5.jpg",
            "table_caption": [
                "Table 5. Catalytic characteristics of fiber materials # 1, 2, and 3 "
                "modified by thermolysin/PLE50 or His6-OPH/PLE50 in reactions of "
                "mycotoxin hydrolysis in a 50 mM phosphate buffer (pH 7.5)."
            ],
            "table_body": TABLE5_HTML,
        }

    def test_complex_free_enzyme_table_with_image_routes_to_multimodal_rescue(self):
        with tempfile.TemporaryDirectory() as tmp:
            paper_dir = Path(tmp)
            (paper_dir / "images").mkdir()
            (paper_dir / "images" / "table5.jpg").write_bytes(b"fake image")
            block = dict(self.block)
            block["table_caption"] = ["Table 5. Kinetic parameters of purified enzyme against mycotoxins."]
            block["table_body"] = FREE_ENZYME_COMPLEX_HTML

            reason = self.extractor._should_force_multimodal_table_rescue(block, paper_dir)

        self.assertEqual(reason, "complex_merged_cell_mycotoxin_table")

    def test_skip_image_rescue_for_complex_material_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            paper_dir = Path(tmp)
            (paper_dir / "images").mkdir()
            (paper_dir / "images" / "table5.jpg").write_bytes(b"fake image")

            reason = self.extractor._should_force_multimodal_table_rescue(self.block, paper_dir)

        self.assertIsNone(reason)
        self.assertEqual(self.block.get("_table_skip_reason"), "table_skipped_out_of_scope_enzyme_material_system")

    def test_toxicity_endpoint_table_does_not_trigger_image_rescue(self):
        with tempfile.TemporaryDirectory() as tmp:
            paper_dir = Path(tmp)
            (paper_dir / "images").mkdir()
            (paper_dir / "images" / "toxicity.jpg").write_bytes(b"fake image")
            block = {
                "type": "table",
                "img_path": "images/toxicity.jpg",
                "table_caption": ["Table 3. Ecotoxicity of reaction media after enzymatic treatment."],
                "table_body": (
                    '<table><tr><td rowspan="2">Mycotoxin</td><td>Enzymatic Treatment</td>'
                    '<td>Residual Bioluminescence, %</td></tr>'
                    '<tr><td>His6-OPH</td><td>99</td></tr></table>'
                ),
            }

            reason = self.extractor._should_force_multimodal_table_rescue(block, paper_dir)

        self.assertIsNone(reason)

    def test_specific_activity_table_skipped_before_text_or_vision_extraction(self):
        block = {
            "type": "table",
            "table_caption": [
                "Table 2. Apparent specific activities "
                "$\\left( \\mathsf { { \\mu m o l } } \\mathsf { m i n } ^ { - 1 } "
                "\\mathsf { m } \\mathsf { g } ^ { - 1 } \\right)$ "
                "of purified wheat glutathione transferases."
            ],
            "table_body": (
                '<table><tr><td>Candidate GST</td><td>DON</td><td>CDNB</td></tr>'
                '<tr><td>TaGST-02</td><td>1.17 10-5</td><td>0.74</td></tr>'
                '<tr><td colspan="3">Apparent specific activity</td></tr></table>'
            ),
        }

        use_text_only, reason = self.extractor._should_use_text_only_extraction(block)

        self.assertIsNone(use_text_only)
        self.assertEqual(reason, "table_skipped_metric_out_of_scope_specific_activity")
        self.assertTrue(block.get("_skip_for_text_extraction"))

    def test_specific_activity_table_skips_image_rescue_even_when_complex(self):
        with tempfile.TemporaryDirectory() as tmp:
            paper_dir = Path(tmp)
            (paper_dir / "images").mkdir()
            (paper_dir / "images" / "activity.jpg").write_bytes(b"fake image")
            block = {
                "type": "table",
                "img_path": "images/activity.jpg",
                "table_caption": [
                    "Table 2. Apparent specific activities with DON and model substrates."
                ],
                "table_body": (
                    '<table><tr><td rowspan="2">Candidate GST</td>'
                    '<td colspan="6">Apparent specific activity (µmol min-1 mg-1)</td></tr>'
                    '<tr><td>DON</td><td>CDNB</td><td>EPNP</td><td>PEITC</td></tr>'
                    '<tr><td>TaGST-02</td><td>1.17 10-5</td><td>0.74</td><td>nd</td><td>4.9</td></tr></table>'
                ),
            }

            reason = self.extractor._should_force_multimodal_table_rescue(block, paper_dir)

        self.assertIsNone(reason)
        self.assertEqual(block.get("_table_skip_reason"), "table_skipped_metric_out_of_scope_specific_activity")

    def test_real_mycotoxin_kinetic_table_is_not_skipped_by_activity_gate(self):
        block = {
            "type": "table",
            "table_caption": ["Table 1. Kinetic parameters of purified His6-OPH with mycotoxins."],
            "table_body": (
                '<table><tr><td>Substrate</td><td>Km, mM</td><td>kcat, min-1</td></tr>'
                '<tr><td>Patulin</td><td>10.9</td><td>27.1</td></tr></table>'
            ),
        }

        self.assertIsNone(self.extractor._table_metric_scope_skip_reason(block))

    @unittest.skipUnless(PANDAS_AVAILABLE, "pandas is required for complex HTML fallback")
    def test_complex_table_fallback_expands_nine_contexts(self):
        records = self.extractor._extract_complex_kinetic_table_from_html(self.block, 5)

        self.assertEqual(len(records), 9)
        pairs = {(r["reported_enzyme_name"], r["substrate"]) for r in records}
        self.assertIn(("Thermolysin/PLE50", "Zearalenone"), pairs)
        self.assertIn(("Thermolysin/PLE50", "Ochratoxin A"), pairs)
        self.assertIn(("His6-OPH/PLE50", "Sterigmatocystin"), pairs)
        # Multiplier is NOT applied during extraction — stored as metadata
        self.assertEqual(records[0]["kcat_Km_value"], 3.37)
        self.assertIn("#1", records[0]["measurement_context_id"])
        self.assertIn("#2", records[1]["measurement_context_id"])
        self.assertIn("#3", records[2]["measurement_context_id"])
        self.assertEqual(records[0]["source_section"], "table_5")
        self.assertTrue(all(r["human_review_required"] for r in records))

    @unittest.skipUnless(PANDAS_AVAILABLE, "pandas is required for complex HTML fallback")
    def test_table_rescue_records_are_preserved_after_aggregation(self):
        rescue_records = self.extractor._extract_complex_kinetic_table_from_html(self.block, 5)
        aggregated_records = [
            {
                "reported_enzyme_name": "His6-OPH",
                "substrate": "Patulin",
                "measurement_type": "kinetic",
                "Km_value": 10.9,
                "kcat_value": 27.1,
                "kcat_Km_value": 2480.0,
            }
        ]

        preserved = self.extractor._preserve_table_rescue_records(aggregated_records, rescue_records)

        self.assertEqual(len(preserved), 10)
        self.assertTrue(any(
            record.get("reported_enzyme_name") == "Thermolysin/PLE50"
            and record.get("substrate") == "Ochratoxin A"
            for record in preserved
        ))
        self.assertTrue(all(
            record.get("human_review_required")
            for record in preserved
            if "table_image_rescue" in (record.get("error_flags") or [])
        ))


if __name__ == "__main__":
    unittest.main()
