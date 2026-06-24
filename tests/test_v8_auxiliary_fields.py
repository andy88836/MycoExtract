import unittest

from src.utils.quality_tier import QualityTierClassifier
from scripts.run_all_papers_full_extraction import build_summary_rows


classifier = QualityTierClassifier()


class V8AuxiliaryFieldTests(unittest.TestCase):
    def test_stability_and_optimum_preserved_in_record(self):
        """Stability/optimum fields are preserved and record is classified."""
        row = classifier.classify_record({
            "measurement_type": "kinetic",
            "enzyme_name": "CotA",
            "substrate": "Aflatoxin B1",
            "Km_value": 0.19,
            "Km_unit": "mM",
            "stability_value": 80,
            "stability_unit": "% residual activity",
            "stability_temperature_value": 50,
            "optimum_ph": 8.0,
            "optimum_temperature_value": 37,
            "notes": "purified recombinant CotA enzyme was used for the kinetic assay",
        })

        # stability/optimum fields are preserved (not deleted)
        self.assertEqual(row["stability_value"], 80)
        self.assertEqual(row["optimum_ph"], 8.0)
        # Record passes classification with 5 field groups
        self.assertIn(row["quality_tier"], ("Gold", "Silver"))
        self.assertGreaterEqual(row["field_group_count"], 5)

    def test_auxiliary_fields_alone_do_not_pass_hard_rules(self):
        """Stability data without a quantitative metric → Rejected (Rule 3 fails)."""
        row = classifier.classify_record({
            "measurement_type": "kinetic",
            "enzyme_name": "CotA",
            "substrate": "Aflatoxin B1",
            "stability_value": 80,
            "stability_unit": "% residual activity",
            "notes": "purified recombinant CotA enzyme was used",
        })

        self.assertEqual(row["quality_tier"], "Rejected")
        self.assertIn("rule_3_quantitative_metric", row["hard_rule_failures"])

    def test_summary_preserves_auxiliary_modalities(self):
        """build_summary_rows still works with classifier-classified records."""
        rows = [
            classifier.classify_record({
                "pdf_file": "paper.pdf",
                "measurement_type": "degradation",
                "enzyme_name": "ZPH1101",
                "substrate": "Zearalenone",
                "degradation_efficiency": 95,
                "stability_note": "retained activity after storage",
            })
        ]
        summary = build_summary_rows(rows)

        self.assertEqual(len(summary), 1)
        self.assertIn(summary[0]["quality_tier"], ("Gold", "Silver", "Bronze", "Rejected"))

    def test_human_review_required_does_not_affect_tier(self):
        """human_review_required is not part of the 5 hard rules or tier calculation."""
        row = classifier.classify_record({
            "measurement_type": "kinetic",
            "enzyme_name": "His6-OPH",
            "substrate": "Patulin",
            "Km_value": 10.9,
            "Km_unit": "mM",
            "kcat_value": 27.1,
            "kcat_unit": "min⁻¹",
            "human_review_required": True,
        })

        self.assertIn(row["quality_tier"], ("Gold", "Silver", "Bronze"))
        self.assertEqual(row["hard_rule_failures"], "")

    def test_error_flags_do_not_affect_tier(self):
        """Error flags are not part of the 5 hard rules or tier calculation."""
        row = classifier.classify_record({
            "measurement_type": "kinetic",
            "enzyme_name": "His6-OPH",
            "substrate": "Patulin",
            "Km_value": 10.9,
            "Km_unit": "mM",
            "error_flags": ["table_multiplier_scaling_error"],
        })

        self.assertIn(row["quality_tier"], ("Gold", "Silver", "Bronze"))
        self.assertEqual(row["hard_rule_failures"], "")


if __name__ == "__main__":
    unittest.main()
