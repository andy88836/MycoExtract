import unittest

from src.utils.quality_constraints import QualityConstraintFilter


class QualityConstraintNoneSafeTests(unittest.TestCase):
    def test_none_substrate_returns_false_not_exception(self):
        qc = QualityConstraintFilter(require_mycotoxin=True, strict_mode=True)
        ok, reason = qc._check_mycotoxin_substrate({"substrate": None})
        self.assertFalse(ok)
        self.assertIn("No substrate", reason)


if __name__ == "__main__":
    unittest.main()
