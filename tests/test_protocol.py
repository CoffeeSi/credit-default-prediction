"""Meaningful data-leakage invariants; run: python -m unittest discover -s tests."""
import unittest

import numpy as np

from src.preprocessing import CATEGORICAL, FEATURES, NUMERIC, _read_data, check_splits, load_partitions, load_splits, make_preprocessor, signed_log_numeric


class ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parts = load_partitions()
        cls.train_x, cls.train_y = cls.parts["train"]

    def test_complete_disjoint_grouped_split(self):
        audit = check_splits(_read_data(), load_splits())
        self.assertEqual(audit["feature_group_overlap"], 0)
        self.assertEqual(audit["n_rows"], 30000)
        for name, expected in [("train", .70), ("validation", .15), ("test", .15)]:
            self.assertLess(abs(audit["splits"][name]["fraction"] - expected), .01)

    def test_training_interface_excludes_test_id_target(self):
        self.assertEqual(set(self.parts), {"train", "validation"})
        self.assertEqual(list(self.train_x.columns), FEATURES)
        self.assertNotIn("ID", self.train_x)
        self.assertNotIn("default", self.train_x)

    def test_scaler_fits_only_training_values(self):
        prep = make_preprocessor()
        x_train = prep.fit_transform(self.train_x)
        scaler = prep.named_steps["columns"].named_transformers_["numeric"].named_steps["scale"]
        np.testing.assert_allclose(scaler.mean_, signed_log_numeric(self.train_x[NUMERIC]).mean(axis=0))
        original_mean = scaler.mean_.copy()
        x_val = self.parts["validation"][0].copy()
        x_val["LIMIT_BAL"] = 1e15
        x_val["SEX"] = 999  # Unseen category must not fit new encoder columns.
        val_out = prep.transform(x_val)
        np.testing.assert_array_equal(scaler.mean_, original_mean)
        self.assertEqual(x_train.dtype, np.float32)
        self.assertEqual(x_train.shape[1], val_out.shape[1])
        self.assertTrue(np.isfinite(val_out).all())
        encoder = prep.named_steps["columns"].named_transformers_["categorical"]
        self.assertNotIn(999, encoder.categories_[CATEGORICAL.index("SEX")])


if __name__ == "__main__":
    unittest.main()
