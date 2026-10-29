import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from run_experiment import create_sequences, derive_rto, prepare, rto_metrics, rtt_metrics

ROOT = Path(__file__).resolve().parents[1]


class ExperimentTests(unittest.TestCase):
    def test_sequences_predict_the_next_sample(self):
        X, y = create_sequences(np.arange(6, dtype=float).reshape(-1, 1), 3)
        self.assertEqual(X.shape, (3, 3, 1))
        self.assertEqual(y.shape, (3, 1))
        np.testing.assert_allclose(X[0].ravel(), [0, 1, 2])
        self.assertEqual(y[0, 0], 3)

    def test_rto_margin_uses_previous_error(self):
        pred = np.array([30.0, 30.0])
        actual = np.array([40.0, 20.0])
        # t=0: 30+4*5; after observing t=0, EWMA=0.75*5+0.25*10=6.25
        np.testing.assert_allclose(derive_rto(pred, actual, k=4, init_err=5), [50, 55])

    def test_metric_meanings(self):
        metrics = rto_metrics(np.array([40.0, 50.0]), np.array([30.0, 80.0]))
        self.assertAlmostEqual(metrics["spurious_pct"], 50)
        self.assertAlmostEqual(metrics["wasted_ms"], 30)
        r = rtt_metrics(np.array([1.0, 3.0]), np.array([2.0, 1.0]))
        self.assertAlmostEqual(r["mae_ms"], 1.5)

    def test_supplied_dataset_shapes(self):
        for f, expected_n, expected_split in [
            ("hotspot.csv", 998, 798),
            ("fast_hotspot.csv", 1000, 800),
        ]:
            d = prepare(ROOT / "data" / f)
            self.assertEqual(len(d["rtt"]), expected_n)
            self.assertEqual(d["split"], expected_split)
            self.assertEqual(len(d["actual"]), 180)
            self.assertEqual(int(d["idx"][0]), expected_split + 20)

    def test_supplied_data_columns(self):
        for f in ("hotspot.csv", "fast_hotspot.csv"):
            df = pd.read_csv(ROOT / "data" / f)
            self.assertEqual(list(df.columns), ["seq", "rtt_ms"])
            self.assertTrue(df["rtt_ms"].notna().all())
            self.assertTrue(np.array_equal(df["seq"].to_numpy(), np.arange(len(df))))


if __name__ == "__main__":
    unittest.main()
