"""Validate consistency of the checked-in experimental CSVs and images.

This does not retrain the LSTM/GRU models or establish data provenance.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "results" / "metrics"
FIGURES = ROOT / "results" / "figures"



def main():
    all_runs = pd.read_csv(METRICS / "all_runs.csv")
    summary = pd.read_csv(METRICS / "results_k4.csv")
    sweep = pd.read_csv(METRICS / "ksweep.csv")
    boot = pd.read_csv(METRICS / "bootstrap.csv")
    numeric = ["rmse_ms", "mae_ms", "spurious_pct", "wasted_ms", "mean_rto_ms"]
    assert len(all_runs) == 156, f"Unexpected all_runs rows: {len(all_runs)}"
    assert len(summary) == 10 and len(sweep) == 60 and len(boot) == 6
    assert set(all_runs["K"]) == {1, 2, 3, 4, 6, 8}
    assert set(all_runs["session"]) == {"hotspot", "fast_hotspot"}
    assert set(all_runs["method"]) == {"LSTM", "GRU", "Linear AR(20)",
                                      "Jacobson (RFC 6298)", "Jacobson SRTT + error margin"}
    assert set(all_runs.loc[all_runs.method.isin(["LSTM", "GRU"]), "seed"]) == set(range(42, 47))
    grouped_k4 = (all_runs[all_runs.K == 4].groupby(["session", "method"])[numeric].mean().round(3)
                  .sort_index())
    saved_k4 = summary.set_index(["session", "method"])[numeric].sort_index()
    np.testing.assert_allclose(grouped_k4.to_numpy(), saved_k4.to_numpy(), atol=0.00051, rtol=0)
    grouped_sweep = (all_runs.groupby(["session", "method", "K"])[numeric].mean().round(3)
                     .sort_index())
    saved_sweep = sweep.set_index(["session", "method", "K"])[numeric].sort_index()
    np.testing.assert_allclose(grouped_sweep.to_numpy(), saved_sweep.to_numpy(), atol=0.00051, rtol=0)
    for name in ("hotspot", "fast_hotspot"):
        for num, kind in [(1, "raw"), (2, "prediction"), (3, "rto")]:
            assert (FIGURES / f"{name}_fig{num}_{kind}.png").is_file()
    assert (FIGURES / "fig4_ksweep_tradeoff.png").is_file()
    print("PASS: 2 datasets, 156 per-run rows, summaries, K sweep, bootstrap rows and 7 figures.")
    print("NOTE: archived neural runs and data provenance were not independently reproduced.")


if __name__ == "__main__":
    main()
