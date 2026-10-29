# Methodology and code walkthrough

This page describes the **actual uploaded scripts**; it does not assume features present only in presentation slides.

## 1. Read and split data

`prepare()` reads `rtt_ms` in chronological order and splits at `int(0.8 * n)`. The `MinMaxScaler` is fitted only on the training portion and then used to transform both partitions. This prevents using future test-set maxima/minima during training.

The `create_sequences()` function maps each 20-value window to the *next* RTT. Windows are built separately within train and test, yielding 180 evaluated test targets in each dataset. For the first test prediction, only earlier test rows form its 20-value history.

## 2. Predict future RTT

All values are treated as milliseconds after inverse scaling.

- **Linear AR(20):** `sklearn.linear_model.Ridge(alpha=1e-3)` fit to flattened windows of 20 samples.
- **LSTM:** Keras Sequential model with input `(20, 1)`, 48 recurrent units, dropout 0.10, dense 16 (ReLU), dense 1.
- **GRU:** same architecture except using a 48-unit GRU layer.
- **Jacobson SRTT:** exponential smoothing with `alpha = 1/8`. The script deliberately uses the smoothing value **after sample t−1** to predict sample t (one-step lag prevents lookahead).

Each neural model is trained with `Adam(1e-3)`, MSE loss, up to 40 epochs, batch size 32, 15% validation split, early-stopping patience 8, `shuffle=False`, and five random seeds (42–46).

## 3. Compute retransmission timeouts

The first Jacobson baseline uses these RFC-style updates (all quantities in ms):

```text
RTTVAR ← (1 − 1/4) × RTTVAR + (1/4) × |SRTT_old − new_RTT|
SRTT   ← (1 − 1/8) × SRTT   + (1/8) × new_RTT
RTO    ← SRTT + max(G, K × RTTVAR), G = 1 ms
```

The one-second lower bound specified in RFC 6298 is **disabled** for this experimental baseline (default `min_rto = 0`). The code does not implement timestamp-based ambiguity handling in TCP.

For all learned predictors, the linear AR(20) predictor, and the second Jacobson comparison, the rule is:

```text
RTO[t]   = predicted_RTT[t] + K × prior_EWMA_error[t]
new_EWMA = 0.75 × prior_EWMA_error[t] + 0.25 × |predicted_RTT[t] − actual_RTT[t]|
```

The error used in the RTO for sample t is from **strictly previous samples**; after evaluating sample t, its observed error updates the state for sample t+1. The EWMA starts from mean absolute adjacent differences in training RTT.

## 4. Evaluate

- **MAE:** mean absolute difference between the one-step RTT estimate and actual RTT.
- **RMSE:** square root of mean squared error.
- **Spurious percentage (proxy):** percentage of cases `RTO < RTT`. It is not an actual TCP retransmission rate.
- **Wasted waiting (proxy):** mean `(RTO − RTT)` only among cases with `RTO ≥ RTT`. This is not an end-to-end throughput result.
- **K sweep:** K = 1, 2, 3, 4, 6, 8.
- **Bootstrap:** paired moving blocks of length 10, with 2,000 resamples. The recorded statistic is model absolute error minus Jacobson absolute error, averaged across seeds per test point for NN models.

## 5. Generate outputs

```bash
python run_experiment.py --hotspot data/hotspot.csv --fast data/fast_hotspot.csv
```

The CLI writes `all_runs.csv`, `results_k4.csv`, `ksweep.csv`, `bootstrap.csv`, and seven PNG figures. Use `--no-nn` to skip training deep-learning models (that run will not contain LSTM/GRU outputs).

## 6. What was intentionally *not* implemented

- Transformer architecture / training.
- Full Karn-style RTT ambiguity exclusion from real retransmitted TCP segments.
- Applying the optional `karn_backoff()` helper inside the main simulation.
- Controlling a live TCP socket's retransmission timer, or measuring actual throughput.
- Missing-network-feature engineering (congestion window, loss signals, queue delay).
