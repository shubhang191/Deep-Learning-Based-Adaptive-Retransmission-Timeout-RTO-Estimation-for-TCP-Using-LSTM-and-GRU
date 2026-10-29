# Presentation guide: what the repository actually shows

## One-minute project explanation

"TCP uses a retransmission timeout to decide how long to wait for acknowledgements before resending data. An excessively short timeout can trigger redundant retransmission, and an excessively long timeout delays recovery. We used two measured RTT traces, trained LSTM and GRU models on past RTT windows, and compared their next-RTT predictions with Jacobson smoothing and a simple linear AR model. Then we converted predictions into simulated timeout values and compared prediction error, potential early timeouts and excess waiting. The evaluation is offline and does not modify the operating system's TCP stack."

## Who explains which file?

- **Networking and baseline:** `jacobson.py`, RTT/SRTT/RTTVAR/RTO, why ACK timing matters, and the 1-second floor experimental change.
- **Data and training:** `data/*.csv` and `run_experiment.py` `prepare()`, `create_sequences()`, `linear_ar()`, and `nn_predict()`.
- **Evaluation and visualization:** `derive_rto()`, `rto_metrics()`, `bootstrap.csv`, `results_k4.csv` and the graphs.

These are suggested *review responsibilities*, not claims about who personally authored the AI-assisted code.

## Key viva clarifications

**What exactly is predicted?** The next RTT, not packet loss or actual RTO.

**Why 20 points?** The implementation uses the previous 20 RTT measurements as a fixed-length input window; the archive does not contain evidence that 20 is optimal.

**Why two sets of 180 predictions, not 200?** The last 20% is held out, but its first 20 rows provide input history. That leaves 180 prediction targets.

**What is spurious_pct?** Percentage of simulated samples for which predicted RTO is less than the measured RTT — a proxy, not real recorded TCP spurious retransmissions.

**How many networks and models?** Two supplied RTT traces; trained LSTM and GRU, plus linear AR(20) and Jacobson variants. **No Transformer** in the provided experiment code.

**Did you implement Karn?** The helper function exists, but the main experiment does not invoke it; no retransmitted TCP segment traces were provided to evaluate Karn's RTT-ambiguity rule.

**What did AI do?** It is accurate to explain that AI-assisted coding and documentation were used, while your team reviewed and ran the experimental pipeline. Avoid claiming experiments/features not present in the files.
