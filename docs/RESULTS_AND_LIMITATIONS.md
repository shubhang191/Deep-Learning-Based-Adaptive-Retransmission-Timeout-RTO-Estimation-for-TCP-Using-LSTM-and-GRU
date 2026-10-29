# Results and interpretation

All numbers below come from the **supplied** `results/metrics/` CSVs. The scripts' classical and linear-model baselines were independently rerun and matched the archived K=4 values. The neural models were **not retrained during packaging**.

## Quantitative outcomes, K=4

| Dataset | Method | RTT MAE (ms) | RTT RMSE (ms) | Timeout proxy (%) | Wasted wait proxy (ms) |
| --- | --- | ---: | ---: | ---: | ---: |
| Hotspot | Jacobson (RFC formula) | 20.314 | 42.461 | 3.333 | 90.183 |
| Hotspot | Jacobson SRTT + error margin | 20.314 | 42.461 | 3.333 | 89.366 |
| Hotspot | Linear AR(20) | 20.323 | 42.951 | 2.778 | 87.939 |
| Hotspot | LSTM | 17.732 | 41.217 | 2.778 | 75.102 |
| Hotspot | GRU | 17.552 | 41.332 | 3.111 | 73.807 |
| Fast hotspot | Jacobson (RFC formula) | 11.666 | 21.786 | 1.667 | 46.779 |
| Fast hotspot | Jacobson SRTT + error margin | 11.666 | 21.786 | 1.667 | 47.029 |
| Fast hotspot | Linear AR(20) | 11.136 | 22.150 | 2.778 | 44.813 |
| Fast hotspot | LSTM | 10.828 | 21.409 | 2.556 | 43.475 |
| Fast hotspot | GRU | 10.644 | 21.388 | 2.556 | 42.270 |

### Interpretation

- On the noisier hotspot trace, GRU's RTT MAE is about **13.6% lower** than Jacobson's (`20.314` → `17.552` ms). At K=4 its conditional excess wait proxy is about **18.2% lower** (`90.183` → `73.807` ms). LSTM's corresponding MAE improvement is about **12.7%**.
- On the fast-hotspot trace, GRU's RTT MAE is about **8.8% lower** (`11.666` → `10.644` ms), but its early-timeout proxy is **higher** (2.556% vs 1.667%).
- Results are trade-offs; neither model dominates across *all* metrics and settings. High RTT spikes remain difficult to predict from historical values.
- In the supplied moving-block bootstrap, the 95% interval for LSTM's MAE difference on the fast-hotspot session crosses zero; GRU's barely excludes zero. These intervals are specific to the supplied traces and simulation.

## Figures

| Figure | Question |
| --- | --- |
| [`hotspot_fig1_raw.png`](../results/figures/hotspot_fig1_raw.png) | Where are spikes and which samples are held out? |
| [`hotspot_fig2_prediction.png`](../results/figures/hotspot_fig2_prediction.png) | How do one-step RTT estimates compare with actual RTT? |
| [`hotspot_fig3_rto.png`](../results/figures/hotspot_fig3_rto.png) | How does a safety margin change timeout behavior? |
| [`fast_hotspot_fig1_raw.png`](../results/figures/fast_hotspot_fig1_raw.png) | How volatile is the second trace? |
| [`fast_hotspot_fig2_prediction.png`](../results/figures/fast_hotspot_fig2_prediction.png) | Does each estimator react to spikes? |
| [`fast_hotspot_fig3_rto.png`](../results/figures/fast_hotspot_fig3_rto.png) | Does the timeout recover quickly after spikes? |
| [`fig4_ksweep_tradeoff.png`](../results/figures/fig4_ksweep_tradeoff.png) | Does lower false-timeout proxy require more waiting? |

## Practical limitations and cautions

1. RTT data are from **ping**, so early RTO expiry is a **proxy**, not observed TCP loss/retransmissions.
2. There are **only two** datasets (998 and 1,000 rows; 180 evaluated test targets each).
3. The **RFC 6298 one-second RTO floor** is disabled; do not claim wire-protocol RFC conformance.
4. **Karn exponential backoff** exists as a standalone helper but is not used in these measured comparisons. No Transformer is present in the scripts.
5. `wasted_ms` computes **conditional** excess wait on samples with RTO ≥ actual RTT; it is not the average of all `max(RTO−RTT,0)` values.
6. Results are provided in the archive; for full neural reproducibility a compatible TensorFlow environment is needed. We only verified classical/linear baselines in the packaging environment.
7. Data provenance includes only processed RTT CSV files; no raw ping logs, full collection commands or independent capture metadata were supplied.
8. Any older slide deck/report making claims about Transformer, Jacobson+Karn results, different sample sizes, actual throughput or deployed TCP needs reconciliation before use.
