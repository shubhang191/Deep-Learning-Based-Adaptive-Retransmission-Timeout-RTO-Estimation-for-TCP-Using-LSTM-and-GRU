# Archived experimental results

These are the supplied reference outputs produced by an LSTM/GRU run, organized for review. They are tracked in Git so the README charts render without rerunning training.

## metrics/

- `all_runs.csv`: per-session/per-method/per-seed metrics for K ∈ {1,2,3,4,6,8}. **156 rows** in the supplied archive.
- `results_k4.csv`: per-method summary (mean and sample standard deviation across runs as computed by the script) for K = 4.
- `ksweep.csv`: mean metrics grouped by session, method, and K.
- `bootstrap.csv`: moving-block bootstrap interval for *model absolute-error difference minus Jacobson absolute error*. A negative difference indicates lower model MAE.

The code used random seeds **42–46** for LSTM and GRU; the uploaded `all_runs.csv` had them encoded as positions `0`–`4`. The archive in this repository normalizes the `seed` metadata to 42–46 **without modifying the measured metrics**.

## figures/

- `*_fig1_raw.png`: observed RTT traces, with the 80/20 boundary.
- `*_fig2_prediction.png`: one-step RTT prediction versus actual RTT.
- `*_fig3_rto.png`: K=4 timeout curves versus actual RTT.
- `fig4_ksweep_tradeoff.png`: trade-off between timeout proxy and conditional surplus wait.

Note: a point `RTO < RTT` is a **simulated potential timeout**, not an observed retransmission. No Karn or Transformer evaluation appears in the supplied CSVs.

To generate new outputs without overwriting reference files, run the experiment from the project root; its default destination is `outputs/` (ignored by Git).
