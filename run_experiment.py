"""
run_experiment.py
=================
Final experiment for Project #10 (adaptive retransmission timeout).

Runs everything on BOTH real ping sessions:
  * Jacobson (RFC 6298)                      - the classical baseline
  * Jacobson SRTT + error margin             - same RTT estimate as Jacobson, but with
                                               the SAME margin rule as the neural models
                                               (isolates RTT-estimate quality from margin style)
  * Linear AR(20)                            - a simple linear model on the same window
                                               (does a neural net beat a straight line?)
  * LSTM, GRU                                - only if TensorFlow is installed
                                               (5 seeds, mean +- std)

Outputs (in --outdir):  results_k4.csv, ksweep.csv, bootstrap.csv, and figures.

Usage:
    python run_experiment.py --hotspot data/hotspot.csv --fast data/fast_hotspot.csv
    python run_experiment.py --hotspot data/hotspot.csv --fast data/fast_hotspot.csv --no-nn   # skip LSTM/GRU
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge

from jacobson import jacobson_rto

KS = [1, 2, 3, 4, 6, 8]
SEEDS = [42, 43, 44, 45, 46]
WINDOW = 20
TRAIN_FRAC = 0.8
EWMA_BETA = 0.25


# ----------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------
def create_sequences(series, w):
    X = np.array([series[i:i + w] for i in range(len(series) - w)], dtype=np.float32)
    y = np.array([series[i + w] for i in range(len(series) - w)], dtype=np.float32)
    return X.reshape(-1, w, 1), y.reshape(-1, 1)


def prepare(csv_path):
    rtt = pd.read_csv(csv_path)["rtt_ms"].to_numpy(float)
    n = len(rtt)
    split = int(n * TRAIN_FRAC)
    # chronological split: test is strictly later than train
    train, test = rtt[:split].reshape(-1, 1), rtt[split:].reshape(-1, 1)
    scaler = MinMaxScaler().fit(train)                     # fitted on TRAIN ONLY
    Xtr, ytr = create_sequences(scaler.transform(train), WINDOW)
    Xte, yte = create_sequences(scaler.transform(test), WINDOW)
    return dict(
        rtt=rtt, split=split, scaler=scaler,
        Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte,
        actual=scaler.inverse_transform(yte).ravel(),
        idx=np.arange(split + WINDOW, n),                  # index of each test target
        # causal starting value for the error margin: mean |RTT_t - RTT_t-1| on TRAIN only
        init_err=float(np.mean(np.abs(np.diff(rtt[:split])))),
    )


# ----------------------------------------------------------------------------
# RTT -> RTO and metrics
# ----------------------------------------------------------------------------
def derive_rto(pred, actual, k, init_err):
    """RTO_t = pred_t + k * EWMA(|error| from steps strictly before t)."""
    rto = np.zeros(len(pred))
    e = init_err
    for t in range(len(pred)):
        rto[t] = pred[t] + k * e
        e = (1 - EWMA_BETA) * e + EWMA_BETA * abs(pred[t] - actual[t])
    return rto


def rto_metrics(actual, rto):
    low = rto < actual
    slack = (rto - actual)[~low]
    return dict(spurious_pct=100.0 * low.mean(),
                wasted_ms=float(slack.mean()) if slack.size else 0.0,
                mean_rto_ms=float(rto.mean()))


def rtt_metrics(actual, pred):
    return dict(rmse_ms=float(np.sqrt(np.mean((actual - pred) ** 2))),
                mae_ms=float(np.mean(np.abs(actual - pred))))


# ----------------------------------------------------------------------------
# predictors
# ----------------------------------------------------------------------------
def linear_ar(d):
    m = Ridge(alpha=1e-3).fit(d["Xtr"].reshape(len(d["Xtr"]), -1), d["ytr"].ravel())
    p = m.predict(d["Xte"].reshape(len(d["Xte"]), -1)).reshape(-1, 1)
    return d["scaler"].inverse_transform(p).ravel()


def build_nn(kind):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Input
    from tensorflow.keras.optimizers import Adam
    layer = {"lstm": LSTM, "gru": GRU}[kind]
    m = Sequential([Input(shape=(WINDOW, 1)), layer(48), Dropout(0.10),
                    Dense(16, activation="relu"), Dense(1)])
    m.compile(optimizer=Adam(1e-3), loss="mse")
    return m


def nn_predict(kind, d, seed):
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping
    tf.keras.utils.set_random_seed(seed)
    m = build_nn(kind)
    m.fit(d["Xtr"], d["ytr"], epochs=40, batch_size=32, validation_split=0.15,
          shuffle=False, verbose=0,
          callbacks=[EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)])
    p = m.predict(d["Xte"], verbose=0)
    return d["scaler"].inverse_transform(p).ravel()


# ----------------------------------------------------------------------------
# bootstrap (paired, moving-block) on the absolute-error difference
# ----------------------------------------------------------------------------
def block_bootstrap_ci(diff, block=10, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(diff)
    nb = int(np.ceil(n / block))
    stats = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n - block + 1, nb)
        sample = np.concatenate([diff[s:s + block] for s in starts])[:n]
        stats[b] = sample.mean()
    return float(diff.mean()), float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


# ----------------------------------------------------------------------------
def run_session(name, csv_path, use_nn, outdir):
    d = prepare(csv_path)
    actual, idx = d["actual"], d["idx"]
    print(f"\n[{name}] samples={len(d['rtt'])}  train windows={len(d['Xtr'])}  test windows={len(actual)}")

    jac = {k: jacobson_rto(d["rtt"], k=k) for k in KS}
    jac_srtt = jac[4]["srtt"][idx - 1]            # estimate after sample t-1 predicts sample t

    # predictions per method: list over seeds, each an RTT prediction in ms
    preds = {"Jacobson SRTT": [jac_srtt], "Linear AR(20)": [linear_ar(d)]}
    if use_nn:
        for kind in ("lstm", "gru"):
            preds[kind.upper()] = []
            for s in SEEDS:
                print(f"  training {kind.upper()} seed {s} ...", flush=True)
                preds[kind.upper()].append(nn_predict(kind, d, s))

    rows = []
    for k in KS:                                   # Jacobson with its own native margin
        rows.append(dict(session=name, method="Jacobson (RFC 6298)", seed=0, K=k,
                         **rtt_metrics(actual, jac_srtt),
                         **rto_metrics(actual, jac[k]["rto"][idx - 1])))
    for method, plist in preds.items():
        label = "Jacobson SRTT + error margin" if method == "Jacobson SRTT" else method
        for si, p in enumerate(plist):
            for k in KS:
                rows.append(dict(session=name, method=label, seed=SEEDS[si] if method in ("LSTM", "GRU") else 0, K=k,
                                 **rtt_metrics(actual, p),
                                 **rto_metrics(actual, derive_rto(p, actual, k, d["init_err"]))))
    df = pd.DataFrame(rows)

    # bootstrap on MAE difference vs Jacobson (seed-averaged abs error per test point)
    boot = []
    base_err = np.abs(actual - jac_srtt)
    for method, plist in preds.items():
        if method == "Jacobson SRTT":
            continue
        err = np.mean([np.abs(actual - p) for p in plist], axis=0)
        mean_d, lo, hi = block_bootstrap_ci(err - base_err)
        verdict = "inconclusive (CI contains 0)" if lo <= 0 <= hi else ("better" if hi < 0 else "worse")
        boot.append(dict(session=name, method=method, mae_diff_vs_jacobson_ms=mean_d,
                         ci_low=lo, ci_high=hi, verdict=verdict))

    # ---- figures ----
    show = slice(0, len(actual))
    colours = {"LSTM": "#1f77b4", "GRU": "#2ca02c", "Linear AR(20)": "#9467bd"}
    plt.figure(figsize=(12, 4))
    plt.plot(d["rtt"], lw=0.7, color="#3b6ea5")
    plt.axvline(d["split"], color="crimson", ls="--", label="train/test split")
    plt.title(f"{name}: measured RTT"); plt.xlabel("packet"); plt.ylabel("RTT (ms)")
    plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(outdir, f"{name}_fig1_raw.png"), dpi=130); plt.close()

    plt.figure(figsize=(12, 4.5))
    plt.plot(actual[show], color="black", lw=1.0, alpha=0.7, label="Actual RTT")
    plt.plot(jac_srtt[show], color="crimson", ls="--", lw=1.3, label="Jacobson SRTT")
    for m in [x for x in preds if x in colours]:
        plt.plot(preds[m][0][show], color=colours[m], lw=1.2, label=f"{m} (seed 42)" if m in ("LSTM", "GRU") else m)
    plt.title(f"{name}: RTT prediction on held-out test data"); plt.xlabel("test sample"); plt.ylabel("RTT (ms)")
    plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(outdir, f"{name}_fig2_prediction.png"), dpi=130); plt.close()

    plt.figure(figsize=(12, 4.5))
    plt.plot(actual[show], color="black", lw=1.0, alpha=0.7, label="Actual RTT")
    plt.plot(jac[4]["rto"][idx - 1][show], color="crimson", ls="--", lw=1.3, label="Jacobson RTO (K=4)")
    for m in [x for x in preds if x in ("LSTM", "GRU")]:
        plt.plot(derive_rto(preds[m][0], actual, 4, d["init_err"])[show], color=colours[m], lw=1.2, label=f"{m} RTO (K=4)")
    plt.title(f"{name}: retransmission timeout vs actual RTT"); plt.xlabel("test sample"); plt.ylabel("ms")
    plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(outdir, f"{name}_fig3_rto.png"), dpi=130); plt.close()

    return df, pd.DataFrame(boot)


def ksweep_figure(df, outdir):
    sessions = df["session"].unique()
    fig, axes = plt.subplots(1, len(sessions), figsize=(6.2 * len(sessions), 4.6))
    axes = np.atleast_1d(axes)
    for ax, s in zip(axes, sessions):
        for method, g in df[df.session == s].groupby("method"):
            m = g.groupby("K")[["spurious_pct", "wasted_ms"]].mean().sort_index()
            ax.plot(m["wasted_ms"], m["spurious_pct"], marker="o", label=method)
            for k, r in m.iterrows():
                ax.annotate(str(k), (r["wasted_ms"], r["spurious_pct"]), fontsize=7,
                            xytext=(3, 3), textcoords="offset points")
        ax.set_title(f"{s}: trade-off as K varies (labels = K)")
        ax.set_xlabel("wasted waiting (ms)  -> worse"); ax.set_ylabel("spurious retransmissions (%)  -> worse")
        ax.legend(fontsize=7)
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "fig4_ksweep_tradeoff.png"), dpi=130); plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hotspot", default="hotspot.csv")
    ap.add_argument("--fast", default="fast_hotspot.csv")
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--no-nn", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    use_nn = not a.no_nn
    if use_nn:
        try:
            import tensorflow as tf
            tf.config.experimental.enable_op_determinism()
        except ImportError:
            print("TensorFlow not installed: running without LSTM/GRU (pip install tensorflow to add them).")
            use_nn = False

    dfs, boots = [], []
    for name, path in (("hotspot", a.hotspot), ("fast_hotspot", a.fast)):
        df, b = run_session(name, path, use_nn, a.outdir)
        dfs.append(df); boots.append(b)
    df, boot = pd.concat(dfs), pd.concat(boots)

    df.to_csv(os.path.join(a.outdir, "all_runs.csv"), index=False)
    boot.to_csv(os.path.join(a.outdir, "bootstrap.csv"), index=False)

    metrics = ["rmse_ms", "mae_ms", "spurious_pct", "wasted_ms", "mean_rto_ms"]
    g = df[df.K == 4].groupby(["session", "method"])[metrics]
    summary = g.mean().round(3)
    if use_nn:
        std = g.std().fillna(0).round(3).add_suffix("_std")
        summary = summary.join(std)
    summary.to_csv(os.path.join(a.outdir, "results_k4.csv"))
    df.groupby(["session", "method", "K"])[metrics].mean().round(3).to_csv(os.path.join(a.outdir, "ksweep.csv"))
    ksweep_figure(df, a.outdir)

    pd.set_option("display.width", 200)
    print("\n=== RESULTS at K=4 (mean over seeds where applicable) ===")
    print(summary.to_string())
    print("\n=== Block bootstrap: MAE difference vs Jacobson (negative = better) ===")
    print(boot.round(3).to_string(index=False))
    print(f"\nFiles written to {a.outdir}/")


if __name__ == "__main__":
    main()
