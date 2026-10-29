# RTT datasets

The two CSV files are the supplied, already-parsed RTT measurements used by `run_experiment.py`.

| Name | Rows | RTT mean (ms) | RTT max (ms) | Schema |
| --- | ---: | ---: | ---: | --- |
| `hotspot.csv` | 998 | 54.03 | 533 | `seq`, `rtt_ms` |
| `fast_hotspot.csv` | 1,000 | 52.98 | 228 | `seq`, `rtt_ms` |

`seq` is the sample index starting at 0; `rtt_ms` is the recorded round-trip time in milliseconds. Both files have sequential indexes and no missing values. The chronological train/test split is applied in code and is not a separate data file.

**Provenance limitation:** the raw terminal output, collection commands, and other collection metadata were not supplied. Do not interpret these CSV files as verified TCP retransmission traces: they represent RTT observations from ping-based measurement.
