# Probe-informed decoding: RESULT (15 Oct 2026) — NOT SUPPORTED

Pre-registered in [PREREGISTRATION.md](PREREGISTRATION.md) (commit 9197722) before the analysis.
Job db88kdgr1c5c738lveeg, 16 arms × 10,000 hardware shots, decoded twice: once with the calibration model,
once with every readout error replaced by the live probe's value.

| | Prediction | Result | Verdict |
|---|---|---|---|
| D1 | probe decoder better in ≥ 12 of 16 arms | 6 of 16 | **FAILS** |
| D2 | pooled McNemar, one-sided p < 0.01 | probe fixes 686 shots, breaks 811; p = 0.999 | **FAILS** |
| D3 | ≥ 2% fewer logical errors | 18,000 → 18,125 (0.69% more) | **FAILS** |

Per-arm numbers are in `result_2026-10-15.json`.

**Post-hoc observation (not a claim):** Z-basis arms had fewer errors in 5 of 8; X-basis arms in 1 of 8.
A plausible, untested reason: each probe readout estimate rests on 3000 shots, so per-qubit readout values
carry about ±0.3% statistical noise, comparable to the drift being corrected. Readout is also only one
part of the error model. Any follow-up must be pre-registered on new data.
