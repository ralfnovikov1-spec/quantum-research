# Daily study protocol, version 1 (pre-registered 15 Oct 2026)

This file is committed **before the first data point**. The git commit timestamp and hash are the
pre-registration. Changing the analysis requires bumping `PROTOCOL_VERSION` in `daily.py` and adding a
dated entry to the change log below. Days collected under an earlier version are reported separately.

## Question
Does IBM's published calibration make logical-error predictions systematically optimistic, and does a
seconds-long live probe of every qubit fix it? Secondary: does reset-free syndrome extraction keep
lowering the Z-basis error per round across days and chips?

Motivation (disclosed): one day of data (15 Oct 2026, ibm_kingston) showed calibration-fed predictions
1.5–2.1× too optimistic and probe-fed predictions within ±30%. An exploratory per-detector analysis of the
same day (`04_hardware_runs/detector_drift.py`) found the excess tracked readout drift (23 qubit sets,
ρ = +0.54, permutation p = 0.009) but not T1 drift (ρ = +0.16, p = 0.46). That day is **not** counted
below. C3 was defined after seeing it, which is why it is tested only on new days.

## Pre-registered claims (decided after ≥ 20 analysed days, not before)
- **C1.** On at least 80% of analysed days, the calibration-fed model's mean relative prediction error is at
  least 2× the probe-fed model's.
- **C2.** On at least 80% of analysed days, the reset-free Z arm's error per round is lower than baseline,
  with 2σ intervals separated.

- **C3.** On at least 60% of analysed days, the per-qubit-set mean detector excess (hardware detection
  rate minus calibration-model rate) correlates positively with the probe-measured readout drift
  (live − calibration) of those qubits: Spearman ρ > 0 with permutation p < 0.05. The bar is 60%, not 80%,
  because one day has only about 20–30 qubit sets, which limits per-day power.

Each claim gets HOLDS or FAILS. If a claim fails, it is reported as failed.

## Fixed design
- **Chip:** `ibm_kingston` and `ibm_marrakesh` alternate by day (`date.toordinal() % 2`). Queue-induced
  delays are allowed; one job per chip-day.
- **Probe:** 3 circuits × 3000 shots on all qubits. Readout, T1 from decay over 20 µs, and echo T2.
- **Unusable qubit:** mean readout > 10%, either one-sided readout > 15%, or T1 not identifiable or > 2 ms.
  Excluded before placement.
- **Placement:** zero-SWAP, calibration-scored, top-8 by Stim simulation (`hhmem2.place_filtered`).
- **Arms:** heavy-hex subsystem code, d = 3. Baseline and reset-free (`COMPRESS`). Z and X bases. Rounds
  1, 2, 3. Alternating arm order. X–X DD. 4000 shots per circuit.
- **Metric:** ε per round from a weighted fit of 1 − 2p(r) = A(1 − 2ε)^r, with plain MWPM decoding (PyMatching
  on the arm's own Stim model).
- **Model error** = |ε_pred − ε_hw| / ε_hw, averaged over the 4 arms. Predictions are computed and stored
  before submission. The calibration model uses IBM backend properties at submission; the probe model uses
  probe T1, T2 and readout plus calibration gate errors.
- **Budget guard:** skip the day if IBM reports < 60 s of QPU time remaining. Skipped and failed days are
  listed and never counted.

## Known limitations (stated in advance)
Open-plan budget (600 s/month) limits days per month. Matching decoder only. d = 3 only. "Days" are
not independent of IBM's recalibration schedule, which is not controlled.

## Change log
- v1, 15 Oct 2026: initial protocol (C1, C2).
- v1, 15 Oct 2026, before the first data point: added C3 and the per-day probe and calibration snapshot.
  No daily data existed when this was committed.
