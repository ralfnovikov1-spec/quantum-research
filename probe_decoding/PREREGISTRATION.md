# Probe-informed decoding: pre-registration (15 Oct 2026)

Committed **before** the analysis below is run. The git commit time is the pre-registration.

## Idea
The 15 Oct hardware data showed that detectors which fire more often than the calibration model predicts
sit on qubits whose readout error drifted since IBM's calibration (exploratory; C3 in the daily study).
The decoder's error model uses IBM's calibrated readout errors. If that is the mechanism, a decoder built
from the **live probe's** readout errors should correct more errors on the **same hardware shots**. That
would be a software-only improvement with no extra QPU time.

## Data (already collected, not yet decoded this way)
Job `db88kdgr1c5c738lveeg` (ibm_kingston, 15 Oct 2026, 16 logical arms × 10,000 shots) and its probe
`db88jo6r6oqs7388vrs0`. Only the decoder's error model changes; the shots are identical.

## Method (fixed in advance)
- For every arm, take the stored Stim model (calibration noise). Replace each measured qubit's readout
  error p with the probe's readout error, keeping the same split: X_ERROR(p/2) before the measurement and
  a record flip M(p/2). Everything else stays the same.
- Decode the same detector data with PyMatching, using (a) the calibration model and (b) the
  probe-readout model.
- Per arm, compare logical error. Pool over all 16 arms with a paired exact sign test on shots where
  exactly one decoder is wrong (McNemar).

## Predictions
- **D1:** the probe-readout decoder has lower logical error in at least 12 of 16 arms.
- **D2:** pooled McNemar test, probe decoder better, one-sided p < 0.01.
- **D3 (effect size):** pooled relative reduction in logical-error count of at least 2%.

If any of D1–D3 fails, it is reported as failed. A success here is one day on one chip. It becomes a claim
only if it repeats prospectively in the daily study, where it would be added as a pre-registered claim
before any further data.
