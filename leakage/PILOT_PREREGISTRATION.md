# Leakage pilot: pre-registration (15 Oct 2026)

Committed **before** any pilot job is submitted. Code: `pilot.py` (build + analyse), tested offline with
`test_pilot_offline.py`.

## Design
- ibm_kingston, one job, `SamplerV2` with `meas_type="kerneled"` (complex IQ per shot and classical bit).
- 6 circuits × 4000 shots: prepare-|0⟩ and prepare-|1⟩ calibration on every qubit used, plus 4 memory arms
  (Z/X × baseline/reset-free), d = 3, 4 rounds. Placement is zero-SWAP, calibration-scored, and excludes
  q146 and qubits with calibrated readout error > 10%.
- Hard bits come from our own discriminator (calibration blobs), not IBM's. Leak flag: Mahalanobis distance
  from the |0⟩–|1⟩ segment above a per-qubit threshold set for a 10⁻³ false-flag rate on calibration shots.

## Predictions and decision rules
- **Safety:** if the mean calibration false-flag rate exceeds 2×10⁻³, the analysis stops and no claim is made.
- **L0 (primary): leakage is detectable.** In the pooled reset-free arms, an ancilla flagged at one
  measurement is flagged again at its next one far more often than chance: ratio ≥ 5, one-sided binomial
  p < 10⁻³, and ≥ 20 repeats. If L0 fails, the result is "leakage not detectable at this sensitivity on this
  chip and day". It is not "no leakage".
- **L1 (secondary, tested only if L0 holds):** the leakage-aware decoder (erasure + neighbours) has fewer
  logical errors than the standard decoder, pooled over the 4 arms, McNemar one-sided p < 0.01.
  **Expected to be underpowered:** in the offline test with 0.5% simulated leakage it gave −1.7% errors at
  p = 0.02. An L1 failure in the pilot is therefore not evidence against the decoder.
- **Exploratory, no verdict:** flag rate by measurement index (does leakage accumulate over rounds?),
  reset-free vs baseline.

## Offline validation (simulation, not hardware)
| simulated truth | L0 | L1 |
|---|---|---|
| no leakage | not detected (0 repeats) | not tested |
| 0.5% leakage | detected (ratio 51, p ≈ 0) | −1.7% errors, p = 0.02 → not significant |

## Budget
QPU cost of a kerneled job of this size has not been measured. The job is submitted only after the owner
decides how to split the remaining IBM open-plan time with the daily study.
