# Reset-free syndrome extraction on IBM heavy-hex hardware

Hardware study of the IBM heavy-hex subsystem code (distance 3) on 156-qubit Heron processors
(`ibm_kingston`, `ibm_marrakesh`). The question: does removing mid-circuit resets from syndrome
extraction lower the logical error per round?

**Provenance.** Every hardware error rate, gain and lifetime here is re-derived from IBM's stored raw
results by independent code: [`04_hardware_runs/verify_hw.py`](04_hardware_runs/verify_hw.py) produces
[`INDEPENDENT_REDERIVATION.json`](04_hardware_runs/INDEPENDENT_REDERIVATION.json) and covers all 9 jobs.
The verdict texts of the 14 Oct predictions P1–P16 are quoted from the original Colab notebook, which is
not published. Every simulation number is labelled SIMULATION.

## Results (hardware, d = 3)

| Claim | Evidence | Status |
|---|---|---|
| Removing resets lowers Z-basis logical error per round | ×1.70, ×1.66, ×2.07, ×1.72 in four `ibm_kingston` runs, each 2σ-separated. The last run was pre-registered at 10,000 shots. Run 3 also changed the placement. | **Supported** |
| The same holds in the X basis | ×1.13–1.42 on `ibm_kingston` (flag resets kept) | **Supported, smaller effect** |
| The same holds on a second chip | ×1.07 / ×1.05 on `ibm_marrakesh`, inside 2σ | **Not supported** (direction only) |
| Break-even (logical outlives best physical qubit) | Logical/physical lifetime 0.092× (Z) and 0.136× (X), pre-registered run. Lifetimes are defined below. | **Not reached** |
| A live probe of every qubit makes the noise model accurate | Probe-fed model within ±30% in 4/4 arms; calibration-fed model 1.5–2.1× optimistic. The probe prediction was decoded with its own noise model, not the calibration decoder applied to the hardware data; fixed for the daily study (protocol v2). | **Supported, with that caveat** (one day, one chip) |
| Detectors that err more than the calibration model sit on qubits whose readout drifted since calibration | 23 qubit sets: ρ = +0.54, 95% CI [+0.17, +0.80], permutation p = 0.009. No detectable association with T1 drift (ρ = +0.16, 95% CI [−0.28, +0.49]). | **Exploratory** (one day, analysis chosen after the data); pre-registered as C3 in the daily study |
| Larger codes help (Λ > 1) | Only d = 3 was run on hardware | **Not tested** |

**Lifetime definitions.** The logical lifetime τ_L comes from 1 − 2p_L(t) = A·exp(−t/τ_L), fitted over
rounds 1–4. Physical references were the best 3 usable qubits, idled with the same X–X dynamical
decoupling for 7–400 µs. Z references start in |1⟩ and X references in |+⟩. For the X–X sequence used,
the decay of 1 − 2p is close to exponential, and the symmetric fit recovers T1 to within 5%. That was
checked against the exact pulse-sequence model in
[`recompute_breakeven.py`](04_hardware_runs/recompute_breakeven.py). With the exact model, the best Z
reference has T1 = 543 µs [522, 565], so the Z ratio is 0.092×; the pre-registered code reported 0.096×.
Either way, the pre-registered band check P19 (0.10–0.35) fails.

The full list of 21 pre-registered predictions, including the 8 that failed, is in the
[corrected report](01_reports/heavy-hex_report_CORRECTED_2026-10-15.pdf).

**Prior work, and what is not new here.** Removing ancilla resets from heavy-hex syndrome extraction on IBM
Heron hardware was published first by Harper et al., *Characterising the failure mechanisms of
error-corrected quantum logic gates*, arXiv:2504.07258 (April 2025), *Nature Communications* 17, 5039
(2026). They used ibm_marrakesh at d = 3, a lower-depth circuit (3.2 µs per round) and reported 96–97%
per-round fidelity. Gehér et al., *To reset, or not to reset* (arXiv:2408.00758), analysed reset-free
extraction in simulation. IBM's 2022 heavy-hex experiment is arXiv:2203.07205.

This repository is an **independent replication** on another chip (`ibm_kingston`) with IBM's original
flag-qubit circuit, plus a pre-registered rerun, raw-data re-derivation and a live-probe analysis. Noise-aware
decoding with live noise estimates is also established (e.g. arXiv:2502.21044, arXiv:2406.02700).
The per-detector readout-drift association (C3) was not found in a targeted literature search, but no
full review has been done.

## Live daily study (automated)

A GitHub Action ([`.github/workflows/daily.yml`](.github/workflows/daily.yml)) runs every day at 06:17 UTC.
It probes the chip of the day, runs baseline and reset-free arms on IBM hardware, decodes the previous
day's job, and updates [`daily/RESULTS.md`](daily/RESULTS.md). Claims C1 (calibration staleness), C2
(reset-free gain) and C3 (detector excess tracks readout drift) are pre-registered in
[`daily/PROTOCOL.md`](daily/PROTOCOL.md). They are decided only after 20 analysed days.

## Noise spectroscopy (prepared, not yet run on hardware)

[`noise_spectroscopy/`](noise_spectroscopy/) measures T1, Ramsey, echo and CPMG-4/16 decays on every
usable qubit to test whether slow (non-white) noise limits coherence. So far only a **simulation**
power test exists ([`SIMULATION_POWER_TEST.md`](noise_spectroscopy/SIMULATION_POWER_TEST.md)). It found
and fixed two false-positive routes before any hardware run. The experiment cannot identify the
microscopic origin of the noise, magnetic or otherwise.

## Layout

| Path | Content |
|---|---|
| `01_reports/` | Corrected report (PDF + HTML). It supersedes an earlier version that had post-hoc and unlabelled failures. |
| `03_code_hhmem/hhmem.py` | Heavy-hex subsystem code builder, GF(2) detector repair, Stim noise model, zero-SWAP placement, X–X DD, decoding |
| `03_code_hhmem/hhmem2.py` | Live probe, dead-qubit filter (applied before selection), pre-registration with timestamp and SHA-256, verdicts |
| `03_code_hhmem/dry_run.py` | End-to-end test on `FakeKingston` with an injected dead qubit; includes a control run with the filter disabled |
| `03_code_hhmem/Qmem3.ipynb` | Colab notebook used for the 15 Oct 2026 pre-registered rerun |
| `04_hardware_runs/` | Plans (circuits, detectors, layouts) for every job, probe data, the rerun's pre-registration and verdicts, `verify_hw.py`, `recompute_breakeven.py`, `detector_drift.py` |
| `05_audit/` | Audit of the author's earlier exploratory notebooks (not published here), with recomputation scripts |
| `daily/` | Daily study: protocol, code, data, results |
| `noise_spectroscopy/` | Noise-spectroscopy analysis, simulator, simulation power test, Colab notebooks |

## Reproduce

```bash
pip install "qiskit>=1.3" qiskit-ibm-runtime stim pymatching rustworkx scipy
python 03_code_hhmem/dry_run.py                      # offline, FakeKingston
python daily/daily.py fake                           # offline daily cycle
export IBM_QUANTUM_KEY=...                           # your own key; never commit it
python 04_hardware_runs/verify_hw.py 04_hardware_runs/2026-10-14_plans 04_hardware_runs
```

Tested with Python 3.11 and 3.13, qiskit 2.5.2, qiskit-ibm-runtime 0.50.0, stim 1.16.0 and pymatching 2.4.0.

## Limitations
One day per chip so far. Distance 3 only. A matching decoder only. Physical-qubit references are the
best 3 usable qubits by live probe. IBM Quantum open plan (600 s QPU per month) bounds the sample size.
