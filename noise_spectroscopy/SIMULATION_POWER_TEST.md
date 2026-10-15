# Power test of the noise-spectroscopy analysis (SIMULATION ONLY)

**These are simulated qubits with known noise. They are not measurements of any IBM device.** The test
only checks whether the pre-registered analysis (`noisespec.py`) can be trusted before spending QPU time.

Setup: 120 simulated qubits per chip, T1 150–400 µs, 1000 shots, the same 7 waits (2–280 µs). Raw
bitstrings go through the same `analyse()` and verdict code as the hardware notebook. 20 independent
simulated chips per scenario. "Confounds" are effects that are not noise but could fake a signal: static
frequency offsets (±5 kHz), static ZZ coupling (≤ 1 kHz to 2 neighbours, not refocused by simultaneous X
pulses), and asymmetric readout (P(1|0) 0.5–2%, P(0|1) 2–6%). Reproduce with `Qmem4_sim_v2.ipynb`
(Colab, no account needed) or `noisesim.py`.

| Simulated truth | M1 echo > Ramsey | M2 CPMG16 > echo | M3 fit sanity (T2 ≤ 2·T1) | M4 β ≥ 1.3 |
|---|---|---|---|---|
| white noise | 0/20 | 0/20 | 20/20 | 0/20 |
| white noise + confounds | 0/20 | 0/20 | 20/20 | 0/20 |
| 1/f noise | 20/20 | 20/20 | 18/20 | 20/20 |
| 1/f noise + confounds | 20/20 | 20/20 | 20/20 | 20/20 |
| mixed | 20/20 | 20/20 | 20/20 | 2/20 |
| mixed + confounds | 20/20 | 20/20 | 20/20 | 3/20 |

**No false positives** for M1, M2 or M4 under white noise, with or without confounds. M1 and M2 detect 1/f
and mixed noise 20/20. M4 has low power for mixed noise, so it is supporting evidence only; an M4 FAIL is
not evidence against slow noise. M3 is a fit sanity check, not a physics test: the simulator cannot
violate T2 ≤ 2·T1.

## Design changes made because of this test (all before any hardware run)
1. The first M2 rule (≥ 60% of qubits) passed under pure white noise. M1 and M2 now each require a
   one-sided sign test p < 0.001 **and** a minimum median ratio (1.2 and 1.1).
2. **Control:** a Ramsey measured only in X gave **10/10 false positives** for M1 under static frequency
   offsets, because echo refocuses the offset and Ramsey does not. Ramsey is now measured in X and Y,
   and the magnitude √(⟨X⟩² + ⟨Y⟩²) is fitted: 0/20.
3. Readout is corrected per qubit with two calibration circuits (prepare |0⟩, prepare |1⟩).

## Scope
A hardware result can show that slow, non-white noise limits coherence. It cannot show the noise is
magnetic: two-level defects, charge noise and flux noise in the tunable couplers all produce slow noise.
β → 2 only for quasi-static noise; 1/f noise gives 1 < β < 2, and T1 pulls β towards 1.
