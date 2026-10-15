# Power test of the noise-spectroscopy analysis (SIMULATION ONLY)

**These are simulated qubits with known noise. They are not measurements of any IBM device.**
The test only checks whether the pre-registered analysis (`noisespec.py`) can be trusted before
spending QPU time.

Setup: 120 simulated qubits per chip, T1 150–400 µs, 2% readout error, 1000 shots, the same 7 waits
(2–280 µs) and the same fitting and verdict code as the hardware notebook. 20 independent simulated
chips per scenario. Reproduce with `Qmem4_sim.ipynb` (Colab, no account needed). It ran in Colab
on 15 Oct 2026 in 290 s with the numbers below.

| Simulated truth | M1 echo > Ramsey | M2 CPMG16 > echo | M3 T2 ≤ 2·T1 (benchmark) | M4 Gaussian decay |
|---|---|---|---|---|
| white (Markovian) noise | 0/20 | 0/20 | 20/20 | 0/20 |
| 1/f noise (telegraph fluctuators) | 20/20 | 20/20 | 20/20 | 20/20 |
| mixed | 20/20 | 20/20 | 20/20 | 13/20 |

**Design change made because of this test (before any hardware run):** the first version of M2
(≥ 60% of qubits) passed under pure white noise, a false positive. M1 and M2 now each require a
one-sided sign test p < 0.001 **and** a minimum median ratio (1.2 and 1.1). M4 has low power for mixed
noise, so an M4 FAIL alone is not evidence against slow noise.

**Scope:** a hardware result can show that slow, non-white noise limits coherence. It cannot show that
the noise is magnetic, because two-level defects and charge noise also produce slow noise.
