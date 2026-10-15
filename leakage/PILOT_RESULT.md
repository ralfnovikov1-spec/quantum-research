# Leakage pilot: RESULT (15 Oct 2026, ibm_kingston)

Pre-registered in [PILOT_PREREGISTRATION.md](PILOT_PREREGISTRATION.md) (commit 3996ef3, 12:11 UTC) before
submission. Job `db8clieil99s73fpo0eg`, run by GitHub Actions run 38736429306. 6 circuits × 4000 shots with
kerneled IQ readout, 23 qubits. QPU time: 9 s (IBM-reported). Raw IQ, plan and analysis are in
`pilot_results/20261015T124446Z_*`.

## Pre-registered verdicts
| | Rule | Result | Verdict |
|---|---|---|---|
| Safety | calibration false-flag rate ≤ 2×10⁻³ | 1.0×10⁻³ | passed |
| **L0** | persistent flags in reset-free arms: ratio ≥ 5, p < 10⁻³, ≥ 20 repeats | ratio 105, 113 repeats, p ≈ 5×10⁻¹⁸⁶ | **HOLDS** |
| **L1** | leakage-aware decoder (erasure + neighbours) better, McNemar p < 0.01 | 3338 → 3394 logical errors (+1.7%); fixes 70, breaks 126; p ≈ 1.0 | **FAILS** |

**What L0 means:** in all 4 arms, readout points far off the |0⟩–|1⟩ axis recur on the same ancilla at its
next measurement about 100× more often than chance. That is consistent with leakage to |2⟩. Because |2⟩ cannot
be prepared on Heron for calibration, this is **not a measured leakage rate**. The flag rate was 0.0015–0.0016
per measurement against a 0.0010 false-flag rate, so only about 0.06% of measurements carry the extra
persistent signal.

**What L1 means:** most flags are false (rate 0.1% against about 0.06% excess). Reacting to them costs more than
the rare real events save. This is the false-flag cost seen in simulation.

## Exploratory analyses (post hoc; NOT claims)
- `pilot_explore.py`: persistent flags occur on 12 of 14 ancilla qubits. q67 and q71 hold 49% of the repeats.
  q71's flagged points form one coherent off-axis cluster (direction coherence 0.97, about 6.6σ from |1⟩). The
  others are mixed (coherence 0.3–0.8).
- Flag rate rises with measurement index in 3 of 4 arms (about 0.0014 → 0.0020–0.0025). This would be
  consistent with accumulation, but is untested.
- `pilot_explore_persistent.py`: acting only on flags seen at two consecutive measurements removes the harm
  (3338 → 3335, −0.09%, p = 0.37). It gives no gain: only about 1% of shots contain a persistent flag.

## Conclusion
On ibm_kingston at d = 3 and 4 rounds, leakage-consistent events are real (L0) but rare. Leakage-aware
decoding gives no benefit at this scale (L1 failed; the post-hoc persistent-flag variant is neutral). If
leakage accumulates, it would matter in **longer** experiments (many rounds). Testing that would need a new
pre-registration and more QPU time.
