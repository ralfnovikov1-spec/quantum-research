# Leakage-aware decoding for reset-free heavy-hex (work in progress — SIMULATION ONLY so far)

**Question:** on IBM Heron, a "reset" is a measurement plus a conditional X, so neither the baseline nor the
reset-free schedule clears a leaked (|2⟩) ancilla. Can leaked ancillas be seen in the analog (kerneled IQ)
readout, and can a decoder that knows about them lower the logical error?

**Prior work:** Varbanov et al., npj Quantum Information 2020 (HMM leakage detection with analog readout,
surface-code simulation; leakage-aware decoding proposed, not demonstrated). Leakage on IBM Heron has been
diagnosed (arXiv:2606.00433), but no rate was given. A targeted search found no hardware demonstration of
leakage-aware decoding on the heavy-hex code. That is not a full review.

| File | What it does |
|---|---|
| `leaksim.py` | Simple model: leaked ancilla readouts only |
| `leaksim2.py` | Realistic model (stim.FlipSimulator): persistent leakage; a leaked qubit gives its CZ partner a Z with probability 1/2. Three decoders: hard, readout-erasure, erasure + neighbours |
| `hwpipeline.py` | Hardware pipeline: Stim-record ↔ classical-bit mapping; |0⟩/|1⟩ blob calibration; leak flag = Mahalanobis distance from the |0⟩–|1⟩ segment, which tolerates T1 decay during readout; threshold set for a 10⁻³ false-flag rate; persistence test (binomial) |
| `test_pipeline_offline.py` | End-to-end test with messy synthetic IQ: random blob geometry, 3% decay tails, off-axis |2⟩ |

**Simulation findings so far (not hardware):**
- With neighbour damage, readout-only erasure makes the logical error **worse**; only erasure + neighbours helps.
- With realistic readout geometry, 63% of leaks are caught. At 1% leakage per measurement, erasure + neighbours
  lowers logical errors by about 1% on one arm, and false flags cost about 2% when there is no leakage. So the
  decoder must be enabled only when the persistence test confirms leakage.
- Controls: no leakage, or leakage invisible in IQ, gives no gain.

**Leakage rates used (0.3%, 1% per measurement) and the neighbour model are assumptions, not IBM measurements.**
Next: a pre-registered hardware pilot (kerneled readout) to measure whether leakage is visible at all.
