# Physics audit of all Colab notebooks — 15 Oct 2026

Scope: 32 exploratory notebooks from the author's Colab workspace (not published in this repository). Read-only;
nothing was run against IBM. "Recomputed" = re-derived offline (`check_scripts/`); everything
else is quoted from the notebooks' own saved outputs. Two severe claims were re-checked by hand
(crystal_quantum "127.000" output; Nisq9 `unitary_step` restarting from |0…0⟩) — both confirmed.

## Verdict per notebook

| Notebook | What it does | Verdict |
|---|---|---|
| Qmem2 | heavy-hex reset-free memory, IBM hardware (14 Oct) | ISSUES — see `01_reports` audit; clean rerun in Qmem3 |
| Qmem3 | clean pre-registered rerun (15 Oct) | new — results in `04_hardware_runs` |
| Qmem | repetition/surface-code memory, Aer + 3 hardware jobs | ISSUES |
| Q_storage | 3-qubit repetition code on FakeBrisbane | ISSUES |
| book_3 | toric-code threshold | ISSUES |
| Nisq4 | MIPT scan + scaling collapse + NARMA | ISSUES |
| Nisq8, nisq8 / Copy_of_nisq8 | MIPT "critical memory"; Lindblad reservoir vs ESN | ISSUES |
| quantum_memory, quantum_memory_2 | reservoir delay task; "cascaded" reservoirs | ISSUES |
| aethel_v125 | stim neutral-atom reservoir | ISSUES |
| Entropy_phase_3, IG_quantum, Hydrodynamic_memory, Hardware1 | classical Lorenz/Chua dynamics labelled quantum | ISSUES |
| Nisq9, Nisq10, memory_a_ja_b | "MIPT memory no-go" and its "decisive test" | INVALID-PHYSICS (reset bug) |
| crystal_quantum | Cirq time crystal / MBL / wormhole | INVALID-PHYSICS (integer overflow) |
| Quantum_energy | quantum battery | INVALID-PHYSICS (energy > maximum, purity > 1) |
| book_4, book5, book6 | holography, SYK wormhole, Page–Wootters | INVALID-PHYSICS (non-Hermitian H, hardcoded result) |
| hybrid_qubit | "autonomous QEC" qubit–qutrit | INVALID-PHYSICS (not a QEC model) |
| Chrono_thermodynamic | Maxwell demon, scars, time crystal | INVALID-PHYSICS (Landauer cost missing, sweep-edge "thresholds") |
| Nisq5, Nisq6, Nisq7, MIET | MIPT scans | exploratory, no claims (Nisq7/MIET share the reset bug) |
| migren | migraine "trial" on synthetic data | not physics; circular synthetic trial |
| quantum | empty | — |

## Defects, most severe first (fix in brackets)

1. **crystal_quantum c0,5,7–11** — `1 - 2*measurements` on uint8 wraps → "Max Crystallinity: 127.000 (Theoretical Max: 1.0)"; conclusions are hardcoded strings. [`.astype(np.int64)` first; compute verdicts from data]
2. **Quantum_energy c4–6, c11** — stored energy 40 249.99 vs maximum 157.08 (N=5); purity 1.095 / 1.139. Battery starts excited, H commutes with the energy operator, the "charging" is RK45 norm drift. [start in |g⟩, add coupling, tight tolerances, assert ‖ψ‖=1, E ≤ NΩ, Tr ρ² ≤ 1]
3. **Reset bug — Nisq9 c1/c2, Nisq10 c0, Nisq8 c0/c1, MIET c0, memory_a_ja_b c0/c1, quantum_memory c0** — the QNode `unitary_step` always starts from |0…0⟩, so the reservoir has no memory *by construction*; the "no-go" and the "decisive test" are artefacts. Nisq9 also lacks the odd brick-wall layer (purity = 1.0000 at every p). [`qml.StatePrep(state, ...)` first; add `range(1, N-1, 2)` layer; re-run]
4. **book_4 c6–7, book5 c0–1** — SYK H built with `1j*J*χχχχ` → anti-Hermitian (‖H−H†‖ = 12.2, recomputed); propagator renormalized to hide it. [drop `1j`, drop single-Majorana terms, assert H = H†]
5. **book_4 c2** — `results.append(1.0)` → "HOLOGRAPHY CONFIRMED"; c5 picks the best of 900 grid points after c4 printed "SIGNAL LOST"; book6 c5 hand-builds the state after c4 printed "FROZEN".
6. **book_4 c4–5, book5, book6 c3** — "right" signal is the LEFT reduced state (−0.0524 = −⟨Y_L⟩, recomputed). [`rho_R = M.T @ M.conj()`]
7. **hybrid_qubit c22–37** — "5-qubit perfect code" state is not an eigenstate of its own stabilizers (⟨ZZZZI⟩=0.5); Lindblad L=ZZ only dephases, cannot correct; Kraus ΣK†K = 1.0000563 then renormalized; "correction" = taking the top eigenvector of ρ (unphysical); "threshold" from one size and one trajectory. [XZZXI stabilizers; jumps X_i(1−Z_iZ_j)/2; threshold only from ≥2 distances]
8. **Chrono_thermodynamic c1–7** — `np.interp` on non-monotonic data; Λ_c equals sweep upper bounds yet printed "ROBUST"; demon's Landauer erasure cost never counted; "Perpetual Clock" from one L=10, 50-cycle, single-disorder run; "need high disorder" is hardcoded and contradicted by its own data (h=0 best at every ε).
9. **Qmem c3,c5,c12,c16,c29–32** — hardware Λ(3→5) = 47 and 27 from zero-count cells; σ=0 at zero counts feeds the 2σ test; decoder rule "pre-registered" after seeing the data; P3 (>70%) missed (68%, 67%) and never marked FAIL; FakeOsaka called "today's real chip". [Wilson/Clopper–Pearson intervals; Λ only when both counts ≥10; PASS/FAIL line per prediction; label simulations]
10. **Q_storage c5,c14,c15** — "EXPECTED HARDWARE FIDELITY 90.33% … PRE-FLIGHT CHECK PASSED" is a FakeBrisbane simulation with pass condition `fidelity > 0`; "BETTER than bare" in 1 of 6 cases on classical basis states with no idle time (= readout majority vote, not memory); c15 crashes (`BEST_BARE` undefined).
11. **Entropy_phase_3** — "Arrow of Time CONFIRMED" KL is set by the smoothing constant (0.91 → 0.39 → 0.19 as α goes 1e-10 → 1e-3 → 0.5, recomputed); entropy rate ~20× the Lorenz λ₁; c2/c4 test identities (ΔH ≡ 0).
12. **Nisq4 c10–12** — scaling collapse uses pasted arrays and an interpolation grid far outside the data; "ν = 2.43" not meaningful.
13. **book_3 c2** — threshold from 200 trials (~0.5σ difference); the 0.103 line is hardcoded.
14. Smaller: IG_quantum (η=0 "causal control", wrong variational equation); Hydrodynamic_memory (⟨σx⟩ of a σz eigenstate ≡ 0 → identical τ at every ε); quantum_memory_2 (cascaded Lindbladian missing the cascaded Hamiltonian term); Hardware1 (classical ODE presented as quantum/"Mind", hardcoded "confirmed"); aethel (Z errors applied right before Z measurement → no effect); migren (circular synthetic trial).

## What is sound and worth keeping
Qmem noiseless self-test (c4), Stim noise self-test (c22), T2 ≤ 2·T1 clamp; Q_storage 16-case noiseless self-test (c11); book_3 c0–1; honest null results in Chrono c5, quantum_memory_2 c1 ("NO-GO"), Quantum_energy c7/c13/c20/c21.

## Not verified
Cirq not installed locally — the uint8 cause in crystal_quantum is inferred from the 127.000 output. Time-crystal recomputation used uniform couplings (approximation). Qmem c16 decoder-weight mismatch is a likely cause, not confirmed.

## Secrets
IBM API key hardcoded in Qmem (3 cells), Qmem2 (10), Q_storage had a token-style assignment (redacted here). Copies in `02_notebooks_colab` are redacted; the originals in Drive are not. Rotate the key.
