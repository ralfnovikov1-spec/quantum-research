# Reset-free syndrome extraction on IBM heavy-hex hardware

Hardware study of the IBM heavy-hex subsystem code (distance 3) on 156-qubit Heron processors
(`ibm_kingston`, `ibm_marrakesh`). The question: does removing mid-circuit resets from syndrome
extraction lower the logical error per round?

Every number in this repository can be traced to an IBM Quantum job ID. All 9 jobs were re-fetched
from IBM and re-analysed with independent code ([`04_hardware_runs/INDEPENDENT_REDERIVATION.json`](04_hardware_runs/INDEPENDENT_REDERIVATION.json)).

## Results (hardware, d = 3)

| Claim | Evidence | Status |
|---|---|---|
| Removing resets lowers Z-basis logical error per round | ×1.66, ×1.70, ×2.07, ×1.72 in four `ibm_kingston` runs, each 2σ-separated; the last run was pre-registered at 10,000 shots | **Supported** |
| The same holds in the X basis | ×1.13–1.42 on `ibm_kingston` (flag resets kept) | **Supported, smaller effect** |
| The same holds on a second chip | ×1.07 / ×1.05 on `ibm_marrakesh`, inside 2σ | **Not supported** (direction only) |
| Break-even (logical outlives best physical qubit) | Logical/physical lifetime 0.096× (Z), 0.136× (X), pre-registered | **Not reached** |
| A live probe of every qubit makes the noise model accurate | Probe-fed model within ±30% in 4/4 arms; calibration-fed model 1.5–2.1× optimistic | **Supported** (one day, one chip) |
| Larger codes help (Λ > 1) | Only d = 3 was run on hardware; simulation gives Λ < 1 | **Not tested on hardware** |

The full list of 21 pre-registered predictions, including the 8 that failed, is in the
[corrected report](01_reports/heavy-hex_report_CORRECTED_2026-10-15.pdf).

Prior work: Gehér et al., *To reset, or not to reset* (arXiv:2408.00758) studied reset-free extraction
in simulation. IBM's 2022 heavy-hex experiment is arXiv:2203.07205. This repository adds a replicated
hardware measurement, not a new theory. No full literature review has been done.

## Layout

| Path | Content |
|---|---|
| `01_reports/` | Corrected report (PDF + HTML). It supersedes an earlier version that had post-hoc and unlabelled failures. |
| `03_code_hhmem/hhmem.py` | Heavy-hex subsystem code builder, GF(2) detector repair, Stim noise model, zero-SWAP placement, X–X DD, decoding |
| `03_code_hhmem/hhmem2.py` | Live probe, dead-qubit filter (applied before selection), pre-registration with timestamp and SHA-256, verdicts |
| `03_code_hhmem/dry_run.py` | End-to-end test on `FakeKingston` with an injected dead qubit; includes a control run with the filter disabled |
| `03_code_hhmem/Qmem3.ipynb` | Colab notebook used for the 15 Oct 2026 pre-registered rerun |
| `04_hardware_runs/` | Plans (circuits, detectors, layouts) for every job, probe data, the rerun's pre-registration and verdicts, and `verify_hw.py` (independent re-derivation) |
| `05_audit/` | Audit of the author's earlier exploratory notebooks (not published here), with recomputation scripts |

## Reproduce

```bash
pip install "qiskit>=1.3" qiskit-ibm-runtime stim pymatching rustworkx
python 03_code_hhmem/dry_run.py                      # offline, FakeKingston
export IBM_QUANTUM_KEY=...                           # your own key; never commit it
python 04_hardware_runs/verify_hw.py 04_hardware_runs/2026-10-14_plans 04_hardware_runs
```

Tested with Python 3.11 and 3.13, qiskit 2.5.2, qiskit-ibm-runtime 0.50.0, stim 1.16.0 and pymatching 2.4.0.

## Limitations
One day per chip. Distance 3 only. A matching decoder only. Physical-qubit references are the best 3
usable qubits by live probe. IBM Quantum open plan (600 s QPU per month) bounds the sample size.
