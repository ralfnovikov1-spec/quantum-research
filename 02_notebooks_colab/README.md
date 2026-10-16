# Colab notebooks (primary sources)

These are the Google Colab notebooks the hardware runs were submitted from, published as run on
14–15 Oct 2026 with their outputs. They are the source of every job ID in `04_hardware_runs/` and of
the P1–P16 verdict texts quoted in the root README. Nothing here was re-executed after the fact; the
only edit is the key handling (below).

| Notebook | What it is | Status of its claims |
|---|---|---|
| `Qmem1_hardware_first_runs.ipynb` | First exploratory study: repetition-code memory, qubit-role assignment, snapshot-swap and self-healing placement, Λ test on `ibm_marrakesh`, nomination/pilot/verification runs on two devices | **Superseded.** Several conclusions did not survive the 15 Oct audit (`05_audit/PHYSICS_AUDIT_2026-10-15.md`, item 9). Kept for provenance. |
| `Qmem1_fixed_audit_2026-10-15.ipynb` | The same study after the audit: every cell labelled SIMULATION or HARDWARE, pre-registered P1–P3 / Q1–Q3 with PASS/FAIL, Newcombe intervals, counts ≥ 10 rule for Λ, hardware cells re-analysed from saved output and not re-run, a benchmark gate (p_L = 3p² − 2p³) that all simulation verdicts must pass | Current. |
| `Qmem2_heavy_hex_reset_free.ipynb` | The heavy-hex study behind the root README results: realistic chip noise model, layout tournament, native heavy-hex subsystem code rebuilt from qiskit-qec's construction, reset-free syndrome extraction, live probe (H10c), Marrakesh replication decode (H17), measurement-cleanliness check (H20), the daily replication cell | Current. Hardware numbers are re-derived independently by `04_hardware_runs/verify_hw.py`. |

Later notebooks live next to the code they belong to: `03_code_hhmem/Qmem3.ipynb` (the 15 Oct
pre-registered rerun) and `noise_spectroscopy/Qmem4*.ipynb`.

## Reading order

1. `Qmem2_heavy_hex_reset_free.ipynb` for the published result.
2. `03_code_hhmem/Qmem3.ipynb` for the pre-registered rerun that confirmed it.
3. `Qmem1_fixed_audit_2026-10-15.ipynb` for the earlier study and what the audit changed; compare with
   `Qmem1_hardware_first_runs.ipynb` to see the original claims.

## Running them

Open in Colab (File → Open notebook → GitHub → `ralfnovikov1-spec/quantum-research`).
The IBM Quantum key is read from **Colab Secrets** under the name `IBM_QUANTUM_KEY`:

```python
from google.colab import userdata
API_KEY = userdata.get("IBM_QUANTUM_KEY")
```

Add the secret in the Colab sidebar (🔑) before running a hardware cell. No notebook in this repository
contains a key, and the repository has GitHub secret scanning with push protection enabled.
Hardware cells submit real jobs and cost QPU time; the simulation cells run free on the Colab CPU.

## What was changed before publishing

- The key lines (`API_KEY = "..."`, `token = "..."`) were replaced by the Colab-secret lookup above.
  Cell count, order, code and outputs are otherwise identical to the Drive originals.
- Colab widget metadata was dropped. Outputs were kept so the numbers can be read without a QPU.
