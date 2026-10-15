"""Dynamical-decoupling noise spectroscopy on every usable qubit at once.

Sequences (all qubits in parallel, delays in microseconds):
  T1     : X - wait - measure                  -> P(1) decay
  RAMSEY : H - wait - H - measure              -> coherence, all noise
  ECHO   : H - w/2 - X - w/2 - H               -> slow noise refocused
  CPMG4  : H - (w/8 X w/4 X ... w/8) - H       -> 4 pulses
  CPMG16 : 16 pulses
Coherence C(w) = 2 P(0) - 1, normalised by its shortest-wait value.
Each decay is fitted to C = exp(-(w/T)^beta): beta ~ 1 means white
(Markovian) noise; beta ~ 2 means slow, quasi-static noise. This
identifies the noise SPECTRUM, not its microscopic origin (magnetic
flux, two-level defects and charge noise can all be slow).
"""
import numpy as np
from qiskit import QuantumCircuit
from qiskit.transpiler.preset_passmanagers import \
    generate_preset_pass_manager

WAITS_US = (2, 6, 15, 35, 70, 140, 280)
SEQS = ("T1", "RAMSEY", "ECHO", "CPMG4", "CPMG16")
NPULSE = {"RAMSEY": 0, "ECHO": 1, "CPMG4": 4, "CPMG16": 16}
SHOTS = 1000


def _dt(hw, us):
    return int(us * 1e-6 / hw.dt) // 32 * 32


def circuit(hw, seq, wait_us, qubits):
    nq = hw.num_qubits
    qc = QuantumCircuit(nq, len(qubits))
    T = _dt(hw, wait_us)
    if seq == "T1":
        qc.x(qubits)
        qc.delay(T, qubits, unit="dt")
    else:
        n = NPULSE[seq]
        qc.h(qubits)
        if n == 0:
            qc.delay(T, qubits, unit="dt")
        else:
            seg = T // (2 * n) // 32 * 32
            qc.delay(seg, qubits, unit="dt")
            for k in range(n):
                qc.x(qubits)
                qc.delay(2 * seg if k < n - 1 else seg, qubits, unit="dt")
        qc.h(qubits)
    for j, q in enumerate(qubits):
        qc.measure(q, j)
    return qc


def build(hw, qubits):
    pm = generate_preset_pass_manager(
        optimization_level=0, target=hw.target,
        initial_layout=list(range(hw.num_qubits)))
    circs, index = [], []
    for seq in SEQS:
        for w in WAITS_US:
            circs.append(pm.run(circuit(hw, seq, w, qubits)))
            index.append((seq, w))
    return circs, index


def fit_decay(ws, c):
    """C = exp(-(w/T)^beta) on points with 0.05 < C < 0.95.
    Returns (T_us, beta) or (nan, nan) if fewer than 3 usable points."""
    ws, c = np.asarray(ws, float), np.asarray(c, float)
    ok = (c > 0.05) & (c < 0.95)
    if ok.sum() < 3:
        return float("nan"), float("nan")
    x = np.log(ws[ok])
    y = np.log(-np.log(c[ok]))
    beta, b0 = np.polyfit(x, y, 1)
    T = float(np.exp(-b0 / beta)) if beta > 0 else float("nan")
    return T, float(beta)


def analyse(res, index, qubits):
    """res[k].data.c bitstrings; returns {q: {seq: (T, beta)}}."""
    curves = {q: {s: [] for s in SEQS} for q in qubits}
    for k, (seq, w) in enumerate(index):
        bs = res[k].data.c.get_bitstrings()
        raw = np.frombuffer("".join(bs).encode(), np.uint8)
        arr = (raw.reshape(len(bs), -1) - 48)[:, ::-1].astype(bool)
        for j, q in enumerate(qubits):
            p1 = arr[:, j].mean()
            val = p1 if seq == "T1" else 1 - 2 * p1   # P(1) or 2P(0)-1
            curves[q][seq].append((w, val))
    out = {}
    for q in qubits:
        out[q] = {}
        for seq in SEQS:
            pts = sorted(curves[q][seq])
            ws = [p[0] for p in pts]
            v = np.array([p[1] for p in pts])
            if v[0] <= 0.2:            # no contrast: dead/unreadable
                out[q][seq] = (float("nan"), float("nan"))
                continue
            out[q][seq] = fit_decay(ws, v / v[0])
    return out


PREDICTIONS = {
    "M1": "echo T2 > Ramsey T2*: one-sided sign test p < 0.001 AND median "
          "ratio echo/Ramsey >= 1.2 (low-frequency noise present)",
    "M2": "CPMG16 T2 > echo T2: one-sided sign test p < 0.001 AND median "
          "ratio >= 1.1 (noise spectrum not white)",
    "M3": "BENCHMARK: CPMG16 T2 <= 2*T1 (+20% fit margin) on >= 95% of "
          "fitted qubits (physical bound T2 <= 2 T1)",
    "M4": "median Ramsey beta >= 1.3 (Gaussian-like decay => slow noise "
          "dominates dephasing) - low power for mixed noise (simulated)",
}


def _sign_p(k, n):
    """one-sided binomial P(X >= k | n, 1/2)."""
    from math import comb
    return sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n


def _paired(fits, a, b):
    rows = [q for q in fits if np.isfinite(fits[q][a][0])
            and np.isfinite(fits[q][b][0]) and fits[q][b][0] > 0]
    if not rows:
        return float("nan"), float("nan"), 0
    r = np.array([fits[q][a][0] / fits[q][b][0] for q in rows])
    return _sign_p(int((r > 1).sum()), len(r)), float(np.median(r)), len(r)


def verdicts(fits):
    p1, r1, n1 = _paired(fits, "ECHO", "RAMSEY")
    p2, r2, n2 = _paired(fits, "CPMG16", "ECHO")
    rows = [q for q in fits if np.isfinite(fits[q]["CPMG16"][0])
            and np.isfinite(fits[q]["T1"][0])]
    m3 = (sum(fits[q]["CPMG16"][0] <= 2.4 * fits[q]["T1"][0] for q in rows)
          / len(rows)) if rows else float("nan")
    betas = [fits[q]["RAMSEY"][1] for q in fits
             if np.isfinite(fits[q]["RAMSEY"][1])]
    mb = float(np.median(betas)) if betas else float("nan")
    return {
        "M1": (p1 < 1e-3 and r1 >= 1.2,
               "p=%.1e, median ratio %.2f, n=%d" % (p1, r1, n1)),
        "M2": (p2 < 1e-3 and r2 >= 1.1,
               "p=%.1e, median ratio %.2f, n=%d" % (p2, r2, n2)),
        "M3": (m3 >= 0.95, "%.0f%% of %d" % (100 * m3, len(rows))),
        "M4": (mb >= 1.3, "median beta %.2f (n=%d)" % (mb, len(betas))),
    }
