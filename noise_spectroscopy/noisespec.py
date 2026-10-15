"""Dynamical-decoupling noise spectroscopy on every usable qubit at once.

Sequences (all qubits in parallel, waits in microseconds):
  RO0, RO1  : readout calibration (prepare |0> / |1>, measure)
  T1        : X - wait - measure                         -> P(1) decay
  RAMSEY    : H - wait - H - measure                     -> <X>
  RAMSEY_Y  : H - wait - Sdg - H - measure               -> <Y>
  ECHO      : H - w/2 - X - w/2 - H                      -> slow noise refocused
  CPMG4/16  : 4 / 16 pulses at w (k + 1/2) / n
Readout is corrected per qubit with RO0/RO1. The Ramsey coherence is the
magnitude sqrt(<X>^2 + <Y>^2), so a static frequency offset (an imperfect
drive frequency) cannot masquerade as decay. Each curve is normalised by
its shortest-wait value and fitted to C = exp(-(w/T)^beta).

Interpretation limits (stated in advance):
  * beta -> 2 only for quasi-static noise; 1/f noise gives 1 < beta < 2,
    and the T1 contribution pulls beta towards 1.
  * Simultaneous X pulses on all qubits do not refocus static ZZ coupling
    (X(x)X commutes with Z(x)Z); ZZ affects every sequence alike.
  * This measures the noise spectrum, not its origin: magnetic flux
    noise, two-level defects and charge noise can all be slow.
"""
import numpy as np
from qiskit import QuantumCircuit
from qiskit.transpiler.preset_passmanagers import \
    generate_preset_pass_manager

WAITS_US = (2, 6, 15, 35, 70, 140, 280)
SEQS = ("T1", "RAMSEY", "RAMSEY_Y", "ECHO", "CPMG4", "CPMG16")
FITTED = ("T1", "RAMSEY", "ECHO", "CPMG4", "CPMG16")
NPULSE = {"RAMSEY": 0, "RAMSEY_Y": 0, "ECHO": 1, "CPMG4": 4, "CPMG16": 16}
SHOTS = 1000


def _dt(hw, us):
    return int(us * 1e-6 / hw.dt) // 32 * 32


def circuit(hw, seq, wait_us, qubits):
    qc = QuantumCircuit(hw.num_qubits, len(qubits))
    if seq in ("RO0", "RO1"):
        if seq == "RO1":
            qc.x(qubits)
    elif seq == "T1":
        qc.x(qubits)
        qc.delay(_dt(hw, wait_us), qubits, unit="dt")
    else:
        T, n = _dt(hw, wait_us), NPULSE[seq]
        qc.h(qubits)
        if n == 0:
            qc.delay(T, qubits, unit="dt")
        else:
            seg = T // (2 * n) // 32 * 32
            qc.delay(seg, qubits, unit="dt")
            for k in range(n):
                qc.x(qubits)
                qc.delay(2 * seg if k < n - 1 else seg, qubits, unit="dt")
        if seq == "RAMSEY_Y":
            qc.sdg(qubits)
        qc.h(qubits)
    for j, q in enumerate(qubits):
        qc.measure(q, j)
    return qc


def build(hw, qubits):
    pm = generate_preset_pass_manager(
        optimization_level=0, target=hw.target,
        initial_layout=list(range(hw.num_qubits)))
    index = [("RO0", 0), ("RO1", 0)] + [(s, w) for s in SEQS
                                        for w in WAITS_US]
    return [pm.run(circuit(hw, s, w, qubits)) for s, w in index], index


def fit_decay(ws, c):
    """C = exp(-(w/T)^beta) on points with 0.05 < C < 0.95.
    Returns (T_us, beta) or (nan, nan) if fewer than 3 usable points."""
    ws, c = np.asarray(ws, float), np.asarray(c, float)
    ok = (c > 0.05) & (c < 0.95)
    if ok.sum() < 3:
        return float("nan"), float("nan")
    beta, b0 = np.polyfit(np.log(ws[ok]), np.log(-np.log(c[ok])), 1)
    T = float(np.exp(-b0 / beta)) if beta > 0 else float("nan")
    return T, float(beta)


def _p1(res, k):
    bs = res[k].data.c.get_bitstrings()
    raw = np.frombuffer("".join(bs).encode(), np.uint8)
    return ((raw.reshape(len(bs), -1) - 48)[:, ::-1]).mean(0)


def analyse(res, index, qubits):
    """res[k].data.c bitstrings; returns {q: {seq: (T, beta)}}."""
    pos = {key: k for k, key in enumerate(index)}
    ro0 = _p1(res, pos[("RO0", 0)])               # P(1 | 0)
    ro1 = 1 - _p1(res, pos[("RO1", 0)])           # P(0 | 1)
    contrast = 1 - ro0 - ro1
    raw = {s: np.array([_p1(res, pos[(s, w)]) for w in WAITS_US])
           for s in SEQS}
    out = {}
    for j, q in enumerate(qubits):
        out[q] = {}
        if contrast[j] < 0.6:                       # unreadable qubit
            for s in FITTED:
                out[q][s] = (float("nan"), float("nan"))
            continue
        p = {s: (raw[s][:, j] - ro0[j]) / contrast[j] for s in SEQS}
        curves = {"T1": p["T1"]}
        cx, cy = 1 - 2 * p["RAMSEY"], 1 - 2 * p["RAMSEY_Y"]
        curves["RAMSEY"] = np.hypot(cx, cy)
        for s in ("ECHO", "CPMG4", "CPMG16"):
            curves[s] = 1 - 2 * p[s]
        for s in FITTED:
            v = curves[s]
            out[q][s] = ((float("nan"), float("nan")) if v[0] <= 0.2
                         else fit_decay(WAITS_US, v / v[0]))
    return out


PREDICTIONS = {
    "M1": "echo T2 > Ramsey T2* (offset-free |<X>,<Y>| magnitude): one-sided "
          "sign test p < 0.001 AND median ratio echo/Ramsey >= 1.2 "
          "(low-frequency noise present)",
    "M2": "CPMG16 T2 > echo T2: one-sided sign test p < 0.001 AND median "
          "ratio >= 1.1 (noise spectrum not white)",
    "M3": "FIT SANITY CHECK: CPMG16 T2 <= 2*T1 (+20% fit margin) on >= 95% of "
          "fitted qubits (physical bound T2 <= 2 T1)",
    "M4": "median Ramsey beta >= 1.3 (non-exponential decay => slow noise "
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
