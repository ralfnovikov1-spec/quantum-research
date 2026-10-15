"""Leakage pilot on IBM hardware with kerneled (IQ) readout.

build(hw)      -> circuits + plan (calibration + 4 arms, 4 rounds)
analyse(res, plan) -> verdicts L0/L1 + exploratory per-round leak rates
Pre-registered in PILOT_PREREGISTRATION.md before submission.
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "03_code_hhmem")]
from math import comb                                       # noqa: E402

import numpy as np                                          # noqa: E402
import stim                                                 # noqa: E402
from qiskit import QuantumCircuit                           # noqa: E402
from qiskit.transpiler.preset_passmanagers import \
    generate_preset_pass_manager                            # noqa: E402

from hhmem import COMPRESS, Chip                            # noqa: E402
import hwpipeline as P                                      # noqa: E402
import leaksim as L                                         # noqa: E402
import leaksim2 as L2                                       # noqa: E402

SHOTS = 4000
ROUNDS = 4
FP_ABORT = 2e-3


def build(hw, exclude=(146,)):
    chip = Chip(hw)
    t = hw.target
    ro = lambda q: (t["measure"].get((q,)).error
                    if t["measure"].get((q,)) is not None else 1.0)
    bad = {q for q in range(hw.num_qubits) if ro(q) > 0.10} | set(exclude)
    import hhmem2 as H
    arms, pubs = [], []
    for basis in ("Z", "X"):
        lay = H.place_filtered(chip, basis, {}, bad)
        for v in ("baseline", "reset-free"):
            kw = COMPRESS[basis] if v == "reset-free" else {}
            isa, circ, arm = chip.build(basis, ROUNDS, kw, lay)
            arm.update(variant=v, rec_clbits=P.record_clbits(isa).tolist())
            arms.append(arm)
            pubs.append(isa)
    qubits = sorted({q for a in arms for q in a["layout"]})
    pm = generate_preset_pass_manager(optimization_level=0, target=hw.target,
                                      initial_layout=list(range(hw.num_qubits)))
    cal = []
    for prep in (0, 1):
        qc = QuantumCircuit(hw.num_qubits, len(qubits))
        if prep:
            qc.x(qubits)
        for j, q in enumerate(qubits):
            qc.measure(q, j)
        cal.append(pm.run(qc))
    plan = {"backend": hw.name, "shots": SHOTS, "rounds": ROUNDS,
            "cal_qubits": qubits, "arms": arms, "excluded": sorted(bad)}
    return cal + pubs, plan


def _iq(res, k):
    return np.asarray(res[k].data.c)            # kerneled: shots x clbits


def analyse(res, plan):
    q_cal = plan["cal_qubits"]
    iq0, iq1 = _iq(res, 0), _iq(res, 1)
    blobs = {q: P.Blobs(iq0[:, j], iq1[:, j]) for j, q in enumerate(q_cal)}
    kq = {q: P.choose_k(blobs[q], iq0[:, j], iq1[:, j])
          for j, q in enumerate(q_cal)}
    fp = float(np.mean([np.mean(np.concatenate(
        [blobs[q].seg_dist(iq0[:, j]), blobs[q].seg_dist(iq1[:, j])]) > kq[q])
        for j, q in enumerate(q_cal)]))
    out = {"cal_false_flag_rate": fp, "arms": []}
    if fp > FP_ABORT:
        out["abort"] = "calibration false-flag rate %.1e > %.0e" % (fp, FP_ABORT)
        return out
    pers_rf = [0, 0]            # pooled repeats / conditioning counts (reset-free)
    base_num = base_den = 0
    H = {"hard": 0, "erasure+N": 0}
    B = C = 0
    for k, arm in enumerate(plan["arms"]):
        circ = stim.Circuit(arm["stim"])
        rq = L.readout_records(circ)
        anc = L.ancilla_records(circ)
        iq = _iq(res, 2 + k)
        rc = np.array(arm["rec_clbits"])
        lay_q = set(arm["layout"])
        if not set(rq.tolist()) <= set(blobs):
            raise ValueError("arm uses a qubit without calibration")
        assert lay_q <= set(q_cal)
        hard = P.hard_bits_for_arm(iq, rc, rq, blobs)
        flags = P.flags_for_arm(iq, rc, rq, blobs, kq, anc)
        pers = P.persistence(flags, rq, anc)
        det, obs = circ.compile_m2d_converter().convert(
            measurements=hard, separate_observables=True)
        h, _, en, _ = L2.decode_three(circ, det, obs[:, 0], flags)
        b = int((h & ~en).sum())
        c = int((~h & en).sum())
        B, C = B + b, C + c
        H["hard"] += int(h.sum())
        H["erasure+N"] += int(en.sum())
        rounds_rate = []
        by_q = {}
        for r in anc:
            by_q.setdefault(int(rq[r]), []).append(r)
        for i in range(max(len(v) for v in by_q.values())):
            cols = [v[i] for v in by_q.values() if i < len(v)]
            rounds_rate.append(float(flags[:, cols].mean()))
        if arm["variant"] == "reset-free":
            pers_rf[0] += pers[3]
            pers_rf[1] += pers[4]
            base_num += int(flags[:, anc].sum())
            base_den += flags[:, anc].size
        out["arms"].append({"basis": arm["basis"], "variant": arm["variant"],
                            "flag_rate": float(flags[:, anc].mean()),
                            "flag_rate_by_measurement_index": rounds_rate,
                            "persistence": [float(x) for x in pers],
                            "errors_hard": int(h.sum()),
                            "errors_aware": int(en.sum())})
    from scipy.stats import binom
    base = base_num / base_den if base_den else 0.0
    hits, cond = pers_rf
    ratio = (hits / cond) / base if cond and base > 0 else float("nan")
    pval = float(binom.sf(hits - 1, cond, base)) if cond and base > 0 else 1.0
    L0 = bool(ratio >= P.PERSIST_MIN and pval < 1e-3 and hits >= 20)
    n = B + C
    p1 = sum(comb(n, i) for i in range(B, n + 1)) / 2 ** n if n else 1.0
    out.update(L0={"holds": L0, "ratio": ratio, "repeats": hits, "p": pval},
               L1={"holds": bool(L0 and p1 < 0.01), "tested": L0,
                   "aware_fixes": B, "aware_breaks": C, "p": p1,
                   "errors": H})
    return out
