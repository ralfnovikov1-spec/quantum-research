"""Offline end-to-end test of hwpipeline with realistic, messy IQ data.

The pipeline only sees what IBM would return: complex IQ per classical bit
(SamplerV2 meas_type="kerneled") plus prepare-|0>/|1> calibration shots.
Per qubit: random blob rotation/scale, 3% of |1> readouts decay part-way to
|0> during readout, and a |2> blob in a random off-axis direction.
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "03_code_hhmem")]
import numpy as np                                          # noqa: E402
from qiskit_ibm_runtime.fake_provider import FakeKingston   # noqa: E402
from hhmem import COMPRESS, Chip                            # noqa: E402
import leaksim as L                                         # noqa: E402
import leaksim2 as L2                                       # noqa: E402
import hwpipeline as P                                      # noqa: E402

SHOTS, CAL = 6000, 6000
rng = np.random.default_rng(3)
hw = FakeKingston()
chip = Chip(hw)
Lay = chip.place("X", {}, top=3)
isa, circ, arm = chip.build("X", 4, COMPRESS["X"], Lay)
rec_cl = P.record_clbits(isa)
rec_q = L.readout_records(circ)
anc = L.ancilla_records(circ)
assert len(rec_cl) == len(rec_q), "record/clbit mismatch"
nq = circ.num_qubits

geo = {}
for q in set(rec_q.tolist()):
    th = rng.uniform(0, 2 * np.pi)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    sep = rng.uniform(4.2, 5.2)
    m0 = rng.normal(size=2) * 3
    m1 = m0 + R @ np.array([sep, 0])
    ang = rng.uniform(0.6, 1.2) * rng.choice([-1, 1])
    m2 = m1 + R @ (np.array([np.cos(ang), np.sin(ang)]) * rng.uniform(3, 4.5))
    sc = rng.uniform(0.8, 1.2)
    geo[q] = (m0, m1, m2, sc)


def iq_points(q, states):
    m0, m1, m2, sc = geo[q]
    mu = np.where(states[:, None] == 0, m0, np.where(states[:, None] == 1, m1, m2))
    decay = (states == 1) & (rng.random(len(states)) < 0.03)
    t = rng.random(len(states))[:, None]
    mu = np.where(decay[:, None], m1 + t * (m0 - m1), mu)
    p = mu + rng.normal(size=mu.shape) * sc
    return p[:, 0] + 1j * p[:, 1]


cal0 = {q: iq_points(q, np.zeros(CAL, int)) for q in geo}
cal1 = {q: iq_points(q, np.ones(CAL, int)) for q in geo}
blobs = {q: P.Blobs(cal0[q], cal1[q]) for q in geo}
kq = {q: P.choose_k(blobs[q], cal0[q], cal1[q]) for q in geo}
fp = np.mean([np.mean(np.concatenate([blobs[q].seg_dist(cal0[q]), blobs[q].seg_dist(cal1[q])]) > kq[q]) for q in geo])
print("calibration false-flag rate (target %.0e): %.1e" % (P.FP_TARGET, fp))

for name, pl in (("no leakage", 0.0), ("1% leakage", 0.01)):
    states = L2.simulate(circ, SHOTS, pl, L.IQModel(), np.random.default_rng(11), return_states=True)
    iq_arm = np.zeros((SHOTS, isa.num_clbits), complex)
    for r, c in enumerate(rec_cl):
        iq_arm[:, c] = iq_points(int(rec_q[r]), states[:, r])
    hard = P.hard_bits_for_arm(iq_arm, rec_cl, rec_q, blobs)
    flags = P.flags_for_arm(iq_arm, rec_cl, rec_q, blobs, kq, anc)
    truth = states[:, anc] == 2
    det, obs = circ.compile_m2d_converter().convert(measurements=hard, separate_observables=True)
    h, e, en, npat = L2.decode_three(circ, det, obs[:, 0], flags)
    pers = P.persistence(flags, rec_q, anc)
    pn, base, ratio, hits, cond, pval = pers
    tp = (flags[:, anc] & truth).sum() / max(truth.sum(), 1)
    fpr = (flags[:, anc] & ~truth).sum() / max((~truth).sum(), 1)
    print("%-11s flag rate %.4f | true leaks caught %.0f%% | false flags %.1e | persistence "
          "P(next|now)=%.3f base=%.4f ratio=%.1f repeats=%d p=%.1e -> %s | errors hard %d erasure+N %d"
          % (name, base, 100 * tp, fpr, pn, base, ratio, hits, pval,
             "LEAKAGE" if P.leakage_detected(pers) else "noise", h.sum(), en.sum()))
