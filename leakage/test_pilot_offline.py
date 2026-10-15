"""Offline end-to-end test of pilot.build + pilot.analyse with fake kerneled
data shaped like SamplerV2(meas_type="kerneled"): complex shots x clbits."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "03_code_hhmem")]
import numpy as np                                          # noqa: E402
import stim                                                 # noqa: E402
from qiskit_ibm_runtime.fake_provider import FakeKingston   # noqa: E402
import leaksim as L                                         # noqa: E402
import leaksim2 as L2                                       # noqa: E402
import pilot                                                # noqa: E402

rng = np.random.default_rng(5)
hw = FakeKingston()
pubs, plan = pilot.build(hw)
print("circuits:", len(pubs), "| calibrated qubits:", len(plan["cal_qubits"]))
geo = {}
for q in range(hw.num_qubits):
    th = rng.uniform(0, 2 * np.pi)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    m0 = rng.normal(size=2) * 3
    m1 = m0 + R @ np.array([rng.uniform(4.2, 5.2), 0])
    a = rng.uniform(0.6, 1.2) * rng.choice([-1, 1])
    geo[q] = (m0, m1, m1 + R @ (np.array([np.cos(a), np.sin(a)]) * 3.8))


def pts(q, st):
    m0, m1, m2 = geo[q]
    mu = np.where(st[:, None] == 0, m0, np.where(st[:, None] == 1, m1, m2))
    dec = (st == 1) & (rng.random(len(st)) < 0.03)
    mu = np.where(dec[:, None], m1 + rng.random(len(st))[:, None] * (m0 - m1), mu)
    p = mu + rng.normal(size=mu.shape)
    return p[:, 0] + 1j * p[:, 1]


class Pub:
    def __init__(self, a):
        self.data = type("D", (), {"c": a})()


for name, pl in (("no leakage", 0.0), ("0.5% leakage", 0.005)):
    S = pilot.SHOTS
    res = [Pub(np.stack([pts(q, np.full(S, b)) for q in plan["cal_qubits"]], 1))
           for b in (0, 1)]
    for arm in plan["arms"]:
        circ = stim.Circuit(arm["stim"])
        st = L2.simulate(circ, S, pl, L.IQModel(), rng, return_states=True)
        rq = L.readout_records(circ)
        n_cl = max(arm["rec_clbits"]) + 1
        a = np.zeros((S, n_cl), complex)
        for r, c in enumerate(arm["rec_clbits"]):
            a[:, c] = pts(int(rq[r]), st[:, r])
        res.append(Pub(a))
    out = pilot.analyse(res, plan)
    print("==", name, "| cal false flags %.1e" % out["cal_false_flag_rate"],
          "| L0", out["L0"], "| L1", {k: out["L1"][k] for k in ("holds", "tested", "p", "errors")})
