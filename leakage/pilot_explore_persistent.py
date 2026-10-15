"""EXPLORATORY (post hoc): leakage-aware decoding using only PERSISTENT flags
(an ancilla flagged at two consecutive measurements; the flag is kept from
the first of them on). Same pilot shots. Not a test - any real claim needs
pre-registration on new data."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "03_code_hhmem")]
from math import comb                                   # noqa: E402
import numpy as np                                      # noqa: E402
import stim                                             # noqa: E402
import hwpipeline as P                                  # noqa: E402
import leaksim as L                                     # noqa: E402
import leaksim2 as L2                                   # noqa: E402

stem = sys.argv[1]
plan = json.load(open(stem + "_plan.json"))
iq = np.load(stem + "_iq.npz")
qc = plan["cal_qubits"]
blobs = {q: P.Blobs(iq["pub0"][:, j], iq["pub1"][:, j]) for j, q in enumerate(qc)}
kq = {q: P.choose_k(blobs[q], iq["pub0"][:, j], iq["pub1"][:, j]) for j, q in enumerate(qc)}
B = C = H = A = 0
for k, arm in enumerate(plan["arms"]):
    circ = stim.Circuit(arm["stim"])
    rq, anc = L.readout_records(circ), L.ancilla_records(circ)
    data = iq["pub%d" % (2 + k)]
    rc = np.array(arm["rec_clbits"])
    hard = P.hard_bits_for_arm(data, rc, rq, blobs)
    fl = P.flags_for_arm(data, rc, rq, blobs, kq, anc)
    keep = np.zeros_like(fl)
    by_q = {}
    for r in anc:
        by_q.setdefault(int(rq[r]), []).append(r)
    for recs in by_q.values():
        recs.sort()
        for a, b in zip(recs[:-1], recs[1:], strict=True):
            both = fl[:, a] & fl[:, b]
            keep[:, a] |= both
            keep[:, b] |= both
    det, obs = circ.compile_m2d_converter().convert(measurements=hard, separate_observables=True)
    h, _, en, _ = L2.decode_three(circ, det, obs[:, 0], keep)
    b, c = int((h & ~en).sum()), int((~h & en).sum())
    B, C, H, A = B + b, C + c, H + int(h.sum()), A + int(en.sum())
    print("%s %-10s shots with persistent flags %4d | errors hard %4d aware %4d (fixes %d, breaks %d)"
          % (arm["basis"], arm["variant"], int(keep.any(1).sum()), h.sum(), en.sum(), b, c))
n = B + C
p = sum(comb(n, i) for i in range(B, n + 1)) / 2 ** n if n else 1.0
print("pooled: hard %d -> aware(persistent) %d (%+.2f%%), fixes %d breaks %d, one-sided p = %.3f"
      % (H, A, 100 * (A / H - 1), B, C, p))
