"""EXPLORATORY (post hoc, not pre-registered): where do the pilot's
persistent leak flags come from?

For each ancilla qubit: flag count, repeat count (flag at measurement i and
i+1), and whether its flagged IQ points form one coherent off-segment
cluster (leakage-like) or scatter (noise / readout fault).
    python leakage/pilot_explore.py leakage/pilot_results/<stamp>
"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "03_code_hhmem")]
import numpy as np                                      # noqa: E402
import stim                                             # noqa: E402
import hwpipeline as P                                  # noqa: E402
import leaksim as L                                     # noqa: E402

stem = sys.argv[1]
plan = json.load(open(stem + "_plan.json"))
iq = np.load(stem + "_iq.npz")
qc = plan["cal_qubits"]
iq0, iq1 = iq["pub0"], iq["pub1"]
blobs = {q: P.Blobs(iq0[:, j], iq1[:, j]) for j, q in enumerate(qc)}
kq = {q: P.choose_k(blobs[q], iq0[:, j], iq1[:, j]) for j, q in enumerate(qc)}
per_q = {}
for k, arm in enumerate(plan["arms"]):
    circ = stim.Circuit(arm["stim"])
    rq, anc = L.readout_records(circ), L.ancilla_records(circ)
    data = iq["pub%d" % (2 + k)]
    rc = np.array(arm["rec_clbits"])
    fl = P.flags_for_arm(data, rc, rq, blobs, kq, anc)
    by_q = {}
    for r in anc:
        by_q.setdefault(int(rq[r]), []).append(r)
    for q, recs in by_q.items():
        recs.sort()
        d = per_q.setdefault(q, {"meas": 0, "flags": 0, "repeats": 0, "pts": []})
        d["meas"] += fl.shape[0] * len(recs)
        d["flags"] += int(fl[:, recs].sum())
        d["repeats"] += int(sum((fl[:, a] & fl[:, b]).sum()
                                for a, b in zip(recs[:-1], recs[1:], strict=True)))
        for r in recs:
            z = data[fl[:, r], rc[r]]
            d["pts"] += list(z)
tot_rep = sum(d["repeats"] for d in per_q.values())
print("ancilla qubits:", len(per_q), "| total repeats:", tot_rep)
print(" qubit  flag-rate  repeats  share  cluster: mean dist from |1> (sigma), spread of direction")
rows = []
for q, d in per_q.items():
    b = blobs[q]
    pts = P.iq2(np.array(d["pts"])) if d["pts"] else np.zeros((0, 2))
    if len(pts) >= 5:
        v = pts - b.m1
        ang = np.arctan2(v[:, 1], v[:, 0])
        spread = float(np.abs(np.mean(np.exp(1j * ang))))    # 1 = one direction
        dist = float(np.median(np.sqrt(np.einsum("ij,jk,ik->i", v, b.icov, v))))
    else:
        spread, dist = float("nan"), float("nan")
    rows.append((d["repeats"], q, d["flags"] / d["meas"], spread, dist))
rows.sort(reverse=True)
for rep, q, fr, sp, di in rows[:12]:
    print(" q%-4d  %.4f    %4d    %4.0f%%  %5.1f sigma, direction coherence %.2f"
          % (q, fr, rep, 100 * rep / max(tot_rep, 1), di, sp))
top2 = sum(r[0] for r in rows[:2])
print("share of repeats on the top 2 qubits: %.0f%%" % (100 * top2 / max(tot_rep, 1)))
