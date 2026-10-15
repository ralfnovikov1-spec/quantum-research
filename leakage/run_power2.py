"""Realistic power test: leakage spreading through CZs (SIMULATION ONLY).
Compares hard / erasure / erasure+neighbours decoders on the same shots."""
import os, sys, time, json
from math import comb
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "03_code_hhmem")]
import numpy as np                                          # noqa: E402
from qiskit_ibm_runtime.fake_provider import FakeKingston   # noqa: E402
from hhmem import COMPRESS, Chip                            # noqa: E402
import leaksim as L                                         # noqa: E402
import leaksim2 as L2                                       # noqa: E402

SHOTS, SEEDS = 10000, 3
hw = FakeKingston()
chip = Chip(hw)
arms = {}
for basis in ("Z", "X"):
    Lay = chip.place(basis, {}, top=3)
    for v, kw in (("baseline", {}), ("reset-free", COMPRESS[basis])):
        arms[(basis, v)] = chip.build(basis, 4, kw, Lay)[1]
SCEN = [("no leakage (control)", 0.0, True),
        ("0.3% leakage, invisible (control)", 0.003, False),
        ("0.3% leakage, visible", 0.003, True),
        ("1% leakage, visible", 0.01, True)]


def p1(b, c):
    n = b + c
    return sum(comb(n, i) for i in range(b, n + 1)) / 2 ** n if n else 1.0


out, t0 = {}, time.time()
for name, pl, vis in SCEN:
    tot = {"hard": 0, "erasure": 0, "erasure+N": 0}
    mc = {"erasure": [0, 0], "erasure+N": [0, 0]}
    per_arm = {}
    for (basis, v), circ in arms.items():
        ha = {"hard": 0, "erasure": 0, "erasure+N": 0}
        for seed in range(SEEDS):
            rng = np.random.default_rng(10007 * seed + 31 * len(v) + ord(basis))
            det, obs, fl = L2.simulate(circ, SHOTS, pl, L.IQModel(leak_visible=vis), rng)
            h, e, en, _ = L2.decode_three(circ, det, obs, fl)
            for k, arr in (("hard", h), ("erasure", e), ("erasure+N", en)):
                tot[k] += int(arr.sum()); ha[k] += int(arr.sum())
            for k, arr in (("erasure", e), ("erasure+N", en)):
                b, c, _ = L.mcnemar(h, arr); mc[k][0] += b; mc[k][1] += c
        per_arm["%s/%s" % (basis, v)] = ha
    out[name] = {"total": tot, "mcnemar": mc, "per_arm": per_arm}
    print("%-34s hard %6d | erasure %6d (%+.1f%%, p_better=%.1e) | erasure+N %6d (%+.1f%%, p_better=%.1e)"
          % (name, tot["hard"], tot["erasure"], 100 * (tot["erasure"] / tot["hard"] - 1), p1(*mc["erasure"]),
             tot["erasure+N"], 100 * (tot["erasure+N"] / tot["hard"] - 1), p1(*mc["erasure+N"])), flush=True)
    for a, ha in per_arm.items():
        print("      %-14s hard %5d  erasure %5d  erasure+N %5d" % (a, ha["hard"], ha["erasure"], ha["erasure+N"]), flush=True)
print("%.0f s" % (time.time() - t0))
json.dump(out, open(os.path.join(HERE, "power2_result.json"), "w"), indent=1)
