"""Power test of leakage-aware (erasure) decoding - SIMULATION ONLY.

Real heavy-hex circuits (hhmem, FakeKingston calibration, reset-free arm,
4 rounds), injected ancilla leakage, synthetic IQ readout.
    python run_power.py
"""
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "03_code_hhmem")]
import numpy as np                                          # noqa: E402
from qiskit_ibm_runtime.fake_provider import FakeKingston   # noqa: E402
from hhmem import COMPRESS, Chip                            # noqa: E402
import leaksim as L                                         # noqa: E402

SHOTS, SEEDS, P_LEAK = 20000, 5, 0.003
hw = FakeKingston()
chip = Chip(hw)
arms = {}
for basis in ("Z", "X"):
    Lay = chip.place(basis, {}, top=3)
    for v, kw in (("baseline", {}), ("reset-free", COMPRESS[basis])):
        _, circ, _ = chip.build(basis, 4, kw, Lay)
        arms[(basis, v)] = circ
SCEN = [("no leakage", 0.0, True), ("leakage, visible in IQ", P_LEAK, True),
        ("leakage, invisible in IQ", P_LEAK, False)]
t0 = time.time()
for name, pl, vis in SCEN:
    B = C = H = E = 0
    for (basis, v), circ in arms.items():
        for seed in range(SEEDS):
            rng = np.random.default_rng(1000 * seed + hash((basis, v)) % 997)
            det, obs, flags = L.simulate(circ, SHOTS, pl, L.IQModel(leak_visible=vis), rng)
            hw_, er, _ = L.decode_both(circ, det, obs, flags)
            b, c, _ = L.mcnemar(hw_, er)
            B, C, H, E = B + b, C + c, H + hw_.sum(), E + er.sum()
    n = B + C
    from math import comb
    p = sum(comb(n, i) for i in range(B, n + 1)) / 2 ** n if n else 1.0
    print("%-26s logical errors hard %6d  erasure %6d  change %+.2f%%  "
          "(fixes %d, breaks %d, one-sided p = %.1e)"
          % (name, H, E, 100 * (E - H) / H, B, C, p), flush=True)
print("%.0f s" % (time.time() - t0))
