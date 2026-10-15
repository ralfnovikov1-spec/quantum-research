"""Local dry run of hhmem2 on FakeKingston: Stim stands in for the QPU.

Truth noise = FakeKingston calibration, except qubit DEAD has 50%
readout (random record), like q146 on 14 Oct. Checks that the filter
removes it, that every stage runs, and that verdicts print.
"""
import sys, time
import numpy as np
from qiskit_ibm_runtime.fake_provider import FakeKingston
import hhmem2 as H
from hhmem import ChipNoise, isa_to_stim

DEAD = int(sys.argv[1]) if len(sys.argv) > 1 else 146


class Truth(ChipNoise):
    def ro(self, q):
        return 0.5 if q == DEAD else super().ro(q)


class _Reg:
    def __init__(self, s):
        self.s = s

    def get_bitstrings(self):
        return self.s


class _Data:
    def __init__(self, s):
        self.c = _Reg(s)


class _Pub:
    def __init__(self, s):
        self.data = _Data(s)


def simulate(isas, noise, shots, seed=5):
    """SamplerV2-shaped result: bitstrings with clbit 0 on the right."""
    out = []
    for n, isa in enumerate(isas):
        pos = []
        for inst in isa.data:
            if inst.operation.name == "measure":
                pos.append(isa.find_bit(inst.clbits[0]).index)
        c = isa_to_stim(isa, noise, [], {}, ro_mode="flip")
        m = c.compile_sampler(seed=seed + n).sample(shots)
        arr = np.zeros((shots, isa.num_clbits), bool)
        for j, cb in enumerate(pos):
            arr[:, cb] = m[:, j]          # later measurement wins
        s = ["".join("1" if b else "0" for b in row[::-1]) for row in arr]
        out.append(_Pub(s))
    return out


hw = FakeKingston()
truth = Truth(hw)
t = time.time()
pcs, w = H.probe_circuits(hw)
live = H.analyze_probe(simulate(pcs, truth, H.PROBE_SHOTS), hw, w)
bad = sorted(q for q, v in live.items() if not v["usable"])
print("probe %.0fs | unusable %d: %s" % (time.time() - t, len(bad), bad[:15]))
print("  dead qubit", DEAD, "usable?", live[DEAD]["usable"], live[DEAD]["why"])
assert not live[DEAD]["usable"], "FILTER MISSED THE DEAD QUBIT"

t = time.time()
rec, pubs = H.build_plan(hw, live, "fake-probe")
print("plan %.0fs | %d circuits | layouts %s" % (
    time.time() - t, len(pubs),
    sorted({(a["basis"], tuple(a["layout"][:4])) for a in rec["arms"]})))
used = {q for a in rec["arms"] for q in a["layout"]} | \
       {q for p in rec["phys"] for q in p["qubits"]}
assert not used & set(bad), "unusable qubit used"
print("  phys refs:", sorted({(p["basis"], tuple(p["qubits"])) for p in rec["phys"]}))
print("  waits us:", sorted({round(p["us"], 1) for p in rec["phys"]}))
H.preregister(rec, "dry_out")
rec["job_id"] = "fake-main"

t = time.time()
res = simulate(pubs, truth, H.SHOTS)
out = H.analyze(res, rec)
print("sim+decode %.0fs" % (time.time() - t))
H.report(out, rec)
