"""Offline stand-in for IBM Quantum used by `daily.py fake`.

FakeKingston calibration is the truth model, except qubit 146 has a
random (50 %) readout, like the real ibm_kingston q146 on 14-15 Oct 2026.
Results come from Stim sampling of the compiled circuits.
"""
import itertools

import numpy as np
from qiskit_ibm_runtime.fake_provider import FakeKingston

from hhmem import ChipNoise, isa_to_stim

DEAD = 146
_JOBS = {}
_ids = itertools.count(1)


class _Truth(ChipNoise):
    def ro(self, q):
        return 0.5 if q == DEAD else super().ro(q)


class _Reg:
    def __init__(self, s):
        self._s = s

    def get_bitstrings(self):
        return self._s


class _Pub:
    def __init__(self, s):
        self.data = type("D", (), {"c": _Reg(s)})()


def _simulate(isas, noise, shots, seed=5):
    out = []
    for n, isa in enumerate(isas):
        pos = [isa.find_bit(i.clbits[0]).index for i in isa.data
               if i.operation.name == "measure"]
        c = isa_to_stim(isa, noise, [], {}, ro_mode="flip")
        m = c.compile_sampler(seed=seed + n).sample(shots)
        arr = np.zeros((shots, isa.num_clbits), bool)
        for j, cb in enumerate(pos):
            arr[:, cb] = m[:, j]
        out.append(_Pub(["".join("1" if b else "0" for b in r[::-1])
                         for r in arr]))
    return out


class _Job:
    def __init__(self, res):
        self._res, self._id = res, "fake-%03d" % next(_ids)
        _JOBS[self._id] = self

    def job_id(self):
        return self._id

    def status(self):
        return "DONE"

    def result(self):
        return self._res

    def usage(self):
        return 0.0


class FakeSampler:
    def __init__(self, mode):
        self.hw = mode

    def run(self, pubs, shots):
        return _Job(_simulate(pubs, _Truth(self.hw), shots))


class FakeService:
    def backend(self, name):
        return FakeKingston()

    def job(self, job_id):
        return _JOBS[job_id]

    def usage(self):
        return {"usage_remaining_seconds": 600}
