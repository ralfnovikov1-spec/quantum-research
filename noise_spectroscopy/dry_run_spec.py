"""Offline control for noisespec: FakeKingston + Stim.

Stim's idle noise is a Markovian Pauli channel (white noise), so in this
simulator echo cannot beat Ramsey and decays are exponential (beta ~ 1).
Expected: M1 FAILS, M4 FAILS, M3 (physical bound) HOLDS. If M1 or M4 held
here, the analysis would be manufacturing the effect.
"""
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "03_code_hhmem"),
                os.path.join(HERE, "..", "daily")]
import numpy as np                                     # noqa: E402
from fake_ibm import _simulate, _Truth                 # noqa: E402
from qiskit_ibm_runtime.fake_provider import FakeKingston  # noqa: E402
import noisespec as N                                  # noqa: E402

hw = FakeKingston()
qubits = [q for q in range(hw.num_qubits) if q != 146][:40]
t = time.time()
circs, index = N.build(hw, qubits)
res = _simulate(circs, _Truth(hw), N.SHOTS)
fits = N.analyse(res, index, qubits)
print("circuits", len(circs), "| %.0fs" % (time.time() - t))
for q in qubits[:4]:
    print(" q%-3d" % q, {s: tuple(round(x, 2) for x in fits[q][s])
                         for s in N.SEQS})
for k, (ok, txt) in N.verdicts(fits).items():
    print(k, "HOLDS" if ok else "FAILS", "-", txt)
