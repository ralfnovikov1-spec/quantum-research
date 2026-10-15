"""Build Qmem4_sim.ipynb: power test of the noise-spectroscopy analysis.
Simulation only - no Google Drive, no IBM account, no API key."""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ns = open(os.path.join(HERE, "noisespec.py"), encoding="utf-8").read()
sim = open(os.path.join(HERE, "noisesim.py"), encoding="utf-8").read()
out = sys.argv[1]


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": s}


cells = [
    md("# Qmem4_sim - Can our noise-spectroscopy test be trusted? (SIMULATION ONLY)\n\n"
       "**Everything in this notebook is SIMULATION.** No IBM hardware, no Google Drive, "
       "no API key. It says nothing about real qubits. It only answers: if a chip has "
       "slow (1/f) noise (from two-level defects, or flux noise in tunable couplers), "
       "does our pre-registered test detect it, and does it stay silent when there is none?\n\n"
       "Simulated qubits have **known** noise: white (Markovian) dephasing, 1/f noise from "
       "telegraph fluctuators, or a mix; optionally with CONFOUNDS that are not noise but could "
       "fake a signal: static frequency offsets (+-5 kHz), static ZZ coupling (<= 1 kHz, 2 "
       "neighbours) and asymmetric readout. T1 150-400 us, 1000 shots, raw bitstrings run through "
       "the same analyse() and verdict code as the hardware notebook (Qmem4).\n\n"
       "Required before any hardware run: **0 false positives** for M1/M2/M4 under white noise "
       "with and without confounds, and >= 19/20 detection of 1/f noise by M1 and M2. "
       "T3 is a control: the old X-only Ramsey must FAIL that requirement under frequency offsets."),
    code("# T0: install (no account needed)\n"
         "!pip install -q \"qiskit>=1.3\"\n"
         "import qiskit; print('qiskit', qiskit.__version__)"),
    code("%%writefile noisespec.py\n" + ns),
    code("%%writefile noisesim.py\n" + sim),
    code("# T1: one simulated chip per scenario (SIMULATION)\n"
         "import importlib, numpy as np, noisespec as N, noisesim as S\n"
         "importlib.reload(N); importlib.reload(S)\n"
         "SCEN = ('white', 'white+confounds', '1/f', '1/f+confounds', 'mixed', 'mixed+confounds')\n"
         "for name in SCEN:\n"
         "    f = S.scenario(name, n_qubits=120, seed=1)\n"
         "    med = {s: round(float(np.nanmedian([f[q][s][0] for q in f]))) for s in N.FITTED}\n"
         "    print('==', name, '| median fitted T (us):', med)\n"
         "    for k, (ok, txt) in N.verdicts(f).items():\n"
         "        print('   ', k, 'HOLDS' if ok else 'FAILS', '-', txt)"),
    code("# T2: false-positive and detection rates over 20 independent simulated chips (SIMULATION, ~6-10 min)\n"
         "import time\n"
         "t0 = time.time()\n"
         "RATES = {}\n"
         "for name in SCEN:\n"
         "    hits = {k: 0 for k in N.PREDICTIONS}\n"
         "    for seed in range(20):\n"
         "        for k, (ok, _) in N.verdicts(S.scenario(name, n_qubits=120, seed=100 + seed)).items():\n"
         "            hits[k] += ok\n"
         "    RATES[name] = hits\n"
         "    print('%-16s' % name, '  '.join('%s %2d/20' % (k, v) for k, v in hits.items()), flush=True)\n"
         "print('%.0f s' % (time.time() - t0))\n"
         "fp = sum(RATES[n][k] for n in ('white', 'white+confounds') for k in ('M1', 'M2', 'M4'))\n"
         "det = min(RATES[n][k] for n in ('1/f', '1/f+confounds') for k in ('M1', 'M2'))\n"
         "print()\n"
         "print('false positives, white +/- confounds (M1, M2, M4):', fp, '-> REQUIRED 0:', 'PASS' if fp == 0 else 'FAIL')\n"
         "print('worst detection of 1/f +/- confounds (M1, M2): %d/20 -> REQUIRED >= 19:' % det, 'PASS' if det >= 19 else 'FAIL')"),
    code("# T3: CONTROL - the old X-only Ramsey (no |<X>,<Y>| magnitude) under frequency offsets (SIMULATION)\n"
         "real = N.np.hypot\n"
         "N.np.hypot = lambda a, b: a\n"
         "try:\n"
         "    fp_old = sum(N.verdicts(S.scenario('white+confounds', seed=200 + s))['M1'][0] for s in range(10))\n"
         "finally:\n"
         "    N.np.hypot = real\n"
         "print('old X-only Ramsey, white noise + offsets: M1 false positives %d/10' % fp_old)\n"
         "print('control', 'WORKS (the fix is necessary)' if fp_old > 0 else 'shows no difference')"),
]
nb = {"nbformat": 4, "nbformat_minor": 0,
      "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                   "language_info": {"name": "python"}},
      "cells": cells}
json.dump(nb, open(out, "w", encoding="utf-8"), indent=1)
print("cells", len(cells))
