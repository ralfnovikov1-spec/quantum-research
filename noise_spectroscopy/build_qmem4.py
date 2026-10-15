"""Build Qmem4.ipynb (noise spectroscopy on IBM Heron). No key inside."""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ns = open(os.path.join(HERE, "noisespec.py"), encoding="utf-8").read()
out = sys.argv[1]


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": s}


cells = [
    md("# Qmem4 - What kind of noise kills the qubits? (DD noise spectroscopy)\n\n"
       "On every usable qubit of one IBM Heron chip, in parallel: T1, Ramsey, echo, "
       "CPMG-4 and CPMG-16 decays at 7 waits (2-280 us), 1000 shots each.\n\n"
       "**Pre-registered** (printed, hashed and saved to Drive before submission):\n"
       "- M1 echo T2 > Ramsey T2*: sign test p < 0.001 AND median ratio >= 1.2\n"
       "- M2 CPMG-16 T2 > echo T2: sign test p < 0.001 AND median ratio >= 1.1\n"
       "- M3 BENCHMARK: CPMG-16 T2 <= 2 T1 (+20%) on >= 95%\n"
       "- M4 median Ramsey decay exponent beta >= 1.3\n\n"
       "**Validated first in simulation (Qmem4_sim, 20 simulated chips per scenario):** "
       "white noise -> 0/20 false positives for M1, M2, M4; 1/f noise -> 20/20 detected "
       "(M4 only 13/20 for mixed noise, so an M4 FAIL alone does not mean 'no slow noise').\n\n"
       "**Limit, stated in advance:** this measures the noise *spectrum*. It cannot "
       "prove the noise is magnetic; slow noise also comes from two-level defects "
       "and charge noise.\n\n"
       "Cells: S1 setup - S2 module - S3 budget + plan (free) - "
       "S4 pre-register + submit (QPU, needs SUBMIT=True) - S5 analyse (free)."),
    code("# S1: setup - key from Colab secret, fallback = key already stored in Qmem2 (never printed)\n"
         "!pip install -q stim pymatching \"qiskit>=1.3\" qiskit-ibm-runtime rustworkx\n"
         "from google.colab import userdata, drive\n"
         "import os, re, json, glob, sys, hashlib, datetime\n"
         "try:\n"
         "    drive.mount(\"/content/drive\")\n"
         "    FOLDER = \"/content/drive/MyDrive/qmem_hw/noise_spectroscopy\"\n"
         "except Exception as e:\n"
         "    print(\"Drive not mounted (\", e, \") - saving locally; download results before closing\")\n"
         "    FOLDER = \"/content/noise_spectroscopy\"\n"
         "os.makedirs(FOLDER, exist_ok=True)\n\n"
         "def _key():\n"
         "    try:\n"
         "        return userdata.get(\"IBM_QUANTUM_KEY\"), \"Colab secret\"\n"
         "    except Exception:\n"
         "        pass\n"
         "    for fn in glob.glob(\"/content/drive/MyDrive/Colab Notebooks/Qmem2*\"):\n"
         "        nb = json.load(open(fn))\n"
         "        m = re.search(r'API_KEY\\s*=\\s*\"([^\"]{30,})\"',\n"
         "                      \"\\n\".join(\"\".join(c[\"source\"]) for c in nb[\"cells\"]))\n"
         "        if m:\n"
         "            return m.group(1), os.path.basename(fn)\n"
         "    raise RuntimeError(\"no key: add Colab secret IBM_QUANTUM_KEY\")\n\n"
         "_k, _src = _key()\n"
         "from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2\n"
         "service = QiskitRuntimeService(channel=\"ibm_quantum_platform\", token=_k)\n"
         "del _k\n"
         "print(\"service OK - key from\", _src)"),
    code("%%writefile noisespec.py\n" + ns),
    code("# S3: budget + plan (free, no QPU)\n"
         "import importlib, numpy as np, noisespec as N\n"
         "importlib.reload(N)\n"
         "u = service.usage()\n"
         "LEFT = u.get(\"usage_remaining_seconds\")\n"
         "print(\"IBM QPU budget left:\", LEFT, \"s (daily study keeps a 60 s reserve)\")\n"
         "hw = service.backend(\"ibm_kingston\")\n"
         "t = hw.target\n"
         "def ro(q):\n"
         "    p = t[\"measure\"].get((q,))\n"
         "    return p.error if p is not None and p.error is not None else 1.0\n"
         "QUBITS = [q for q in range(hw.num_qubits) if ro(q) <= 0.10 and q != 146]\n"
         "print(hw.name, \"| usable qubits (calibrated readout <= 10%, q146 excluded):\", len(QUBITS))\n"
         "CIRCS, INDEX = N.build(hw, QUBITS)\n"
         "print(len(CIRCS), \"circuits x\", N.SHOTS, \"shots\")"),
    code("# S4: PRE-REGISTER then submit (QPU). Set SUBMIT = True only when the budget allows.\n"
         "SUBMIT = False\n"
         "plan = {\"backend\": hw.name, \"qubits\": QUBITS, \"index\": INDEX,\n"
         "        \"shots\": N.SHOTS, \"predictions\": N.PREDICTIONS,\n"
         "        \"prereg_utc\": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec=\"seconds\")}\n"
         "plan[\"sha256\"] = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()\n"
         "PLAN = FOLDER + \"/spec_plan_%s.json\" % plan[\"prereg_utc\"].replace(\":\", \"\")\n"
         "print(\"PRE-REGISTERED\", plan[\"prereg_utc\"], plan[\"sha256\"][:16])\n"
         "for k, v in N.PREDICTIONS.items():\n"
         "    print(\" \", k, v)\n"
         "if SUBMIT:\n"
         "    json.dump(plan, open(PLAN, \"w\"))\n"
         "    job = SamplerV2(mode=hw).run(CIRCS, shots=N.SHOTS)\n"
         "    plan[\"job_id\"] = job.job_id()\n"
         "    json.dump(plan, open(PLAN, \"w\"))\n"
         "    print(\"SUBMITTED\", plan[\"job_id\"], \"->\", PLAN)\n"
         "else:\n"
         "    print(\"SUBMIT is False - nothing sent to IBM\")"),
    code("# S5: analyse + verdicts (free). Reloads the newest plan from Drive.\n"
         "PLAN = sorted(glob.glob(FOLDER + \"/spec_plan_*.json\"))[-1]\n"
         "plan = json.load(open(PLAN))\n"
         "job = service.job(plan[\"job_id\"])\n"
         "print(\"status:\", job.status(), \"- waiting...\")\n"
         "res = job.result()\n"
         "fits = N.analyse(res, [tuple(x) for x in plan[\"index\"]], plan[\"qubits\"])\n"
         "V = N.verdicts(fits)\n"
         "for k, (ok, txt) in V.items():\n"
         "    print(k, \"HOLDS\" if ok else \"FAILS\", \"-\", txt, \"|\", plan[\"predictions\"][k])\n"
         "med = {s: float(np.nanmedian([fits[q][s][0] for q in fits])) for s in N.SEQS}\n"
         "print(\"median T (us):\", {s: round(v) for s, v in med.items()})\n"
         "print(\"QPU seconds used:\", job.usage())\n"
         "plan[\"result\"] = {\"verdicts\": {k: [bool(a), b] for k, (a, b) in V.items()},\n"
         "                  \"fits\": {str(q): fits[q] for q in fits}, \"median_T_us\": med}\n"
         "json.dump(plan, open(PLAN, \"w\"))"),
]
nb = {"nbformat": 4, "nbformat_minor": 0,
      "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                   "language_info": {"name": "python"}},
      "cells": cells}
json.dump(nb, open(out, "w", encoding="utf-8"), indent=1)
print("cells", len(cells))
