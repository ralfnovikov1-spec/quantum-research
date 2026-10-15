"""EXPLORATORY (not pre-registered): per-detector hardware excess over the
calibration model vs probe-measured drift of the detector's qubits.

Data: 15 Oct 2026 rerun (job db88kdgr1c5c738lveeg) + its live probe
(db88jo6r6oqs7388vrs0) + IBM calibration as of the job time.
    IBM_QUANTUM_KEY=... python detector_drift.py <dir with rerun_plan + probe json>
"""
import glob, json, os, sys
import numpy as np, stim
from qiskit_ibm_runtime import QiskitRuntimeService

D = sys.argv[1]
rec = json.load(open(glob.glob(os.path.join(D, "rerun_plan_*.json"))[0]))
live = {int(q): v for q, v in json.load(open(glob.glob(
    os.path.join(D, "probe_*.json"))[0]))["live"].items()}
svc = QiskitRuntimeService(channel="ibm_quantum_platform",
                           token=os.environ["IBM_QUANTUM_KEY"])
job = svc.job(rec["job_id"])
res = job.result()
hw = svc.backend(rec["backend"])
props = hw.properties(datetime=job.creation_date)
def _get(f, q):
    try:
        return f(q)
    except Exception:          # IBM reports no value (e.g. dead q146)
        return None


cal = {q: {"t1": _get(props.t1, q), "ro": _get(props.readout_error, q)}
       for q in range(hw.num_qubits)}
missing = sorted(q for q, v in cal.items() if None in v.values())
print("qubits with no IBM calibration value:", missing)


def det_qubits(circ):
    """detector index -> set of physical qubits of its measurements."""
    meas_q, out = [], []
    for ins in circ.flattened():
        if ins.name in ("M", "MR"):
            meas_q += [t.value for t in ins.targets_copy()]
        elif ins.name == "DETECTOR":
            n = len(meas_q)
            out.append({meas_q[n + t.value] for t in ins.targets_copy()})
    return out


rows = []
for arm in rec["arms"]:
    circ = stim.Circuit(arm["stim"])
    bs = res[arm["k"]].data.c.get_bitstrings()
    arr = np.array([[c == "1" for c in reversed(s)] for s in bs])
    det = np.stack([arr[:, d].sum(1) % 2 == 1 for d in arm["dets"]], 1)
    det ^= np.array(arm["ref_d"], bool)
    hw_rate = det.mean(0)
    mod = circ.compile_detector_sampler(seed=4).sample(40000).mean(0)
    for j, qs in enumerate(det_qubits(circ)):
        if any(None in cal[q].values() for q in qs):
            continue
        d_ro = max(live[q]["ro"] - cal[q]["ro"] for q in qs)
        t1r = max(cal[q]["t1"] / max(live[q]["t1"], 1e-9) for q in qs)
        rows.append((hw_rate[j] - mod[j], d_ro, t1r, arm["basis"],
                     arm["variant"], tuple(sorted(qs))))

ex = np.array([r[0] for r in rows])
X = {"readout drift (live - cal)": np.array([r[1] for r in rows]),
     "T1 ratio (cal / live)": np.array([r[2] for r in rows])}


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


rng = np.random.default_rng(0)
print("detectors (all arms pooled):", len(ex))
print("hardware excess over calibration model: mean %+.4f, %d of %d positive"
      % (ex.mean(), (ex > 0).sum(), len(ex)))
out = {"n": len(ex), "mean_excess": float(ex.mean()),
       "frac_positive": float((ex > 0).mean())}
for name, x in X.items():
    rho = spearman(ex, x)
    bs = [spearman(ex[i], x[i]) for i in
          (rng.integers(0, len(ex), len(ex)) for _ in range(2000))]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    print("Spearman(excess, %s) = %+.3f  95%% CI [%+.3f, %+.3f]"
          % (name, rho, lo, hi))
    out[name] = [rho, float(lo), float(hi)]

# pseudo-replication check: one point per distinct qubit set (mean excess)
grp = {}
for r in rows:
    grp.setdefault(r[5], []).append(r)
g_ex = np.array([np.mean([r[0] for r in v]) for v in grp.values()])
g_ro = np.array([v[0][1] for v in grp.values()])
g_t1 = np.array([v[0][2] for v in grp.values()])
print("distinct qubit sets:", len(grp))
for name, x in (("readout drift", g_ro), ("T1 ratio", g_t1)):
    rho = spearman(g_ex, x)
    bs = [spearman(g_ex[i], x[i]) for i in
          (rng.integers(0, len(g_ex), len(g_ex)) for _ in range(4000))]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    perm = [spearman(g_ex, rng.permutation(x)) for _ in range(4000)]
    pval = float(np.mean(np.abs(perm) >= abs(rho)))
    print("GROUPED Spearman(excess, %s) = %+.3f  95%% CI [%+.3f, %+.3f]"
          "  permutation p = %.4f" % (name, rho, lo, hi, pval))
    out["grouped " + name] = [rho, float(lo), float(hi), pval, len(g_ex)]
json.dump(out, open(os.path.join(D, "detector_drift_2026-10-15.json"), "w"),
          indent=1)
