"""Independent re-derivation of every heavy-hex hardware number from IBM's
stored raw results. Own bit parsing, own fits; only the decoder model
(the arm's stored Stim circuit -> DEM -> PyMatching) is shared, because it
IS the definition of the experiment. Key from env IBM_QUANTUM_KEY. No QPU time: only job.result() / metadata of finished jobs."""
import glob, json, os, sys
import numpy as np, stim, pymatching
from qiskit_ibm_runtime import QiskitRuntimeService

PLANS = sys.argv[1]   # folder with hh_hw*.json plan files
OUT = sys.argv[2]     # folder with rerun_plan_*.json; result written here
key = os.environ["IBM_QUANTUM_KEY"]          # never hard-code a key
svc = QiskitRuntimeService(channel="ibm_quantum_platform", token=key)
del key


def bitarray(pub):
    """shots x clbits, column j = classical bit j (independent parser)."""
    bs = pub.data.c.get_bitstrings()
    n = len(bs[0])
    a = np.empty((len(bs), n), dtype=bool)
    for i, s in enumerate(bs):
        a[i] = [ch == "1" for ch in reversed(s)]
    return a


def decode(arr, arm):
    det = np.zeros((arr.shape[0], len(arm["dets"])), bool)
    for j, d in enumerate(arm["dets"]):
        det[:, j] = arr[:, d].sum(1) % 2 == 1
    if "ref_d" in arm:
        det ^= np.array(arm["ref_d"], bool)
    ob = arr[:, arm["obs"]].sum(1) % 2 == 1
    if "ref_o" in arm:
        ob ^= bool(arm["ref_o"])
    dem = stim.Circuit(arm["stim"]).detector_error_model(
        decompose_errors=True, ignore_decomposition_failures=True,
        approximate_disjoint_errors=True)
    m = pymatching.Matching.from_detector_error_model(dem)
    pred = m.decode_batch(det)[:, 0].astype(bool)
    return float(np.mean(pred != ob))


def eps_fit(rs, ps, N):
    """Weighted LS of ln(1-2p) = a + r ln(1-2e); 2-sigma from the weights."""
    rs, ps = np.asarray(rs, float), np.asarray(ps, float)
    F = 1 - 2 * ps
    if np.any(F <= 0):
        return float("nan"), float("nan"), float("nan")
    y = np.log(F)
    var = 4 * ps * (1 - ps) / N / F ** 2          # delta method
    W = np.diag(1 / var)
    X = np.c_[np.ones_like(rs), rs]
    cov = np.linalg.inv(X.T @ W @ X)
    b = cov @ X.T @ W @ y
    s = np.sqrt(cov[1, 1])
    e = lambda t: (1 - np.exp(t)) / 2
    return e(b[1]), e(b[1] + 2 * s), e(b[1] - 2 * s)


rows, meta = [], []
plans = sorted(glob.glob(os.path.join(PLANS, "hh_hw*.json")))
plans += glob.glob(os.path.join(OUT, "rerun_plan_*.json"))
for fn in plans:
    rec = json.load(open(fn))
    job = svc.job(rec["job_id"])
    res = job.result()
    shots_actual = sorted({len(res[k].data.c.get_bitstrings())
                           for k in range(len(res))})
    m = {"job": rec["job_id"], "file": os.path.basename(fn),
         "backend_ibm": job.backend().name, "backend_plan": rec["backend"],
         "created": str(job.creation_date), "status": str(job.status()),
         "pubs": len(res), "shots_plan": rec["shots"],
         "shots_actual": shots_actual, "qpu_s": job.usage()}
    meta.append(m)
    print(m, flush=True)
    G = {}
    for i, arm in enumerate(rec["arms"]):
        k = arm.get("k", i)
        if "stim" not in arm:
            continue
        p = decode(bitarray(res[k]), arm)
        R = arm.get("rounds", rec.get("rounds"))
        dd = arm.get("dd")             # False / True / "xx" / "stag"
        dd = ("DD-" + dd if isinstance(dd, str) else
              "DD" if dd else "noDD")
        grp = arm.get("variant") or "%s/%s" % (arm.get("layout_name"), dd)
        G.setdefault((arm["basis"], grp), []).append((R, p))
    N = shots_actual[0]
    for (b, g), pts in sorted(G.items()):
        pts.sort()
        rs = [r for r, _ in pts]
        ps = [p for _, p in pts]
        if len(set(rs)) >= 2:
            e = eps_fit(rs, ps, N)
            kind = "eps_fit"
        else:
            R, p = rs[0], ps[0]
            e = ((1 - (1 - 2 * p) ** (1 / R)) / 2, float("nan"), float("nan"))
            kind = "eps_1pt(F-def) p=%.4f old_def=%.4f" % (
                p, 1 - (1 - min(p, .4999)) ** (1 / R))
        rows.append({"job": rec["job_id"], "basis": b, "arm": g,
                     "rounds": rs, "pL": [round(x, 5) for x in ps],
                     "eps": e, "kind": kind})
        print("  %s %-22s eps %.2f%% [%.2f, %.2f]  %s" % (
            b, g, e[0] * 100, e[1] * 100, e[2] * 100, kind), flush=True)

json.dump({"meta": meta, "rows": rows},
          open(os.path.join(OUT, "INDEPENDENT_REDERIVATION.json"), "w"),
          indent=1, default=str)
print("saved")
