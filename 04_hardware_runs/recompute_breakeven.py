"""Break-even audit (15 Oct 2026): which physical-lifetime model is right?

The Z-basis physical reference qubits were prepared in |1> and idled with
X-X dynamical decoupling: X pulses at 1/4 and 3/4 of the wait
(hhmem.Chip.add_dd), so the qubit sits in |1> for the first and last
quarters and in |0> for the middle half. Pure amplitude damping with rate
1/T1 then gives, with d(x) = 1 - exp(-x/T1):

    P(1 at end) = [1 - d(t/4) (1 - d(t/2))] (1 - d(t/4))
    P(0 | prep 1) = a + s * (1 - P(1 at end))

where a = readout floor and s = contrast (1 - a - ro0). This script fits
T1 with that exact model and compares it with
  * the symmetric model used in hhmem2.tau_fit (1 - 2p = A exp(-t/tau)),
  * the no-DD model P(0) = a + s (1 - exp(-t/T1)) proposed in review.
The logical lifetime is defined by 1 - 2 p_L = A exp(-t/tau_L); for a
qubit averaged over |0>/|1> preparations that quantity decays as
exp(-t/T1), so T1 is the matching physical reference.

    IBM_QUANTUM_KEY=... python recompute_breakeven.py <plan.json> [...]
    python recompute_breakeven.py --selftest
"""
import json, os, sys
import numpy as np
from scipy.optimize import curve_fit

RO0 = (0.005, 0.01, 0.03)       # assumed |0> readout error (sensitivity)


def p_dd(t, T1):
    d = lambda x: 1 - np.exp(-x / T1)
    return 1 - (1 - d(t / 4) * (1 - d(t / 2))) * (1 - d(t / 4))


def fit_dd(ts, ps, ro0):
    ts, ps = np.asarray(ts, float), np.asarray(ps, float)
    f = lambda t, a, T: a + (1 - a - ro0) * p_dd(t, T)
    popt, _ = curve_fit(f, ts, ps, p0=(max(ps[0], 1e-3), 300.0),
                        bounds=([0, 1], [0.5, 1e5]), maxfev=20000)
    return float(popt[1])


def fit_nodd(ts, ps):
    ts, ps = np.asarray(ts, float), np.asarray(ps, float)
    f = lambda t, a, b, T: a + b * (1 - np.exp(-t / T))
    popt, _ = curve_fit(f, ts, ps, p0=(max(ps[0], 1e-3), 0.5, 300.0),
                        bounds=([0, 0, 1], [1.0, 1.2, 1e5]), maxfev=20000)
    return float(popt[2])


def boot(fn, ts, ps, N, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        try:
            out.append(fn(ts, rng.binomial(N, ps) / N))
        except (RuntimeError, ValueError):
            pass
    return [float(x) for x in np.percentile(out, [2.5, 97.5])]


def tau_sym(ts, ps, N):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(
        __file__)), "..", "03_code_hhmem"))
    from hhmem2 import tau_fit
    return float(tau_fit(ts, ps, N)[0])


def selftest():
    """Synthetic DD data with known T1: which model recovers it?"""
    ts = np.array([7.4, 12.7, 17.8, 22.9, 100, 200, 400])
    for T1 in (200, 300, 500):
        ps = 0.015 + (1 - 0.015 - 0.01) * p_dd(ts, T1)
        print("true T1 %d | exact-DD fit %.0f | symmetric fit %.0f | "
              "no-DD fit %.0f" % (T1, fit_dd(ts, ps, 0.01),
                                  tau_sym(ts, ps, 10000), fit_nodd(ts, ps)))


def main(paths):
    from qiskit_ibm_runtime import QiskitRuntimeService
    svc = QiskitRuntimeService(channel="ibm_quantum_platform",
                               token=os.environ["IBM_QUANTUM_KEY"])
    out = {}
    for p in paths:
        rec = json.load(open(p))
        res = svc.job(rec["job_id"]).result()
        N = rec["shots"]
        ph = sorted((x for x in rec["phys"] if x["basis"] == "Z"),
                    key=lambda x: x["us"])
        rows = {}
        for j, q in enumerate(ph[0]["qubits"]):
            ts, ps = [], []
            for x in ph:
                bs = res[x["k"]].data.c.get_bitstrings()
                bit = np.array([s[::-1][j] == "1" for s in bs])
                ps.append(float(1 - bit.mean()))          # P(0 | prep 1)
                ts.append(x["us"])
            if ps[0] > 0.2:
                print("%s q%d excluded: P(0|1) = %.2f at %.0f us (dead)"
                      % (rec["job_id"], q, ps[0], ts[0]))
                continue
            r = {"raw": [(round(t, 1), round(v, 4)) for t, v in zip(ts, ps, strict=True)],
                 "symmetric_tau": tau_sym(ts, ps, N),
                 "exact_dd_T1": {str(ro): fit_dd(ts, ps, ro) for ro in RO0},
                 "exact_dd_T1_ci": boot(lambda a, b: fit_dd(a, b, 0.01),
                                        ts, ps, N)}
            rows[q] = r
            print("%s Z q%-3d sym %.0f | exact-DD T1 %s us (ro0 .005/.01/.03),"
                  " 95%% [%.0f, %.0f] | max wait %.0f us"
                  % (rec["job_id"], q, r["symmetric_tau"],
                     "/".join("%.0f" % v for v in r["exact_dd_T1"].values()),
                     *r["exact_dd_T1_ci"], ts[-1]))
        out[rec["job_id"]] = rows
    return out


if __name__ == "__main__":
    if sys.argv[1:] == ["--selftest"]:
        selftest()
    else:
        res = main(sys.argv[1:])
        json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(
            __file__)), "breakeven_recheck_2026-10-15.json"), "w"), indent=1)
