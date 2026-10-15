"""Probe-informed decoding (pre-registered in PREREGISTRATION.md).

Same hardware shots, two decoders: the stored calibration Stim model, and
the same model with every measurement's readout error replaced by the live
probe's value (X_ERROR(p/2) before M, record flip M(p/2), as in hhmem).

    IBM_QUANTUM_KEY=... python probe_decoding.py <rerun_plan.json> <probe.json>
"""
import json, os, sys
from math import comb

import numpy as np
import pymatching
import stim

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "03_code_hhmem"))
from hhmem import dem_of                                      # noqa: E402


def relive_readout(circ, ro):
    """Copy of circ with each (X_ERROR q, M q) readout pair re-parameterised
    to the probe's readout error ro[q]; all other noise unchanged."""
    ops = list(circ.flattened())
    out, i, n_re = stim.Circuit(), 0, 0
    while i < len(ops):
        a = ops[i]
        b = ops[i + 1] if i + 1 < len(ops) else None
        if (a.name == "X_ERROR" and b is not None and b.name == "M"
                and len(a.targets_copy()) == 1
                and [t.value for t in a.targets_copy()]
                == [t.value for t in b.targets_copy()]):
            q = a.targets_copy()[0].value
            p = min(max(ro[q], 1e-4), 0.45) / 2
            out.append("X_ERROR", [q], p)
            out.append("M", [q], p)
            i += 2
            n_re += 1
            continue
        out.append(a)
        i += 1
    return out, n_re


def bits(res, k):
    bs = res[k].data.c.get_bitstrings()
    raw = np.frombuffer("".join(bs).encode(), np.uint8)
    return (raw.reshape(len(bs), -1) - 48)[:, ::-1].astype(bool)


def main(plan_path, probe_path):
    from qiskit_ibm_runtime import QiskitRuntimeService
    rec = json.load(open(plan_path))
    live = {int(q): v["ro"] for q, v in json.load(open(probe_path))["live"].items()}
    svc = QiskitRuntimeService(channel="ibm_quantum_platform",
                               token=os.environ["IBM_QUANTUM_KEY"])
    res = svc.job(rec["job_id"]).result()
    rows, B, C = [], 0, 0
    for arm in rec["arms"]:
        arr = bits(res, arm["k"])
        det = np.stack([np.logical_xor.reduce(arr[:, d], axis=1)
                        for d in arm["dets"]], axis=1)
        det ^= np.array(arm["ref_d"], bool)[None, :]
        ob = np.logical_xor.reduce(arr[:, arm["obs"]], axis=1)
        ob ^= bool(arm["ref_o"])
        cal = stim.Circuit(arm["stim"])
        liv, n_re = relive_readout(cal, live)
        wrong = {}
        for name, c in (("cal", cal), ("probe", liv)):
            m = pymatching.Matching.from_detector_error_model(dem_of(c))
            wrong[name] = m.decode_batch(det)[:, 0].astype(bool) != ob
        b = int((wrong["cal"] & ~wrong["probe"]).sum())   # probe fixes
        c = int((~wrong["cal"] & wrong["probe"]).sum())   # probe breaks
        B, C = B + b, C + c
        rows.append({"basis": arm["basis"], "variant": arm["variant"],
                     "rounds": arm["rounds"], "readouts_replaced": n_re,
                     "err_cal": int(wrong["cal"].sum()),
                     "err_probe": int(wrong["probe"].sum()),
                     "probe_fixes": b, "probe_breaks": c})
        print("%s %-10s r%d  errors cal %4d  probe %4d  (fixes %d, breaks %d,"
              " readouts re-parameterised %d)" % (
                  arm["basis"], arm["variant"], arm["rounds"],
                  rows[-1]["err_cal"], rows[-1]["err_probe"], b, c, n_re))
    n = B + C
    p = sum(comb(n, i) for i in range(B, n + 1)) / 2 ** n if n else 1.0
    better = sum(r["err_probe"] < r["err_cal"] for r in rows)
    tot_cal = sum(r["err_cal"] for r in rows)
    tot_probe = sum(r["err_probe"] for r in rows)
    red = (tot_cal - tot_probe) / tot_cal
    V = {"D1": better >= 12, "D2": p < 0.01, "D3": red >= 0.02}
    print("\narms with fewer logical errors under the probe decoder: %d / %d"
          % (better, len(rows)))
    print("pooled McNemar: probe fixes %d, breaks %d, one-sided p = %.2e"
          % (B, C, p))
    print("pooled logical errors: calibration %d, probe %d -> reduction %.2f%%"
          % (tot_cal, tot_probe, 100 * red))
    for k, ok in V.items():
        print(k, "HOLDS" if ok else "FAILS")
    json.dump({"rows": rows, "mcnemar": [B, C, p], "arms_better": better,
               "reduction": red, "verdicts": V},
              open(os.path.join(HERE, "result_2026-10-15.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
