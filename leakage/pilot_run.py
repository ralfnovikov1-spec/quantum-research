"""Run the pre-registered leakage pilot (PILOT_PREREGISTRATION.md).

    IBM_QUANTUM_KEY=... python leakage/pilot_run.py [--dry]

Writes leakage/pilot_results/<utc>_plan.json BEFORE submission, waits for the
job, saves the raw kerneled IQ (npz) and the analysis (json).
--dry builds and saves the plan without contacting IBM (FakeKingston).
"""
import datetime, json, os, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "daily"),
                os.path.join(HERE, "..", "03_code_hhmem")]
import numpy as np                                          # noqa: E402
import pilot                                                # noqa: E402

MIN_LEFT_S = 15
OUT = os.path.join(HERE, "pilot_results")


def main(dry):
    os.makedirs(OUT, exist_ok=True)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if dry:
        from qiskit_ibm_runtime.fake_provider import FakeKingston
        hw, svc = FakeKingston(), None
    else:
        from daily import ibm_key
        from qiskit_ibm_runtime import QiskitRuntimeService
        svc = QiskitRuntimeService(channel="ibm_quantum_platform", token=ibm_key())
        left = svc.usage().get("usage_remaining_seconds")
        print("IBM budget left:", left, "s")
        if left is None or left < MIN_LEFT_S:
            raise SystemExit("budget unknown or below %d s - not submitting" % MIN_LEFT_S)
        hw = svc.backend("ibm_kingston")
    pubs, plan = pilot.build(hw)
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                text=True, cwd=HERE).stdout.strip()
    except OSError:
        commit = "unknown"
    plan.update(prereg="leakage/PILOT_PREREGISTRATION.md", code_commit=commit,
                planned_utc=stamp)
    for a in plan["arms"]:
        a["layout"] = [int(x) for x in a["layout"]]
    fn = os.path.join(OUT, stamp + "_plan.json")
    json.dump(plan, open(fn, "w"), default=str)
    print("plan written:", fn, "| circuits", len(pubs), "| qubits", len(plan["cal_qubits"]))
    if dry:
        return
    from qiskit_ibm_runtime import SamplerV2
    sampler = SamplerV2(mode=hw)
    sampler.options.execution.meas_type = "kerneled"
    job = sampler.run(pubs, shots=pilot.SHOTS)
    plan["job_id"] = job.job_id()
    json.dump(plan, open(fn, "w"), default=str)
    print("SUBMITTED", plan["job_id"])
    res = job.result()
    np.savez_compressed(os.path.join(OUT, stamp + "_iq.npz"),
                        **{"pub%d" % k: np.asarray(res[k].data.c).astype(np.complex64)
                           for k in range(len(pubs))})
    out = pilot.analyse(res, plan)
    try:
        out["qpu_s"] = job.usage()
    except Exception:
        pass
    json.dump(out, open(os.path.join(OUT, stamp + "_result.json"), "w"),
              indent=1, default=str)
    print(json.dumps({k: out.get(k) for k in ("cal_false_flag_rate", "abort", "L0", "L1", "qpu_s")},
                     indent=1, default=str))


if __name__ == "__main__":
    main("--dry" in sys.argv)
