"""Daily calibration-staleness study (see daily/PROTOCOL.md).

Each run:
  1. collect: decode every submitted-but-unanalysed job that is DONE;
  2. submit:  if no job is pending and the IBM budget allows, probe the
              chip of the day, build baseline/reset-free arms, write the
              plan (with both model predictions) and submit one job;
  3. report:  regenerate daily/RESULTS.md from all analysed days.

    python daily/daily.py run        # real IBM (needs IBM_QUANTUM_KEY)
    python daily/daily.py fake       # offline end-to-end test (Stim)
"""
import datetime, glob, json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "03_code_hhmem"))
import hhmem2 as H                                      # noqa: E402
from hhmem import (COMPRESS, Chip, isa_to_stim, logical_error,  # noqa: E402
                   slope_fit)

DATA = os.path.join(HERE, "data")
CHIPS = ("ibm_kingston", "ibm_marrakesh")     # alternate by day of year
ROUNDS = (1, 2, 3)
SHOTS = 4000
RESERVE_S = 60          # never spend the last 60 s of the monthly budget
PROTOCOL_VERSION = 1


def chip_of_day(day=None):
    day = day or datetime.date.today()
    return CHIPS[day.toordinal() % len(CHIPS)]


def plan_day(hw, live, probe_id, cal_stale):
    """Same placement/filters as hhmem2, compression arms only."""
    chip = Chip(hw)
    lchip = H.LiveNoise(hw, live)
    bad = {q for q, v in live.items() if not v["usable"]}
    rec = {"protocol": PROTOCOL_VERSION, "backend": hw.name,
           "shots": SHOTS, "probe_job": probe_id,
           "date": datetime.date.today().isoformat(),
           "excluded": sorted(bad), "stale_t1_half": cal_stale,
           "arms": [], "pred": {}}
    pubs = []
    for basis in ("Z", "X"):
        L = H.place_filtered(chip, basis, {}, bad)
        for i, R in enumerate(ROUNDS):
            for v in ("baseline", "compressed")[::(-1) ** i]:
                kw = COMPRESS[basis] if v == "compressed" else {}
                isa, circ, arm = chip.build(basis, R, kw, L)
                arm.update(variant=v, k=len(pubs))
                lc = isa_to_stim(isa, lchip, arm["dets"], {0: arm["obs"]})
                arm["pred_cal"] = logical_error(circ, 30000, seed=9)
                arm["pred_live"] = logical_error(lc, 30000, seed=9)
                rec["arms"].append(arm)
                pubs.append(isa)
    for b in ("Z", "X"):
        for v in ("baseline", "compressed"):
            rows = sorted((a["rounds"], a["pred_cal"], a["pred_live"])
                          for a in rec["arms"]
                          if a["basis"] == b and a["variant"] == v)
            rs = [r[0] for r in rows]
            rec["pred"][b + "_" + v] = {
                "cal": float(slope_fit(rs, [r[1] for r in rows])[0]),
                "live": float(slope_fit(rs, [r[2] for r in rows])[0])}
    return rec, pubs


def analyze_day(res, rec):
    import pymatching, stim
    G = {}
    for arm in rec["arms"]:
        arr = H.bits(res, arm["k"])
        det = np.stack([np.logical_xor.reduce(arr[:, d], axis=1)
                        for d in arm["dets"]], axis=1)
        det ^= np.array(arm["ref_d"], bool)[None, :]
        ob = np.logical_xor.reduce(arr[:, arm["obs"]], axis=1)
        ob ^= bool(arm["ref_o"])
        m = pymatching.Matching.from_detector_error_model(
            H.dem_of(stim.Circuit(arm["stim"])))
        p = float(np.mean(m.decode_batch(det)[:, 0].astype(bool) != ob))
        G.setdefault(arm["basis"] + "_" + arm["variant"], []).append(
            (arm["rounds"], p))
    out = {"eps": {}, "gain": {}, "sep": {}}
    for k, rows in G.items():
        rows.sort()
        out["eps"][k] = [float(x) for x in slope_fit(
            [r[0] for r in rows], [r[1] for r in rows], rec["shots"])]
    for b in ("Z", "X"):
        eb, ec = out["eps"][b + "_baseline"], out["eps"][b + "_compressed"]
        out["gain"][b] = eb[0] / ec[0]
        out["sep"][b] = bool(ec[2] < eb[1])
    err = {"cal": [], "live": []}
    for k, (e, lo, hi) in out["eps"].items():
        for m in ("cal", "live"):
            err[m].append(abs(rec["pred"][k][m] - e) / e)
    out["model_err"] = {m: float(np.mean(v)) for m, v in err.items()}
    out["live_2x_better"] = bool(out["model_err"]["cal"]
                                 >= 2 * out["model_err"]["live"])
    return out


# ------------------------------------------------------------------
def collect(service):
    n = 0
    for fn in sorted(glob.glob(os.path.join(DATA, "*_plan.json"))):
        rec = json.load(open(fn))
        if "result" in rec or "job_id" not in rec:
            continue
        job = service.job(rec["job_id"])
        st = str(job.status()).upper()
        if "DONE" not in st:
            print("pending", rec["job_id"], st)
            if any(s in st for s in ("ERROR", "CANCEL")):
                rec["result"] = {"failed": st}
                json.dump(rec, open(fn, "w"), default=str)
            continue
        rec["result"] = analyze_day(job.result(), rec)
        try:
            rec["qpu_s"] = job.usage()
        except Exception:
            pass
        json.dump(rec, open(fn, "w"), default=str)
        print("collected", rec["job_id"], rec["result"]["gain"])
        n += 1
    return n


def pending():
    return [fn for fn in glob.glob(os.path.join(DATA, "*_plan.json"))
            if "result" not in json.load(open(fn))]


def submit(service, sampler_cls):
    if pending():
        print("a job is still pending - not submitting")
        return None
    try:
        u = service.usage()
        left = u.get("usage_remaining_seconds")
        print("IBM budget left:", left, "s")
        if left is not None and left < RESERVE_S:
            print("budget below reserve - skipping today")
            return None
    except Exception as e:
        print("usage unavailable:", e)
    hw = service.backend(chip_of_day())
    pcs, w = H.probe_circuits(hw)
    pj = sampler_cls(mode=hw).run(pcs, shots=H.PROBE_SHOTS)
    live = H.analyze_probe(pj.result(), hw, w)
    cal = H.ChipNoise(hw)
    stale = sum(live[q]["t1"] < cal.t1(q) / 2 for q in live)
    rec, pubs = plan_day(hw, live, pj.job_id(), stale)
    job = sampler_cls(mode=hw).run(pubs, shots=SHOTS)
    rec["job_id"] = job.job_id()
    os.makedirs(DATA, exist_ok=True)
    fn = os.path.join(DATA, "%s_%s_plan.json" % (rec["date"], hw.name))
    json.dump(rec, open(fn, "w"), default=str)
    print("submitted", rec["job_id"], "->", os.path.basename(fn))
    return fn


def report(out_dir=HERE):
    rows = []
    for fn in sorted(glob.glob(os.path.join(DATA, "*_plan.json"))):
        rec = json.load(open(fn))
        r = rec.get("result")
        if not r or "failed" in r:
            continue
        rows.append((rec, r))
    n = len(rows)
    wins = sum(r["live_2x_better"] for _, r in rows)
    zsep = sum(r["sep"]["Z"] and r["gain"]["Z"] > 1 for _, r in rows)
    lines = [
        "# Daily results (auto-generated, do not edit)", "",
        "Protocol: [PROTOCOL.md](PROTOCOL.md) - pre-registered claims C1, C2. "
        "Every row is one IBM hardware job; job IDs are in `data/`.", "",
        "| date | chip | excluded | live T1 < cal/2 | Z gain | X gain "
        "| model error cal | model error live | live 2x better | job |",
        "|---|---|---|---|---|---|---|---|---|---|"]
    for rec, r in rows:
        lines.append("| %s | %s | %d | %d | x%.2f%s | x%.2f%s | %.0f%% "
                     "| %.0f%% | %s | `%s` |" % (
                         rec["date"], rec["backend"], len(rec["excluded"]),
                         rec["stale_t1_half"],
                         r["gain"]["Z"], "" if r["sep"]["Z"] else "*",
                         r["gain"]["X"], "" if r["sep"]["X"] else "*",
                         100 * r["model_err"]["cal"],
                         100 * r["model_err"]["live"],
                         "yes" if r["live_2x_better"] else "no",
                         rec["job_id"]))
    lines += ["", "\\* = 2-sigma intervals overlap.", "",
              "## Running tally (decided only at %d days)" % MIN_DAYS, "",
              "- days analysed: **%d**" % n,
              "- C1 live model >= 2x more accurate: **%d / %d**"
              " (claim needs >= 80%% of days)" % (wins, n),
              "- C2 Z reset-free gain > 1, 2-sigma separated: **%d / %d**"
              " (claim needs >= 80%% of days)" % (zsep, n)]
    if n >= MIN_DAYS:
        lines += ["", "**Verdicts:** C1 %s, C2 %s" % (
            "HOLDS" if wins >= 0.8 * n else "FAILS",
            "HOLDS" if zsep >= 0.8 * n else "FAILS")]
    else:
        lines += ["", "No verdict before %d days (pre-registered)." % MIN_DAYS]
    open(os.path.join(out_dir, "RESULTS.md"), "w", encoding="utf-8").write(
        "\n".join(lines) + "\n")
    print("RESULTS.md:", n, "days")


MIN_DAYS = 20


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "run"
    if mode == "run":
        from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
        svc = QiskitRuntimeService(channel="ibm_quantum_platform",
                                   token=os.environ["IBM_QUANTUM_KEY"])
        collect(svc)
        submit(svc, SamplerV2)
        report()
    elif mode == "fake":
        from fake_ibm import FakeService, FakeSampler
        global DATA
        DATA = os.path.join(HERE, "_fake_data")
        svc = FakeService()
        submit(svc, FakeSampler)       # probe + plan + "submit"
        collect(svc)                   # decode the simulated job
        report(DATA)                   # never touches the real RESULTS.md
    else:
        raise SystemExit("usage: daily.py run|fake")


if __name__ == "__main__":
    main()
