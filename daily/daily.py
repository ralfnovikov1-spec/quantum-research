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
PROTOCOL_VERSION = 2


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
           "live": {str(q): {k: live[q][k] for k in ("t1", "t2", "ro")}
                    for q in live},
           "cal": {str(q): {"t1": chip.noise.t1(q), "t2": chip.noise.t2(q),
                            "ro": chip.noise.ro(q)} for q in live},
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
                arm["pred_live"] = H.pred_with_decoder(lc, circ, 30000, 9)
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


def _det_qubits(circ):
    meas_q, out = [], []
    for ins in circ.flattened():
        if ins.name in ("M", "MR"):
            meas_q += [t.value for t in ins.targets_copy()]
        elif ins.name == "DETECTOR":
            n = len(meas_q)
            out.append(tuple(sorted({meas_q[n + t.value]
                                     for t in ins.targets_copy()})))
    return out


def _spearman(a, b):
    """Spearman rho with average ranks for ties (tied readout drift is
    common: qubit sets that share their worst qubit)."""
    from scipy.stats import rankdata
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])


def detector_drift(rec, det_rates):
    """C3: per qubit-set mean (hardware - calibration-model) detector rate
    vs probe readout drift (live - cal), Spearman + permutation p."""
    import stim
    grp = {}
    for arm in rec["arms"]:
        circ = stim.Circuit(arm["stim"])
        mod = circ.compile_detector_sampler(seed=4).sample(40000).mean(0)
        hw = det_rates[arm["k"]]
        for j, qs in enumerate(_det_qubits(circ)):
            grp.setdefault(qs, []).append(hw[j] - mod[j])
    ex, dro = [], []
    for qs, v in grp.items():
        ex.append(float(np.mean(v)))
        dro.append(max(rec["live"][str(q)]["ro"] - rec["cal"][str(q)]["ro"]
                       for q in qs))
    ex, dro = np.array(ex), np.array(dro)
    rho = _spearman(ex, dro)
    rng = np.random.default_rng(0)
    perm = [_spearman(ex, rng.permutation(dro)) for _ in range(4000)]
    return {"rho": rho, "p": float(np.mean(np.abs(perm) >= abs(rho))),
            "n_sets": int(len(ex)), "mean_excess": float(ex.mean())}


def analyze_day(res, rec):
    import pymatching, stim
    G, det_rates = {}, {}
    for arm in rec["arms"]:
        arr = H.bits(res, arm["k"])
        det = np.stack([np.logical_xor.reduce(arr[:, d], axis=1)
                        for d in arm["dets"]], axis=1)
        det ^= np.array(arm["ref_d"], bool)[None, :]
        det_rates[arm["k"]] = det.mean(0)
        ob = np.logical_xor.reduce(arr[:, arm["obs"]], axis=1)
        ob ^= bool(arm["ref_o"])
        m = pymatching.Matching.from_detector_error_model(
            H.dem_of(stim.Circuit(arm["stim"])))
        p = float(np.mean(m.decode_batch(det)[:, 0].astype(bool) != ob))
        G.setdefault(arm["basis"] + "_" + arm["variant"], []).append(
            (arm["rounds"], p))
    out = {"eps": {}, "gain": {}, "sep": {}, "worse": {}}
    for k, rows in G.items():
        rows.sort()
        out["eps"][k] = [float(x) for x in slope_fit(
            [r[0] for r in rows], [r[1] for r in rows], rec["shots"])]
    for b in ("Z", "X"):
        eb, ec = out["eps"][b + "_baseline"], out["eps"][b + "_compressed"]
        out["gain"][b] = eb[0] / ec[0]
        out["sep"][b] = bool(ec[2] < eb[1])      # reset-free better, 2-sigma
        out["worse"][b] = bool(eb[2] < ec[1])    # reset-free worse, 2-sigma
    err = {"cal": [], "live": []}
    for k, (e, _lo, _hi) in out["eps"].items():
        for m in ("cal", "live"):
            err[m].append(abs(rec["pred"][k][m] - e) / e)
    out["model_err"] = {m: float(np.mean(v)) for m, v in err.items()}
    out["live_2x_better"] = bool(out["model_err"]["cal"]
                                 >= 2 * out["model_err"]["live"])
    if "live" in rec and "cal" in rec:
        out["c3"] = detector_drift(rec, det_rates)
        out["c3"]["pass"] = bool(out["c3"]["rho"] > 0
                                 and out["c3"]["p"] < 0.05)
    return out


# ------------------------------------------------------------------
def collect(service):
    n = 0
    for fn in sorted(glob.glob(os.path.join(DATA, "*_plan.json"))):
        rec = json.load(open(fn))
        if "result" in rec:
            continue
        if "job_id" not in rec:
            if rec.get("date") != datetime.date.today().isoformat():
                rec["result"] = {"failed": "never submitted"}
                json.dump(rec, open(fn, "w"), default=str)
            continue
        job = service.job(rec["job_id"])
        st = str(job.status()).upper()
        if "DONE" not in st:
            print("pending", rec["job_id"], st)
            if any(s in st for s in ("ERROR", "CANCEL")):
                rec["result"] = {"failed": st}
                json.dump(rec, open(fn, "w"), default=str)
            continue
        try:
            rec["result"] = analyze_day(job.result(), rec)
        except Exception as e:                 # recorded, never silent
            rec["result"] = {"failed": "analysis error: %r" % (e,)}
            json.dump(rec, open(fn, "w"), default=str)
            print("analysis failed", rec["job_id"], e)
            continue
        try:
            rec["qpu_s"] = job.usage()
        except Exception:
            pass
        json.dump(rec, open(fn, "w"), default=str)
        print("collected", rec["job_id"], rec["result"].get("gain"))
        n += 1
    return n


def pending():
    return [fn for fn in glob.glob(os.path.join(DATA, "*_plan.json"))
            if "result" not in json.load(open(fn))]


def submit(service, sampler_cls):
    if pending():
        print("a job is still pending - not submitting")
        return None
    today = datetime.date.today().isoformat()
    if glob.glob(os.path.join(DATA, today + "_*_plan.json")):
        print("today already has a plan - not submitting twice")
        return None
    try:
        left = service.usage().get("usage_remaining_seconds")
    except Exception as e:
        print("usage unavailable - not submitting (fail closed):", e)
        return None
    print("IBM budget left:", left, "s")
    if left is None or left < RESERVE_S:
        print("budget unknown or below reserve - skipping today")
        return None
    hw = service.backend(chip_of_day())
    pcs, w = H.probe_circuits(hw)
    pj = sampler_cls(mode=hw).run(pcs, shots=H.PROBE_SHOTS)
    live = H.analyze_probe(pj.result(), hw, w)
    cal = H.ChipNoise(hw)
    stale = sum(live[q]["t1"] < cal.t1(q) / 2 for q in live)
    rec, pubs = plan_day(hw, live, pj.job_id(), stale)
    os.makedirs(DATA, exist_ok=True)
    fn = os.path.join(DATA, "%s_%s_plan.json" % (rec["date"], hw.name))
    json.dump(rec, open(fn, "w"), default=str)    # predictions on disk first
    job = sampler_cls(mode=hw).run(pubs, shots=SHOTS)
    rec["job_id"] = job.job_id()
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
    v2 = [(rec, r) for rec, r in rows if rec.get("protocol", 1) >= 2]
    wins = sum(r["live_2x_better"] for _, r in v2)
    zsep = sum(r["sep"]["Z"] and r["gain"]["Z"] > 1 for _, r in rows)
    c3 = sum(r.get("c3", {}).get("pass", False) for _, r in rows)
    lines = [
        "# Daily results (auto-generated, do not edit)", "",
        "Protocol: [PROTOCOL.md](PROTOCOL.md) - pre-registered claims C1, C2. "
        "Every row is one IBM hardware job; job IDs are in `data/`.", "",
        "| date | chip | excluded | live T1 < cal/2 | Z gain | X gain "
        "| model error cal | model error live | live 2x better "
        "| C3 rho (p) | job |",
        "|---|---|---|---|---|---|---|---|---|---|---|"]
    for rec, r in rows:
        lines.append("| %s | %s | %d | %d | x%.2f%s | x%.2f%s | %.0f%% "
                     "| %.0f%% | %s | %s | `%s` |" % (
                         rec["date"], rec["backend"], len(rec["excluded"]),
                         rec["stale_t1_half"],
                         r["gain"]["Z"], _mark(r, "Z"),
                         r["gain"]["X"], _mark(r, "X"),
                         100 * r["model_err"]["cal"],
                         100 * r["model_err"]["live"],
                         ("yes" if r["live_2x_better"] else "no")
                         if rec.get("protocol", 1) >= 2 else "v1 (not counted)",
                         ("%+.2f (%.3f)" % (r["c3"]["rho"], r["c3"]["p"])
                          if "c3" in r else "n/a"),
                         rec["job_id"]))
    lines += ["", "\\* = 2-sigma intervals overlap; (worse) = reset-free "
              "significantly worse.", "",
              "## Running tally (decided only at %d days)" % MIN_DAYS, "",
              "- days analysed: **%d**" % n,
              "- C1 live model >= 2x more accurate: **%d / %d** protocol-v2"
              " days (claim needs >= 80%%)" % (wins, len(v2)),
              "- C2 Z reset-free gain > 1, 2-sigma separated: **%d / %d**"
              " (claim needs >= 80%% of days)" % (zsep, n),
              "- C3 detector excess tracks readout drift (rho > 0, p < 0.05):"
              " **%d / %d** (claim needs >= 60%% of days)" % (c3, n)]
    if n >= MIN_DAYS:
        lines += ["", "**Verdicts:** C1 %s, C2 %s, C3 %s" % (
            ("HOLDS" if wins >= 0.8 * len(v2) else "FAILS")
            if len(v2) >= MIN_DAYS else "pending (needs %d v2 days)" % MIN_DAYS,
            "HOLDS" if zsep >= 0.8 * n else "FAILS",
            "HOLDS" if c3 >= 0.6 * n else "FAILS")]
    else:
        lines += ["", "No verdict before %d days (pre-registered)." % MIN_DAYS]
    open(os.path.join(out_dir, "RESULTS.md"), "w", encoding="utf-8").write(
        "\n".join(lines) + "\n")
    print("RESULTS.md:", n, "days")


MIN_DAYS = 20


def _mark(r, b):
    if r["sep"][b]:
        return ""
    return " (worse)" if r.get("worse", {}).get(b) else "*"


def ibm_key():
    """IBM_QUANTUM_KEY may hold the bare key, a quoted key, or the JSON
    file IBM lets you download ({"apikey": "..."}). Never printed."""
    raw = os.environ["IBM_QUANTUM_KEY"].strip()
    if raw.startswith("{"):
        d = json.loads(raw)
        for k in ("apikey", "api_key", "apiKey", "token", "key"):
            if k in d:
                return str(d[k]).strip()
        raise SystemExit("IBM_QUANTUM_KEY is JSON without an apikey field "
                         "(fields: %s)" % sorted(d))
    return raw.strip("\"' ").strip()


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "run"
    if mode == "run":
        from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
        svc = QiskitRuntimeService(channel="ibm_quantum_platform",
                                   token=ibm_key())
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
