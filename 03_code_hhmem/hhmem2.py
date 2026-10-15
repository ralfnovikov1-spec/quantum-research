"""hhmem2: hhmem + live probe, dead-qubit filter, clean break-even test.

Fixes the defects found in the 15 Oct 2026 audit of the 14 Oct run:
  * the probe could rank a dead qubit (q146, ~50% readout) as the best
    physical qubit -> usable() rejects it BEFORE any selection;
  * break-even took the best of two logical arms after the fact -> one
    logical arm per basis is fixed in advance;
  * physical waits stopped at ~29 us against 300-500 us lifetimes ->
    waits now reach 400 us;
  * predictions are written to disk with a UTC stamp and a hash before
    the job is submitted, and every verdict prints HOLDS or FAILS.

Colab (Qmem3, cells R1-R4):
    live = H.analyze_probe(probe_result, hw, w)       # probe job
    rec, pubs = H.build_plan(hw, live, probe_job_id)  # no QPU
    H.preregister(rec, FOLDER)                        # before submit
    out = H.analyze(result, rec); H.report(out, rec)  # after DONE
"""
import datetime, hashlib, json, os

import numpy as np
from qiskit import QuantumCircuit
from qiskit.transpiler.preset_passmanagers import \
    generate_preset_pass_manager

from hhmem import (COMPRESS, Chip, ChipNoise, candidate_layouts,
                   code_graph, dem_of, isa_to_stim, logical_error,
                   make_variant, model_score, role_counts, slope_fit)
import pymatching
import stim

# ---- pre-registered constants (do not change after a plan is saved) --
RO_MAX = 0.10          # mean readout error above this = unusable
RO_EACH_MAX = 0.15     # either |0> or |1> readout above this = unusable
T1_MAX = 2e-3          # live T1 above 2 ms on Heron = fit artefact
PROBE_WAIT = 20e-6
PROBE_SHOTS = 3000
SHOTS = 10000
ROUNDS = (1, 2, 3, 4)
PHYS_EXTRA_US = (100, 200, 400)
RATIO_RANGE = (0.10, 0.35)
MODEL_TOL = 0.30


def pred_with_decoder(sample_circ, decode_circ, shots, seed):
    """Logical error when shots come from sample_circ (e.g. probe noise) but
    are decoded with decode_circ's model - exactly what happens to hardware
    data, which is always decoded with the calibration model. (The 15 Oct
    rerun's P21 used the probe circuit's own model; see the corrected report.)"""
    m = pymatching.Matching.from_detector_error_model(dem_of(decode_circ))
    det, ob = sample_circ.compile_detector_sampler(seed=seed).sample(
        shots, separate_observables=True)
    return float(np.mean(np.any(m.decode_batch(det) != ob, axis=1)))


def bits(res, k):
    """Bit array (shots x clbits), clbit 0 in column 0."""
    bs = res[k].data.c.get_bitstrings()
    raw = np.frombuffer("".join(bs).encode(), np.uint8)
    return (raw.reshape(len(bs), -1) - 48)[:, ::-1].astype(bool)


def _full_pm(hw):
    return generate_preset_pass_manager(
        optimization_level=1, target=hw.target,
        initial_layout=list(range(hw.num_qubits)))


# ------------------------------------------------------------------
# live probe: readout, T1 and echo T2 of every qubit in one job
# ------------------------------------------------------------------
def probe_circuits(hw):
    nq = hw.num_qubits
    w = int(PROBE_WAIT / hw.dt) // 32 * 32
    a = QuantumCircuit(nq, nq)
    a.measure(range(nq), range(nq))
    b = QuantumCircuit(nq, 2 * nq)
    b.x(range(nq))
    b.measure(range(nq), range(nq))
    b.delay(w, range(nq), unit="dt")
    b.measure(range(nq), range(nq, 2 * nq))
    c = QuantumCircuit(nq, nq)
    c.h(range(nq))
    c.delay(w // 2, range(nq), unit="dt")
    c.x(range(nq))
    c.delay(w // 2, range(nq), unit="dt")
    c.h(range(nq))
    c.measure(range(nq), range(nq))
    pm = _full_pm(hw)
    return [pm.run(x) for x in (a, b, c)], w


def analyze_probe(res, hw, w):
    """q -> dict(t1, t2, ro, ro0, ro1, usable, why). Same estimators
    as the 14 Oct probe, plus the usability rule."""
    nq = hw.num_qubits
    A, B, C = bits(res, 0), bits(res, 1), bits(res, 2)
    ro0, ro1 = A.mean(0), 1 - B[:, :nq].mean(0)
    ech = C.mean(0)
    tw, out = w * hw.dt, {}
    for q in range(nq):
        sel = B[:, q]
        dec = float(np.mean(~B[sel, nq + q])) if sel.any() else 0.5
        ro = 0.5 * (ro0[q] + ro1[q])
        d_raw = dec - ro1[q]
        d = min(max(d_raw, 1e-4), 0.99)
        t1 = -tw / np.log(1 - d)
        e_raw = ech[q] - ro
        e = min(max(e_raw, 1e-4), 0.49)
        t2 = min(-tw / np.log(1 - 2 * e), 2 * t1)
        why = []
        if ro > RO_MAX:
            why.append("readout %.1f%%" % (100 * ro))
        if max(ro0[q], ro1[q]) > RO_EACH_MAX:
            why.append("one-sided readout %.1f/%.1f%%"
                       % (100 * ro0[q], 100 * ro1[q]))
        if d_raw <= 1e-4 or t1 > T1_MAX:
            why.append("T1 not identifiable")
        out[q] = {"t1": float(t1), "t2": float(t2), "ro": float(ro),
                  "ro0": float(ro0[q]), "ro1": float(ro1[q]),
                  "usable": not why, "why": why}
    return out


class LiveNoise(ChipNoise):
    """Calibration gate errors, probe T1/T2/readout per qubit."""

    def __init__(self, backend, live):
        super().__init__(backend)
        self.live = live

    def t1(self, q):
        return self.live[q]["t1"]

    def t2(self, q):
        return self.live[q]["t2"]

    def ro(self, q):
        return float(min(max(self.live[q]["ro"], 1e-4), .45))


# ------------------------------------------------------------------
# placement that never touches an unusable qubit
# ------------------------------------------------------------------
def place_filtered(chip, basis, kw, bad, top=8, seed=7):
    qc, dets, fl, obs, S, _ = make_variant(3, 3, basis, **kw)
    cg, uses = code_graph(qc)
    meas, rst = role_counts(qc)
    pool = [L for L in candidate_layouts(chip.hw, cg,
                                         np.random.default_rng(seed))
            if not bad.intersection(L)]
    sc = lambda L: model_score(L, uses, meas, rst, chip.noise,
                               S["n"], 3 * 5e-6)
    best = None
    for L in sorted(pool, key=sc)[:top]:
        try:
            _, circ, _ = chip.build(basis, 3, kw, L)
        except Exception:
            continue
        p = logical_error(circ, 20000, seed=11)
        if best is None or p < best[0]:
            best = (p, L)
    if best is None:
        raise RuntimeError("no usable zero-SWAP placement for " + basis)
    return best[1]


def physical_circuit(chip, pm, qubits, basis, wait_us):
    hw = chip.hw
    T = int(wait_us * 1e-6 / hw.dt) // 32 * 32
    qc = QuantumCircuit(hw.num_qubits, len(qubits))
    for j, q in enumerate(qubits):
        if basis == "X":
            qc.h(q)
        else:
            qc.x(q)
        qc.delay(T, q, unit="dt")
        if basis == "X":
            qc.h(q)
        qc.measure(q, j)
    isa, _ = chip.add_dd(chip.sched(pm.run(qc)))
    return isa, T * hw.dt * 1e6


# ------------------------------------------------------------------
# plan: probe -> filter -> placement -> arms -> pre-registration
# ------------------------------------------------------------------
PREDICTIONS = {
    "P17": "compressed eps < baseline eps in both bases, 2-sigma "
           "intervals separated (plain matching, %d shots)" % SHOTS,
    "P18": "break-even NOT reached in either basis: tau_logical / "
           "tau_physical upper 2-sigma bound < 1",
    "P19": "ratio tau_logical / tau_physical inside %.2f-%.2f in both "
           "bases" % RATIO_RANGE,
    "P20": "no physical reference qubit errs above 10% at its "
           "shortest wait (dead-qubit filter works)",
    "P21": "live-probe model eps within +-%d%% of hardware for all 4 "
           "arms (rounds 1-4 fit)" % int(MODEL_TOL * 100),
}
RULES = ("Logical arm for break-even = the 'compressed' arm of each "
         "basis. Physical reference = the 3 usable qubits with the "
         "longest probe T1 (Z) / T2 (X); the best FITTED lifetime of "
         "the 3 is used. Unusable = mean readout > %.0f%%, either "
         "readout > %.0f%%, or T1 not identifiable / > %.0f us."
         % (RO_MAX * 100, RO_EACH_MAX * 100, T1_MAX * 1e6))


def build_plan(hw, live, probe_job=None):
    chip = Chip(hw)                      # calibration noise scores
    lchip = LiveNoise(hw, live)
    bad = {q for q, v in live.items() if not v["usable"]}
    pm = _full_pm(hw)
    rec = {"backend": hw.name, "shots": SHOTS, "probe_job": probe_job,
           "excluded": {str(q): live[q]["why"] for q in sorted(bad)},
           "arms": [], "phys": [], "pred": {}}
    pubs = []
    for basis in ("Z", "X"):
        L = place_filtered(chip, basis, {}, bad)
        for i, R in enumerate(ROUNDS):
            order = ("baseline", "compressed")[::(-1) ** i]
            for v in order:
                kw = COMPRESS[basis] if v == "compressed" else {}
                isa, circ, arm = chip.build(basis, R, kw, L)
                arm["variant"] = v
                arm["k"] = len(pubs)
                lc = isa_to_stim(isa, lchip, arm["dets"],
                                 {0: arm["obs"]})
                arm["pred_cal"] = logical_error(circ, 30000, seed=9)
                arm["pred_live"] = pred_with_decoder(lc, circ, 30000, 9)
                rec["arms"].append(arm)
                pubs.append(isa)
        key = 0 if basis == "Z" else 1   # Z: T1 ranks, X: echo T2
        best = sorted((q for q in live if q not in bad),
                      key=lambda q: -live[q]["t1" if key == 0 else "t2"])[:3]
        waits = sorted({a["us"] for a in rec["arms"]
                        if a["basis"] == basis
                        and a["variant"] == "compressed"}
                       | set(PHYS_EXTRA_US))
        for wu in waits:
            isa, real_us = physical_circuit(chip, pm, best, basis, wu)
            rec["phys"].append({"basis": basis, "qubits": best,
                                "us": real_us, "k": len(pubs)})
            pubs.append(isa)
    for b in ("Z", "X"):
        for v in ("baseline", "compressed"):
            rows = sorted((a["rounds"], a["pred_cal"], a["pred_live"])
                          for a in rec["arms"]
                          if a["basis"] == b and a["variant"] == v)
            rec["pred"][b + "_" + v] = {
                "cal": slope_fit([r[0] for r in rows],
                                 [r[1] for r in rows])[0],
                "live": slope_fit([r[0] for r in rows],
                                  [r[2] for r in rows])[0]}
    rec["predictions"] = PREDICTIONS
    rec["rules"] = RULES
    return rec, pubs


def preregister(rec, folder):
    """Stamp, hash and save the plan BEFORE submission."""
    rec["prereg_utc"] = datetime.datetime.now(
        datetime.timezone.utc).isoformat(timespec="seconds")
    core = {k: rec[k] for k in ("backend", "shots", "predictions",
                                "rules", "excluded", "pred")}
    core["layouts"] = sorted({(a["basis"], tuple(a["layout"]))
                              for a in rec["arms"]})
    core["phys_qubits"] = sorted({(p["basis"], tuple(p["qubits"]))
                                  for p in rec["phys"]})
    rec["prereg_sha256"] = hashlib.sha256(
        json.dumps(core, sort_keys=True, default=str).encode()).hexdigest()
    os.makedirs(folder, exist_ok=True)
    fn = os.path.join(folder, "rerun_plan_%s.json"
                      % rec["prereg_utc"].replace(":", ""))
    json.dump(rec, open(fn, "w"), default=str)
    print("PRE-REGISTERED", rec["prereg_utc"], "sha256",
          rec["prereg_sha256"][:16], "->", fn)
    for k, v in PREDICTIONS.items():
        print("  %s %s" % (k, v))
    print("  rules:", RULES)
    print("  excluded qubits:", len(rec["excluded"]),
          sorted(int(q) for q in rec["excluded"])[:20])
    return fn


# ------------------------------------------------------------------
# analysis
# ------------------------------------------------------------------
def tau_fit(ts_us, ps, N):
    """Lifetime (us) and 2-sigma range from F(t) = A (1 - 2e)^t.

    Valid for errors that saturate at 1/2: logical qubits, |+> references,
    and |1> references idled with this module's X-X DD (pulses at 1/4 and
    3/4 of each wait). For the DD case it recovers T1 within ~5% (checked
    in 04_hardware_runs/recompute_breakeven.py). It is NOT valid for a |1>
    reference idling without DD (error -> 1): there it returns ~0.4 T1."""
    e, lo, hi = slope_fit(ts_us, ps, N)
    f = lambda x: (-1 / np.log(1 - 2 * x)) if 0 < x < 0.5 else float("inf")
    return f(e), f(hi), f(lo)      # lo <= 0 -> upper lifetime bound = inf


def analyze(res, rec):
    N = rec["shots"]
    out = {"eps": {}, "tau_L": {}, "phys": {}, "verdict": {}}
    G = {}
    for arm in rec["arms"]:
        arr = bits(res, arm["k"])
        det = np.stack([np.logical_xor.reduce(arr[:, d], axis=1)
                        for d in arm["dets"]], axis=1)
        det ^= np.array(arm["ref_d"], bool)[None, :]
        ob = np.logical_xor.reduce(arr[:, arm["obs"]], axis=1)
        ob ^= bool(arm["ref_o"])
        m = pymatching.Matching.from_detector_error_model(
            dem_of(stim.Circuit(arm["stim"])))
        p = float(np.mean(m.decode_batch(det)[:, 0].astype(bool) != ob))
        G.setdefault((arm["basis"], arm["variant"]), []).append(
            (arm["rounds"], p, arm["us"]))
    for (b, v), rows in G.items():
        rows.sort()
        out["eps"][b + "_" + v] = slope_fit([r[0] for r in rows],
                                            [r[1] for r in rows], N)
        if v == "compressed":
            out["tau_L"][b] = tau_fit([r[2] for r in rows],
                                      [r[1] for r in rows], N)
    for b in ("Z", "X"):
        ph = sorted((p for p in rec["phys"] if p["basis"] == b),
                    key=lambda p: p["us"])
        for j, q in enumerate(ph[0]["qubits"]):
            ts, ps = [], []
            for p in ph:
                bit = bits(res, p["k"])[:, j]
                ps.append(float(bit.mean() if b == "X"
                                else 1 - bit.mean()))
                ts.append(p["us"])
            out["phys"][(b, q)] = {"ts": ts, "ps": ps,
                                   "tau": tau_fit(ts, ps, N)}
    V = out["verdict"]
    sep = all(out["eps"][b + "_compressed"][0]
              < out["eps"][b + "_baseline"][0]
              and out["eps"][b + "_compressed"][2]
              < out["eps"][b + "_baseline"][1] for b in ("Z", "X"))
    V["P17"] = bool(sep)
    r18, r19 = True, True
    out["ratio"] = {}
    for b in ("Z", "X"):
        tl = out["tau_L"][b]
        bq = max((v for k, v in out["phys"].items() if k[0] == b),
                 key=lambda v: v["tau"][0])
        tp = bq["tau"]
        ratio = tl[0] / tp[0]
        ratio_hi = tl[2] / max(tp[1], 1e-9)
        out["ratio"][b] = (ratio, ratio_hi)
        r18 &= ratio_hi < 1
        r19 &= RATIO_RANGE[0] <= ratio <= RATIO_RANGE[1]
    V["P18"], V["P19"] = bool(r18), bool(r19)
    V["P20"] = bool(all(v["ps"][0] <= 0.10 for v in out["phys"].values()))
    ok21 = True
    for k, (e, _lo, _hi) in out["eps"].items():
        pl = rec["pred"][k]["live"]
        ok21 &= abs(pl - e) <= MODEL_TOL * e
    V["P21"] = bool(ok21)
    return out


def report(out, rec):
    print("job", rec.get("job_id"), "|", rec["backend"], "|",
          rec["shots"], "shots per circuit | pre-registered",
          rec.get("prereg_utc"))
    print("\nerror per round (plain matching, 2-sigma)   model cal / live")
    for k, (e, lo, hi) in sorted(out["eps"].items()):
        p = rec["pred"][k]
        print("  %-13s %5.2f%% [%5.2f, %5.2f]   %5.2f%% / %5.2f%%"
              % (k, e * 100, lo * 100, hi * 100,
                 p["cal"] * 100, p["live"] * 100))
    for b in ("Z", "X"):
        eb, ec = out["eps"][b + "_baseline"][0], out["eps"][b + "_compressed"][0]
        print("  %s gain x%.2f" % (b, eb / ec))
    print("\nlifetimes (us, 2-sigma)")
    for b in ("Z", "X"):
        t = out["tau_L"][b]
        print("  %s logical (compressed)  %6.0f [%.0f, %.0f]" % (b, *t))
        for (bb, q), v in sorted(out["phys"].items()):
            if bb == b:
                print("  %s physical q%-3d        %6.0f [%.0f, %.0f]"
                      "   err at %.0f us: %.1f%%"
                      % (b, q, *v["tau"], v["ts"][0], v["ps"][0] * 100))
        r, rh = out["ratio"][b]
        print("  %s ratio %.2fx of break-even (upper bound %.2fx)"
              % (b, r, rh))
    print("\nVERDICTS (every pre-registered prediction)")
    for k, ok in out["verdict"].items():
        print("  %s %s  - %s" % (k, "HOLDS" if ok else "FAILS",
                                 rec["predictions"][k]))
