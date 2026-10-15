"""hhmem: reset-free heavy-hex memory experiments on IBM Heron.

Daily use (Colab):
    import hhmem
    hhmem.update_log(service, FOLDER)            # decode finished runs
    hhmem.submit_daily(service, "ibm_kingston", FOLDER)

This is the pipeline used for the 14 Oct 2026 hardware runs. Known
limitations and corrections: see hhmem2.py and the corrected report.
"""
import glob, json, math, os, time
from collections import Counter
import numpy as np
import pymatching
import rustworkx as rx
import stim
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import ALAPScheduleAnalysis, PadDelay
from qiskit.transpiler.preset_passmanagers import \
    generate_preset_pass_manager

COMPRESS = {"Z": dict(merge=True, skip_safe=True, skip_flags=True),
            "X": dict(merge=True, skip_safe=True)}


# ------------------------------------------------------------------
# noise model from calibration (T1, echo T2, gate/readout errors)
# ------------------------------------------------------------------
class ChipNoise:
    def __init__(self, backend):
        self.b, t = backend, backend.target
        self.dt = backend.dt
        self.qp = t.qubit_properties or [None] * t.num_qubits
        self.t_ = t

    def _err(self, name, qargs):
        if name not in self.t_.operation_names:
            return None
        p = self.t_[name].get(qargs)
        return p.error if p is not None else None

    def t1(self, q):
        p = self.qp[q]
        return p.t1 if p is not None and p.t1 else 100e-6

    def t2(self, q):
        p = self.qp[q]
        t2 = p.t2 if p is not None and p.t2 else 100e-6
        return min(t2, 2 * self.t1(q))

    def ro(self, q):
        e = self._err("measure", (q,))
        return min(e, 0.45) if e is not None else 0.03

    def sx(self, q):
        e = self._err("sx", (q,))
        return min(e, 0.1) if e is not None else 3e-4

    def cz(self, a, b):
        e = self._err("cz", (a, b))
        if e is None:
            e = self._err("cz", (b, a))
        return min(e, 0.5) if e is not None else 5e-3

    def idle(self, q, dur):
        a = 1 - math.exp(-dur / self.t1(q))
        b = 1 - math.exp(-dur / self.t2(q))
        return a / 4, a / 4, max(b / 2 - a / 4, 0.0)


RZ = {0: None, 1: "S", 2: "Z", 3: "S_DAG"}


def isa_to_stim(isa, noise, dets, obs, noisy=True, ro_mode="mix"):
    """Compiled circuit -> Stim. Readout error: 'flip' (qubit flips),
    'record' (only the record), 'mix' (half each)."""
    out, pos, nmeas, active = stim.Circuit(), {}, 0, set()
    for inst in isa.data:
        op, name = inst.operation, inst.operation.name
        qs = [isa.find_bit(q).index for q in inst.qubits]
        if name in ("barrier", "id"):
            continue
        if name == "delay":
            if noisy and qs[0] in active:
                val = float(op.params[0])
                u = getattr(op, "unit", "dt")
                dur = val * noise.dt if u == "dt" else val
                if dur > 0:
                    out.append("PAULI_CHANNEL_1", qs,
                               list(noise.idle(qs[0], dur)))
            continue
        if name == "rz":
            ang = float(op.params[0]) / (math.pi / 2)
            k = round(ang)
            if abs(ang - k) > 1e-6:
                raise ValueError("non-Clifford rz")
            if RZ[k % 4]:
                out.append(RZ[k % 4], qs)
            active.update(qs)
        elif name in ("sx", "sxdg", "x"):
            g = {"sx": "SQRT_X", "sxdg": "SQRT_X_DAG", "x": "X"}[name]
            out.append(g, qs)
            if noisy:
                out.append("DEPOLARIZE1", qs, noise.sx(qs[0]))
            active.update(qs)
        elif name == "cz":
            out.append("CZ", qs)
            if noisy:
                out.append("DEPOLARIZE2", qs, noise.cz(*qs))
            active.update(qs)
        elif name == "measure":
            p = noise.ro(qs[0]) if noisy else 0.0
            if ro_mode == "flip":
                if p:
                    out.append("X_ERROR", qs, p)
                out.append("M", qs)
            elif ro_mode == "record":
                out.append("M", qs, p)
            else:
                if p:
                    out.append("X_ERROR", qs, p / 2)
                out.append("M", qs, p / 2)
            pos[isa.find_bit(inst.clbits[0]).index] = nmeas
            nmeas += 1
            active.update(qs)
        elif name == "reset":
            out.append("R", qs)
            if noisy:
                out.append("X_ERROR", qs, noise.ro(qs[0]))
            active.difference_update(qs)
        else:
            raise ValueError("unsupported gate " + name)
    for det in dets:
        out.append("DETECTOR",
                   [stim.target_rec(pos[c] - nmeas) for c in det])
    for k, lst in obs.items():
        out.append("OBSERVABLE_INCLUDE",
                   [stim.target_rec(pos[c] - nmeas) for c in lst], k)
    return out


def dem_of(circ):
    return circ.detector_error_model(
        decompose_errors=True, ignore_decomposition_failures=True,
        approximate_disjoint_errors=True)


def clean_dem(dem):
    """Drop the few undecomposable terms (for correlated matching)."""
    out = stim.DetectorErrorModel()
    for ins in dem.flattened():
        if ins.type == "error":
            comps, cur = [], 0
            for t in ins.targets_copy():
                if t.is_separator():
                    comps.append(cur)
                    cur = 0
                elif t.is_relative_detector_id():
                    cur += 1
            comps.append(cur)
            if max(comps) > 2:
                continue
        out.append(ins)
    out += stim.DetectorErrorModel(
        "detector D%d" % (dem.num_detectors - 1))
    return out


def logical_error(circ, shots, seed=1):
    try:
        m = pymatching.Matching.from_detector_error_model(dem_of(circ))
        det, ob = circ.compile_detector_sampler(seed=seed).sample(
            shots, separate_observables=True)
        return float(np.mean(np.any(m.decode_batch(det) != ob, axis=1)))
    except ValueError:
        return float("nan")


def slope_fit(rs, ps, N=None):
    """F(r) = 1 - 2 pL(r) = A (1 - 2 eps)^r -> (eps, lo, hi)."""
    rs = np.array(rs, float)
    ps = np.array(ps, float)
    F = np.clip(1 - 2 * ps, 1e-3, None)
    y = np.log(F)
    if N:
        sp = np.sqrt(np.clip(ps * (1 - ps), 1e-9, None) / N)
        w = (F / (2 * sp)) ** 2
    else:
        w = np.ones_like(y)
    X = np.vstack([np.ones_like(rs), rs]).T
    cov = np.linalg.inv(X.T @ (w[:, None] * X))
    beta = cov @ X.T @ (w * y)
    se = np.sqrt(cov[1, 1]) if N else 0.0
    f = lambda s: (1 - np.exp(s)) / 2
    return f(beta[1]), f(beta[1] + 2 * se), f(beta[1] - 2 * se)


# ------------------------------------------------------------------
# IBM heavy-hex subsystem code (qiskit-qec construction)
# ------------------------------------------------------------------
def hhc_structure(d):
    idx = lambda r, c: r * d + c
    xg = [[idx(i, j), idx(i + 1, j)]
          for i in range(d - 1) for j in range(d)]
    zg, zcol = [], []
    for i in range(d + 1):
        for j in range(d - 1):
            if i % 2 == j % 2:
                if i == 0:
                    zg.append([idx(0, j), idx(0, j + 1)])
                elif i == d:
                    zg.append([idx(d - 1, j), idx(d - 1, j + 1)])
                else:
                    zg.append([idx(i - 1, j), idx(i - 1, j + 1),
                               idx(i, j), idx(i, j + 1)])
                zcol.append(j)
    xstab = []
    for i in range(d - 1):
        for j in range(d):
            g = i * d + j
            if i % 2 == j % 2:
                xstab.append([g] if j == d - 1 else [g, g + 1])
            elif j == 0:
                xstab.append([g])
    n = d * d
    xanc = [n + k for k in range(len(xg))]
    total, zanc = len(xg), []
    for z in zg:
        if len(z) == 2:
            zanc.append([n + total, n + total + 1, n + total + 2])
            total += 3
        else:
            left = next(n + k for k, x in enumerate(xg)
                        if z[0] in x and z[2] in x)
            right = next(n + k for k, x in enumerate(xg)
                         if z[1] in x and z[3] in x)
            zanc.append([left, n + total, right])
            total += 1
    return {"d": d, "n": n, "xg": xg, "zg": zg, "zcol": zcol,
            "xstab": xstab, "xanc": xanc, "zanc": zanc,
            "nq": n + total}


def mod2(lst):
    return sorted(b for b, k in Counter(lst).items() if k % 2)


def build_hhc(d, rounds, basis, merge=False, skip_safe=False,
              skip_flags=False):
    S = hhc_structure(d)
    n, xg, zg = S["n"], S["xg"], S["zg"]
    sched_ = ("xz" if basis == "Z" else "zx") * rounds
    nx, nz = len(xg), len(zg)
    ncl = sum(nx if r == "x" else 3 * nz for r in sched_) + n
    qc = QuantumCircuit(QuantumRegister(S["nq"], "q"),
                        ClassicalRegister(ncl, "c"))
    cb = iter(range(ncl))
    last, pend, prev = {}, {}, {}

    def RESET(q, skip):
        if skip:
            if q in last:
                pend[q] = last[q]
        else:
            qc.reset(q)
            pend.pop(q, None)

    def MEAS(q):
        b = next(cb)
        qc.measure(q, b)
        if q in pend:
            prev[b] = pend.pop(q)
        last[q] = b
        return b

    for i in range(n):
        qc.reset(i)
    if basis == "X":
        for i in range(n):
            qc.h(i)
    log = []
    for r in sched_:
        if r == "x":
            for a in S["xanc"]:
                RESET(a, skip_safe)
                qc.h(a)
            for step in range(2):
                for k, g in enumerate(xg):
                    qc.cx(S["xanc"][k], g[step])
            bits = []
            for a in S["xanc"]:
                qc.h(a)
                bits.append(MEAS(a))
            log.append(("x", bits))
            continue
        gb, lf, rf = [None] * nz, [None] * nz, [None] * nz
        for step in range(7):
            for j, z in enumerate(zg):
                l, m, rr = S["zanc"][j]
                bnd, top = len(z) == 2, z[0] < d
                if step == 0:
                    RESET(m, skip_safe)
                    RESET(rr, skip_flags)
                    qc.h(rr)
                elif step == 1:
                    qc.cx(rr, m)
                    RESET(l, skip_flags)
                    qc.h(l)
                elif step == 2:
                    qc.cx(l, m)
                    if not (bnd and top):
                        qc.cx(z[1], rr)
                elif step == 3:
                    if bnd and top:
                        qc.cx(z[0], l)
                        qc.cx(z[1], rr)
                    elif not bnd:
                        qc.cx(z[2], l)
                        qc.cx(z[3], rr)
                elif step == 4:
                    qc.cx(rr, m)
                    if not (bnd and top):
                        qc.cx(z[0], l)
                elif step == 5:
                    qc.cx(l, m)
                    qc.h(rr)
                    if not merge:
                        rf[j] = MEAS(rr)
                elif step == 6:
                    qc.h(l)
                    if merge:
                        rf[j] = MEAS(rr)
                    gb[j] = MEAS(m)
                    lf[j] = MEAS(l)
        log.append(("z", gb, lf, rf))
    if basis == "X":
        for i in range(n):
            qc.h(i)
    data_bits = [MEAS(i) for i in range(n)]

    eff = lambda b: [b] + ([prev[b]] if b in prev else [])
    E = lambda bits: [x for b in bits for x in eff(b)]
    dets, flag_dets = [], []
    zr = [e for e in log if e[0] == "z"]
    xr = [e for e in log if e[0] == "x"]
    zs = lambda e, j: E([e[1][g] for g in range(nz)
                         if S["zcol"][g] == j])
    xs = lambda e, s: E([e[1][g] for g in S["xstab"][s]])
    for j in range(d - 1):
        for t, e in enumerate(zr):
            if t == 0:
                if basis == "Z":
                    dets.append(zs(e, j))
            else:
                dets.append(zs(e, j) + zs(zr[t - 1], j))
        if basis == "Z":
            cols = [data_bits[r_ * d + j] for r_ in range(d)]
            cols += [data_bits[r_ * d + j + 1] for r_ in range(d)]
            dets.append(cols + zs(zr[-1], j))
    for s, gs in enumerate(S["xstab"]):
        for t, e in enumerate(xr):
            if t == 0:
                if basis == "X":
                    dets.append(xs(e, s))
            else:
                dets.append(xs(e, s) + xs(xr[t - 1], s))
        if basis == "X":
            sup = sorted({q_ for g in gs for q_ in xg[g]})
            dets.append([data_bits[q_] for q_ in sup] + xs(xr[-1], s))
    for e in zr:
        flag_dets += [E([b]) for b in e[2] + e[3]]
    Q = sorted(set(prev.values()))
    return qc, dets, flag_dets, list(data_bits), S, Q


def clean_samples(qc, shots, seed=0):
    c, order = stim.Circuit(), []
    for inst in qc.data:
        nm = inst.operation.name
        qs = [qc.find_bit(q).index for q in inst.qubits]
        if nm == "h":
            c.append("H", qs)
        elif nm == "cx":
            c.append("CX", qs)
        elif nm == "reset":
            c.append("R", qs)
        elif nm == "measure":
            c.append("M", qs)
            order.append(qc.find_bit(inst.clbits[0]).index)
    raw = c.compile_sampler(seed=seed).sample(shots)
    B = np.zeros((shots, qc.num_clbits), np.uint8)
    B[:, order] = raw
    return B


def gf2_solve(A, y):
    M = np.concatenate([A, y[:, None]], 1).astype(np.uint8) & 1
    nr, nc = A.shape
    piv, r = [], 0
    for c in range(nc):
        rows = np.nonzero(M[r:, c])[0]
        if len(rows) == 0:
            continue
        p = r + rows[0]
        M[[r, p]] = M[[p, r]]
        oth = np.nonzero(M[:, c])[0]
        oth = oth[oth != r]
        M[oth] ^= M[r]
        piv.append(c)
        r += 1
        if r == nr:
            break
    if M[r:, -1].any():
        return None
    x = np.zeros(nc, np.uint8)
    for i, c in enumerate(piv):
        x[c] = M[i, -1]
    return x


def repair(lst, B, Q):
    """Make a detector deterministic after resets were removed."""
    lst = mod2(lst)
    y = B[:, lst].sum(1) % 2
    if y.min() == y.max():
        return lst
    if not Q:
        return None
    A = np.concatenate([B[:, Q], np.ones((len(B), 1), np.uint8)], 1)
    x = gf2_solve(A, y.astype(np.uint8))
    if x is None:
        return None
    return mod2(lst + [Q[i] for i in range(len(Q)) if x[i]])


def make_variant(d, rounds, basis, **kw):
    qc, dets, fl, obs, S, Q = build_hhc(d, rounds, basis, **kw)
    B = clean_samples(qc, 3 * qc.num_clbits + 200)
    dets2 = [repair(x, B, Q) for x in dets]
    bad = sum(x is None for x in dets2)
    fl2 = [x for x in (repair(f, B, Q) for f in fl) if x is not None]
    ob2 = repair(obs, B, Q)
    assert ob2 is not None, "observable could not be repaired"
    return qc, [x for x in dets2 if x is not None], fl2, \
        {0: ob2}, S, bad


# ------------------------------------------------------------------
# placement candidates (zero-SWAP VF2 matches) and scoring
# ------------------------------------------------------------------
def code_graph(qc):
    uses = {}
    for inst in qc.data:
        if inst.operation.name == "cx":
            a, b = sorted(qc.find_bit(q).index for q in inst.qubits)
            uses[(a, b)] = uses.get((a, b), 0) + 1
    g = rx.PyGraph()
    g.add_nodes_from(range(qc.num_qubits))
    g.add_edges_from_no_data(list(uses))
    return g, uses


def role_counts(qc):
    meas, rst = np.zeros(qc.num_qubits), np.zeros(qc.num_qubits)
    for inst in qc.data:
        i = qc.find_bit(inst.qubits[0]).index
        if inst.operation.name == "measure":
            meas[i] += 1
        elif inst.operation.name == "reset":
            rst[i] += 1
    return meas, rst


def candidate_layouts(backend, cg, rng, n_shuffle=30, per=40):
    nq = backend.num_qubits
    edges = {tuple(sorted(e)) for e in backend.coupling_map.get_edges()}
    seen, out = set(), []
    for _ in range(n_shuffle):
        perm = rng.permutation(nq)
        inv = {int(p): i for i, p in enumerate(perm)}
        chipg = rx.PyGraph()
        chipg.add_nodes_from(range(nq))
        chipg.add_edges_from_no_data([(inv[a], inv[b])
                                      for a, b in edges])
        it = rx.vf2_mapping(chipg, cg, subgraph=True, induced=False,
                            id_order=False, call_limit=5_000_000)
        for k, m in enumerate(it):
            if k >= per:
                break
            lay = [0] * cg.num_nodes()
            for cn, vn in m.items():
                lay[vn] = int(perm[cn])
            t = tuple(lay)
            if t not in seen:
                seen.add(t)
                out.append(list(t))
    return out


def model_score(L, uses, meas, rst, nz, n_data, T):
    s = sum(k * nz.cz(L[a], L[b]) for (a, b), k in uses.items())
    for i, p in enumerate(L):
        s += (meas[i] + rst[i]) * nz.ro(p)
        s += sum(nz.idle(p, T)) * (1.0 if i < n_data else 0.5)
        s += nz.sx(p) * (40 if i < n_data else 20)
    return s


# ------------------------------------------------------------------
# one chip: routing, scheduling, dynamical decoupling, building
# ------------------------------------------------------------------
class Chip:
    SPACING = {2: [0.25, 0.5, 0.25]}

    def __init__(self, backend, noise=None):
        self.hw = backend
        self.noise = noise or ChipNoise(backend)
        t = backend.target
        self.align = max(getattr(t, "granularity", 16) or 16,
                         getattr(t, "pulse_alignment", 16) or 16)
        self.xd = {}

    def xdur(self, q):
        if q not in self.xd:
            d_ = self.hw.target["x"][(q,)].duration
            self.xd[q] = int(round(d_ / self.hw.dt))
        return self.xd[q]

    def route(self, qc, lay=None):
        pm = generate_preset_pass_manager(
            optimization_level=1, target=self.hw.target,
            initial_layout=lay, seed_transpiler=1)
        return pm.run(qc)

    def sched(self, routed):
        pm = PassManager([ALAPScheduleAnalysis(target=self.hw.target),
                          PadDelay(target=self.hw.target)])
        return pm.run(routed)

    def add_dd(self, sc):
        """X-X dynamical decoupling in every idle gap of an active qubit."""
        A = self.align
        out = sc.copy_empty_like()
        fresh = set(range(sc.num_qubits))
        npulse = 0
        for inst in sc.data:
            op = inst.operation
            qs = [sc.find_bit(q).index for q in inst.qubits]
            unit = getattr(op, "unit", "dt")
            if (op.name == "delay" and qs[0] not in fresh
                    and unit == "dt"):
                q, T = qs[0], int(op.params[0])
                xd = self.xdur(q)
                F = T - 2 * xd
                if T % A == 0 and xd % A == 0 and F >= 4 * A:
                    p1 = int(F * 0.25) // A * A
                    p2 = int(F * 0.5) // A * A
                    p3 = F - p1 - p2
                    qb = inst.qubits[0]
                    if p1:
                        out.delay(p1, qb, unit="dt")
                    out.x(qb)
                    if p2:
                        out.delay(p2, qb, unit="dt")
                    out.x(qb)
                    if p3:
                        out.delay(p3, qb, unit="dt")
                    npulse += 2
                    continue
            out.append(inst)
            if op.name == "reset":
                fresh.update(qs)
            elif op.name not in ("delay", "barrier", "measure"):
                fresh.difference_update(qs)
        return out, npulse

    def check(self, isa, dets, obs):
        c = isa_to_stim(isa, self.noise, dets, obs, noisy=False)
        det, ob = c.compile_detector_sampler(seed=3).sample(
            400, separate_observables=True)
        return det.any(axis=0), ob.any()

    def duration_us(self, isa):
        try:
            return isa.estimate_duration(self.hw.target, unit="s") * 1e6
        except Exception:
            return float("nan")

    def build(self, basis, R, kw, L):
        qc, dets, fl, obs, S, bad = make_variant(3, R, basis, **kw)
        assert bad == 0, "detectors lost"
        routed = self.route(qc, L)
        ncz = routed.count_ops().get("cz", 0)
        assert ncz == qc.count_ops()["cx"], "SWAPs appeared"
        isa, npul = self.add_dd(self.sched(routed))
        assert npul > 0, "no DD pulses"
        fired, obf = self.check(isa, dets, obs)
        assert not fired.any() and not obf, "noiseless check failed"
        ff, _ = self.check(isa, fl, obs)
        alld = dets + [f for f, b_ in zip(fl, ff) if not b_]
        B = clean_samples(qc, 8)
        circ = isa_to_stim(isa, self.noise, alld, obs)
        arm = {"basis": basis, "rounds": R, "dets": alld,
               "obs": obs[0], "stim": str(circ),
               "ref_d": [int(B[0, x].sum() % 2) for x in alld],
               "ref_o": int(B[0, obs[0]].sum() % 2),
               "layout": [int(x) for x in L],
               "us": self.duration_us(isa)}
        return isa, circ, arm

    def place(self, basis, kw, top=8, seed=7):
        """Calibration tournament: score nominates, simulation picks."""
        qc, dets, fl, obs, S, bad = make_variant(3, 3, basis, **kw)
        cg, uses = code_graph(qc)
        meas, rst = role_counts(qc)
        pool = candidate_layouts(self.hw, cg,
                                 np.random.default_rng(seed))
        sc = lambda L: model_score(L, uses, meas, rst, self.noise,
                                   S["n"], 3 * 5e-6)
        best = None
        for L in sorted(pool, key=sc)[:top]:
            try:
                _, circ, _ = self.build(basis, 3, kw, L)
            except Exception:
                continue
            p = logical_error(circ, 20000, seed=11)
            if best is None or p < best[0]:
                best = (p, L)
        return best[1]


# ------------------------------------------------------------------
# daily replication: baseline vs compressed, same placement
# ------------------------------------------------------------------
def submit_daily(service, backend_name, folder, shots=5000,
                 rounds=(1, 2, 3)):
    from qiskit_ibm_runtime import SamplerV2
    hw = service.backend(backend_name)
    chip = Chip(hw)
    print(hw.name, hw.status().pending_jobs, "jobs in queue")
    rec = {"backend": hw.name, "shots": shots, "arms": [],
           "date": time.strftime("%Y-%m-%d"), "kind": "daily"}
    pubs = []
    day = int(time.strftime("%j"))
    for basis in ("X", "Z") if day % 2 else ("Z", "X"):
        L = chip.place(basis, {})
        for i, R in enumerate(rounds):
            order = ("baseline", "compressed")
            if (i + day) % 2:
                order = order[::-1]
            for v in order:
                kw = COMPRESS[basis] if v == "compressed" else {}
                isa, circ, arm = chip.build(basis, R, kw, L)
                arm["variant"] = v
                arm["pred_pL"] = logical_error(circ, 20000, seed=9)
                rec["arms"].append(arm)
                pubs.append(isa)
    job = SamplerV2(mode=hw).run(pubs, shots=shots)
    rec["job_id"] = job.job_id()
    os.makedirs(folder, exist_ok=True)
    fn = os.path.join(folder, "daily_%s_%s.json"
                      % (rec["date"], rec["job_id"]))
    json.dump(rec, open(fn, "w"))
    print("SUBMITTED", rec["job_id"], "-", len(pubs), "circuits ->", fn)
    return rec["job_id"]


def decode_record(service, rec):
    res = service.job(rec["job_id"]).result()
    G = {}
    for k, arm in enumerate(rec["arms"]):
        bs = res[k].data.c.get_bitstrings()
        raw = np.frombuffer("".join(bs).encode(), np.uint8)
        arr = (raw.reshape(len(bs), -1) - 48)[:, ::-1].astype(bool)
        det = np.stack([np.logical_xor.reduce(arr[:, d], axis=1)
                        for d in arm["dets"]], axis=1)
        det ^= np.array(arm["ref_d"], bool)[None, :]
        ob = np.logical_xor.reduce(arr[:, arm["obs"]], axis=1)
        ob ^= bool(arm["ref_o"])
        m = pymatching.Matching.from_detector_error_model(
            dem_of(stim.Circuit(arm["stim"])))
        pred = m.decode_batch(det)[:, 0].astype(bool)
        G.setdefault((arm["basis"], arm["variant"]), []).append(
            (arm["rounds"], float(np.mean(pred != ob))))
    out = {}
    for (b, v), rows in G.items():
        rows.sort()
        out[b + "_" + v] = slope_fit([r[0] for r in rows],
                                     [r[1] for r in rows],
                                     rec["shots"])
    return out


def update_log(service, folder):
    """Decode every finished run (daily files and the original
    hh_hw11_/hh_hw18_ runs) and print the multi-day table."""
    files, seen = [], set()
    for d_ in (folder, os.getcwd()):
        for pat in ("daily_*.json", "hh_hw11_*.json", "hh_hw18_*.json"):
            for fn in sorted(glob.glob(os.path.join(d_, pat))):
                if os.path.basename(fn) not in seen:
                    seen.add(os.path.basename(fn))
                    files.append(fn)
    rows = []
    for fn in files:
        rec = json.load(open(fn))
        vs = {a.get("variant") for a in rec.get("arms", [])}
        if not {"baseline", "compressed"} <= vs:
            continue
        if "result" not in rec:
            try:
                st = str(service.job(rec["job_id"]).status())
            except Exception as e:
                print("  skip", rec.get("job_id"), e)
                continue
            if "DONE" not in st.upper():
                print("  pending:", rec["job_id"], st)
                continue
            rec["result"] = {k: list(v) for k, v in
                             decode_record(service, rec).items()}
            json.dump(rec, open(fn, "w"))
        r = rec["result"]
        date = rec.get("date") or time.strftime(
            "%Y-%m-%d", time.localtime(os.path.getmtime(fn)))
        for b in ("Z", "X"):
            eb, lb, hb = r[b + "_baseline"]
            ec, lc, hc = r[b + "_compressed"]
            rows.append((date, rec["backend"], b, eb, ec, eb / ec,
                         hc < lb, rec["job_id"]))
    rows.sort()
    print()
    print("date        chip            basis  baseline  compressed"
          "  gain   2-sigma")
    for d_, bk, b, eb, ec, g, sep, j in rows:
        print("%s  %-14s  %s     %5.2f%%    %5.2f%%     x%.2f  %s"
              % (d_, bk, b, eb * 100, ec * 100, g,
                 "yes" if sep else "no"))
    for b in ("Z", "X"):
        gs = [r[5] for r in rows if r[2] == b]
        if gs:
            print("%s: %d runs, compressed wins %d, median gain x%.2f,"
                  " separated %d" % (b, len(gs),
                  sum(g > 1 for g in gs), float(np.median(gs)),
                  sum(r[6] for r in rows if r[2] == b)))
    return rows