"""Leakage-aware decoding for reset-free heavy-hex memory: SIMULATION.

Stim cannot model leakage, so it is injected after sampling:
  * each ancilla measurement can start a leak with probability p_leak;
    once leaked, the ancilla stays leaked for the rest of the shot (on IBM
    Heron a 'reset' is measure + conditional X, which does not clear |2>);
  * a leaked measurement's analog (IQ) point is drawn from a |2> blob, and
    its hard bit is whatever the |0>/|1> discriminator says (mostly '1');
  * every non-leaked measurement gets an IQ point from its true |0>/|1>
    blob, so readout errors come from Gaussian overlap.
Decoders compared on the SAME shots:
  hard    : standard PyMatching on the circuit's detector error model;
  erasure : shots with IQ-flagged measurements are decoded with those
            records' flip probability set to 0.5 (no information).
"""
import numpy as np
import pymatching
import stim


def readout_records(circ):
    """record index -> qubit, in measurement order."""
    q = []
    for ins in circ.flattened():
        if ins.name == "M":
            q += [t.value for t in ins.targets_copy()]
    return np.array(q)


def ancilla_records(circ):
    """Indices of mid-circuit (ancilla) measurement records: qubits that
    are measured more than once; data qubits are read only at the end."""
    q = readout_records(circ)
    counts = {x: int((q == x).sum()) for x in set(q.tolist())}
    return np.array([i for i, x in enumerate(q) if counts[x] > 1])


class IQModel:
    """Isotropic Gaussian blobs in the IQ plane (arbitrary units)."""

    def __init__(self, sep=4.6, leak_offset=(1.7, 1.2), leak_visible=True):
        self.mu = {0: np.array([0.0, 0.0]), 1: np.array([sep, 0.0])}
        self.mu[2] = (np.array([sep * leak_offset[0], sep * leak_offset[1]])
                      if leak_visible else self.mu[1].copy())

    def sample(self, rng, states):
        pts = np.stack([self.mu[s] for s in states.ravel()]).reshape(
            states.shape + (2,))
        return pts + rng.normal(size=pts.shape)

    def hard_bit(self, iq):
        d0 = ((iq - self.mu[0]) ** 2).sum(-1)
        d1 = ((iq - self.mu[1]) ** 2).sum(-1)
        return d1 < d0

    def leak_flag(self, iq, n_sigma=4.0):
        """Flag points that are > n_sigma from BOTH calibrated blobs.
        Uses only |0>/|1> calibration - the |2> blob position is unknown,
        as on IBM Heron, where |2> cannot be prepared on purpose."""
        d0 = np.sqrt(((iq - self.mu[0]) ** 2).sum(-1))
        d1 = np.sqrt(((iq - self.mu[1]) ** 2).sum(-1))
        return (d0 > n_sigma) & (d1 > n_sigma)


def simulate(circ, shots, p_leak, iq, rng):
    """Returns det, obs, flags (shots x n_records) for one arm."""
    # readout errors come only from the IQ model, so sample with record flips off
    clean = set_m_flip(circ, 0.0)
    rec = clean.compile_sampler(seed=int(rng.integers(1 << 30))).sample(shots)
    rec = rec.astype(bool)
    anc = ancilla_records(circ)
    qrec = readout_records(circ)
    state = rec.astype(int)                          # true |0>/|1> per record
    if p_leak > 0 and len(anc):
        for qa in np.unique(qrec[anc]):
            idx = [i for i in anc if qrec[i] == qa]   # this ancilla's records
            start = rng.random((shots, len(idx))) < p_leak
            leaked = np.cumsum(start, axis=1) > 0     # stays leaked
            for j, r in enumerate(idx):
                state[leaked[:, j], r] = 2
    pts = iq.sample(rng, state)
    hard = iq.hard_bit(pts)
    flags = iq.leak_flag(pts)
    flags[:, [i for i in range(rec.shape[1]) if i not in set(anc.tolist())]] = False
    m2d = circ.compile_m2d_converter()
    det, obs = m2d.convert(measurements=hard, separate_observables=True)
    return det, obs[:, 0], flags


def _dem(circ):
    return circ.detector_error_model(decompose_errors=True,
                                     ignore_decomposition_failures=True,
                                     approximate_disjoint_errors=True)


def set_m_flip(circ, p):
    """Copy of circ with every measurement's record-flip probability = p."""
    out = stim.Circuit()
    for ins in circ.flattened():
        if ins.name == "M":
            for t in ins.targets_copy():
                out.append("M", [t.value], p)
        else:
            out.append(ins)
    return out


def erase_records(circ, records):
    """Copy of circ where the listed measurement records carry no
    information (record-flip probability 0.5)."""
    rs, out, k = set(records), stim.Circuit(), 0
    for ins in circ.flattened():
        if ins.name == "M":
            for t in ins.targets_copy():
                out.append("M", [t.value], 0.5 if k in rs else
                           (ins.gate_args_copy() or [0.0])[0])
                k += 1
        else:
            out.append(ins)
    return out


def decode_both(circ, det, obs, flags):
    """Logical-failure arrays for the hard and erasure decoders."""
    m = pymatching.Matching.from_detector_error_model(_dem(circ))
    pred = m.decode_batch(det)[:, 0].astype(bool)
    hard_wrong = pred != obs
    er_wrong = hard_wrong.copy()
    patterns = {}
    for s in np.nonzero(flags.any(1))[0]:
        patterns.setdefault(tuple(np.nonzero(flags[s])[0]), []).append(s)
    for pat, shots in patterns.items():
        mm = pymatching.Matching.from_detector_error_model(
            _dem(erase_records(circ, pat)))
        p2 = mm.decode_batch(det[shots])[:, 0].astype(bool)
        er_wrong[shots] = p2 != obs[shots]
    return hard_wrong, er_wrong, len(patterns)


def mcnemar(hard_wrong, er_wrong):
    from math import comb
    b = int((hard_wrong & ~er_wrong).sum())     # erasure fixes
    c = int((~hard_wrong & er_wrong).sum())     # erasure breaks
    n = b + c
    p = sum(comb(n, i) for i in range(b, n + 1)) / 2 ** n if n else 1.0
    return b, c, p
