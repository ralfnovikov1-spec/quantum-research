"""Hardware pipeline for leakage-aware decoding with kerneled (IQ) readout.

1. record_clbits(isa): Stim record order -> classical bit, so IQ data
   (shots x clbits from SamplerV2 meas_type="kerneled") can be aligned with
   the decoder's measurement records.
2. Blob calibration from prepare-|0> / prepare-|1> shots per qubit.
3. Leak flag: Mahalanobis distance (pooled covariance) from the SEGMENT
   between the |0> and |1> centres. Points smeared along that segment by T1
   decay during readout are never flagged; a third (leaked) cluster off the
   segment is. The threshold k is the smallest value whose false-flag rate
   on the calibration shots is <= FP_TARGET (fixed before the pilot).
4. Persistence: a leaked ancilla stays leaked (IBM resets do not clear |2>),
   so P(flag at its next measurement | flag now) must far exceed the base
   flag rate. Ratio < PERSIST_MIN => flags are noise, not leakage.
"""
import numpy as np

FP_TARGET = 1e-3          # per-measurement false-flag rate on calibration
PERSIST_MIN = 5.0         # minimum persistence ratio to call flags leakage


def record_clbits(isa):
    """Classical bit written by each measurement, in Stim record order
    (isa_to_stim appends one record per measure instruction, in order)."""
    out = []
    for inst in isa.data:
        if inst.operation.name == "measure":
            out.append(isa.find_bit(inst.clbits[0]).index)
    return np.array(out)


def iq2(z):
    """complex IQ -> (..., 2) real array"""
    z = np.asarray(z)
    return np.stack([z.real, z.imag], axis=-1)


class Blobs:
    """|0>/|1> centres and pooled covariance for ONE qubit."""

    def __init__(self, iq0, iq1):
        a, b = iq2(iq0), iq2(iq1)
        self.m0, self.m1 = a.mean(0), b.mean(0)
        # robust pooled covariance: core of each blob (drop decayed tails)
        ca = a[np.linalg.norm(a - self.m0, axis=1)
               < np.percentile(np.linalg.norm(a - self.m0, axis=1), 90)]
        cb = b[np.linalg.norm(b - self.m1, axis=1)
               < np.percentile(np.linalg.norm(b - self.m1, axis=1), 90)]
        self.cov = 0.5 * (np.cov(ca.T) + np.cov(cb.T))
        self.icov = np.linalg.inv(self.cov)

    def _maha(self, d):
        return np.sqrt(np.einsum("...i,ij,...j->...", d, self.icov, d))

    def hard_bit(self, iq):
        p = iq2(iq)
        return self._maha(p - self.m1) < self._maha(p - self.m0)

    def seg_dist(self, iq):
        """Mahalanobis distance to the |0>-|1> segment."""
        p = iq2(iq)
        u = self.m1 - self.m0
        t = np.clip(((p - self.m0) @ self.icov @ u) / (u @ self.icov @ u),
                    0, 1)
        proj = self.m0 + t[..., None] * u
        return self._maha(p - proj)


def choose_k(blobs, iq0, iq1, fp=FP_TARGET):
    d = np.concatenate([blobs.seg_dist(iq0), blobs.seg_dist(iq1)])
    return float(max(np.quantile(d, 1 - fp), 3.0))


def flags_for_arm(iq_arm, rec_clbits, rec_qubits, blobs, kq, ancilla_recs):
    """shots x records boolean leak flags (ancilla records only)."""
    n_shots, n_rec = iq_arm.shape[0], len(rec_clbits)
    fl = np.zeros((n_shots, n_rec), bool)
    for r in ancilla_recs:
        q = int(rec_qubits[r])
        z = iq_arm[:, rec_clbits[r]]
        fl[:, r] = blobs[q].seg_dist(z) > kq[q]
    return fl


def hard_bits_for_arm(iq_arm, rec_clbits, rec_qubits, blobs):
    return np.stack([blobs[int(rec_qubits[r])].hard_bit(iq_arm[:, c])
                     for r, c in enumerate(rec_clbits)], axis=1)


def persistence(flags, rec_qubits, ancilla_recs):
    """(P(flag next | flag now), base flag rate, ratio) over all ancillas."""
    by_q = {}
    for r in ancilla_recs:
        by_q.setdefault(int(rec_qubits[r]), []).append(r)
    hits = cond = 0
    for recs in by_q.values():
        recs.sort()
        for a, b in zip(recs[:-1], recs[1:], strict=True):
            cond += int(flags[:, a].sum())
            hits += int((flags[:, a] & flags[:, b]).sum())
    base = float(flags[:, list(ancilla_recs)].mean()) if len(ancilla_recs) else 0.0
    p_next = hits / cond if cond else float("nan")
    ratio = p_next / base if base > 0 and cond else float("nan")
    # one-sided binomial: are repeat flags more common than chance (base)?
    from scipy.stats import binom
    pval = float(binom.sf(hits - 1, cond, base)) if cond and base > 0 else 1.0
    return p_next, base, ratio, hits, cond, pval


def leakage_detected(pers, ratio_min=PERSIST_MIN, alpha=1e-3, min_hits=20):
    """Leakage is called only if repeat flags are far above chance:
    ratio >= ratio_min AND binomial p < alpha AND >= min_hits repeats."""
    _, _, ratio, hits, _, pval = pers
    return bool(ratio >= ratio_min and pval < alpha and hits >= min_hits)
