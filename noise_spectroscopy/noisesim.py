"""Power test for noisespec: a simulated chip with KNOWN noise, run through
the same analyse() code as the hardware data (raw bitstrings in, fits out).

Per qubit:
  * T1 amplitude damping, white (Markovian) dephasing with time T_phi;
  * optional 1/f dephasing: a sum of random-telegraph fluctuators with
    log-spaced switching rates (standard model of 1/f noise);
  * optional CONFOUNDS that are not noise but could fake a signal:
      - static frequency offset d0 (imperfect drive frequency),
      - static ZZ coupling to 2 neighbours (not refocused by simultaneous
        X pulses: envelope cos(zeta t / 2)^2 on every sequence),
      - asymmetric readout (P(1|0) != P(0|1)).
A pi-pulse sequence has switching function s(t) = +-1; the accumulated
phase is phi = int s(t) (delta(t) + d0) dt, and <X> + i<Y> = env <e^{i phi}>.
"""
import numpy as np

import noisespec as N

DT = 0.5                                   # us


def _switch(seq, w, t):
    n = N.NPULSE.get(seq, 0)
    if n == 0:
        return np.ones_like(t)
    pulses = w * (np.arange(n) + 0.5) / n
    return (-1.0) ** (t[:, None] > pulses[None, :]).sum(1)


def _telegraph(rng, M, steps, rates, amp):
    out = np.zeros((M, steps))
    for g, a in zip(rates, amp, strict=True):
        flips = rng.random((M, steps)) < 1 - np.exp(-g * DT)
        s0 = rng.choice([-1.0, 1.0], size=(M, 1))
        out += a * s0 * (-1.0) ** np.cumsum(flips, axis=1)
    return out


def qubit_p1(rng, T1, Tphi, f_amp, d0=0.0, zeta=0.0, ro0=0.02, ro1=0.02,
             M=300):
    """True P(measure 1) for every (seq, wait) in noisespec's index."""
    t = np.arange(0, max(N.WAITS_US) + DT, DT)
    delta = (_telegraph(rng, M, len(t), np.logspace(-3, 0, 10),
                        f_amp * np.ones(10)) if f_amp > 0 else None)
    p = {("RO0", 0): 0.0, ("RO1", 0): 1.0}
    for s in N.SEQS:
        for w in N.WAITS_US:
            if s == "T1":
                p[(s, w)] = np.exp(-w / T1)
                continue
            k = int(round(w / DT))
            sw = _switch(s, w, t[:k])
            env = (np.exp(-w / (2 * T1)) * np.exp(-w / Tphi)
                   * np.cos(zeta * w / 2) ** 2)
            phi0 = d0 * sw.sum() * DT
            if delta is not None:
                z = np.mean(np.exp(1j * ((delta[:, :k] * sw).sum(1) * DT
                                         + phi0)))
            else:
                z = np.exp(1j * phi0)
            c = env * (z.imag if s == "RAMSEY_Y" else z.real)
            p[(s, w)] = (1 - c) / 2
    return {key: v * (1 - ro1) + (1 - v) * ro0 for key, v in p.items()}


class _Reg:
    def __init__(self, s):
        self._s = s

    def get_bitstrings(self):
        return self._s


class _Pub:
    def __init__(self, s):
        self.data = type("D", (), {"c": _Reg(s)})()


def scenario(name, n_qubits=120, seed=0, shots=N.SHOTS):
    """Simulated chip -> {q: fits} via noisespec.analyse (same code path)."""
    rng = np.random.default_rng(seed)
    base, conf = name.split("+") if "+" in name else (name, "")
    probs = []
    for _ in range(n_qubits):
        T1 = rng.uniform(150, 400)
        if base == "white":
            Tphi, fa = rng.uniform(150, 600), 0.0
        elif base == "1/f":
            Tphi, fa = 1e9, rng.lognormal(np.log(0.006), 0.4)
        elif base == "mixed":
            Tphi, fa = rng.uniform(400, 1500), rng.lognormal(np.log(0.004), 0.5)
        else:
            raise ValueError(name)
        kw = {}
        if conf == "confounds":
            kw = dict(d0=rng.uniform(-1, 1) * 2 * np.pi * 5e-3,   # +-5 kHz
                      zeta=rng.uniform(0, 1) * 2 * np.pi * 1e-3,  # <=1 kHz
                      ro0=rng.uniform(0.005, 0.02),
                      ro1=rng.uniform(0.02, 0.06))
        probs.append(qubit_p1(rng, T1, Tphi, fa, **kw))
    _, index = N.WAITS_US, [("RO0", 0), ("RO1", 0)] + [
        (s, w) for s in N.SEQS for w in N.WAITS_US]
    res = []
    for key in index:
        pr = np.array([pq[key] for pq in probs])
        bits = rng.random((shots, n_qubits)) < pr[None, :]
        res.append(_Pub(["".join("1" if b else "0" for b in row[::-1])
                         for row in bits]))
    return N.analyse(res, index, list(range(n_qubits)))
