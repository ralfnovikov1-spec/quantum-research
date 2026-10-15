"""Power test for noisespec: simulated qubits with KNOWN noise.

Dephasing noise is a frequency fluctuation delta(t) (rad/us). A sequence
with pi-pulses has a switching function s(t) = +-1; the accumulated phase
is phi = int s(t) delta(t) dt and the coherence is <cos phi> over noise
realisations, times the T1 factor exp(-w / 2T1).

  white : Markovian dephasing, coherence exp(-w / T_phi) for every sequence
          (pulses cannot refocus it)
  1/f   : sum of random-telegraph fluctuators with log-spaced switching
          rates (standard model of 1/f flux / two-level-defect noise)

Shots are drawn binomially with readout error, then the SAME fitting and
verdict code as the hardware analysis (noisespec) is applied.
"""
import numpy as np

import noisespec as N

DT = 0.5                                   # us


def _switch(seq, w, t):
    """+-1 switching function on grid t for a sequence of length w."""
    n = N.NPULSE.get(seq, 0)
    if n == 0:
        return np.ones_like(t)
    pulses = w * (np.arange(n) + 0.5) / n     # CPMG / echo pulse times
    return (-1.0) ** (t[:, None] > pulses[None, :]).sum(1)


def _telegraph(rng, M, steps, rates, amp):
    """M realisations x steps of sum of telegraph processes (rad/us)."""
    out = np.zeros((M, steps))
    for g, a in zip(rates, amp):
        p = 1 - np.exp(-g * DT)
        flips = rng.random((M, steps)) < p
        s0 = rng.choice([-1.0, 1.0], size=(M, 1))
        out += a * s0 * (-1.0) ** np.cumsum(flips, axis=1)
    return out


def qubit_curves(rng, T1, Tphi_white, f_amp, M=300, ro=0.02, shots=N.SHOTS):
    """Measured curves {seq: [(w, value)]} for one simulated qubit."""
    wmax = max(N.WAITS_US)
    t = np.arange(0, wmax + DT, DT)
    rates = np.logspace(-3, 0, 10)        # 1/ms .. 1/us switching rates
    amp = f_amp * np.ones_like(rates)
    delta = _telegraph(rng, M, len(t), rates, amp) if f_amp > 0 else None
    curves = {}
    for seq in N.SEQS:
        pts = []
        for w in N.WAITS_US:
            if seq == "T1":
                p1 = np.exp(-w / T1)
            else:
                k = int(round(w / DT))
                c = np.exp(-w / (2 * T1)) * np.exp(-w / Tphi_white)
                if delta is not None:
                    s = _switch(seq, w, t[:k])
                    phi = (delta[:, :k] * s).sum(1) * DT
                    c *= np.mean(np.cos(phi))
                p1 = (1 - c) / 2
            p1 = p1 * (1 - ro) + (1 - p1) * ro          # readout error
            m = rng.binomial(shots, p1) / shots
            pts.append((w, m if seq == "T1" else 1 - 2 * m))
        curves[seq] = pts
    return curves


def fit_curves(curves):
    out = {}
    for seq, pts in curves.items():
        ws = [p[0] for p in pts]
        v = np.array([p[1] for p in pts])
        out[seq] = ((float("nan"), float("nan")) if v[0] <= 0.2
                    else N.fit_decay(ws, v / v[0]))
    return out


def scenario(name, n_qubits=120, seed=0):
    """Returns {q: fits} for a population of simulated qubits."""
    rng = np.random.default_rng(seed)
    fits = {}
    for q in range(n_qubits):
        T1 = rng.uniform(150, 400)
        if name == "white":
            Tw, fa = rng.uniform(150, 600), 0.0
        elif name == "1/f":
            Tw, fa = 1e9, rng.lognormal(np.log(0.006), 0.4)
        elif name == "mixed":
            Tw, fa = rng.uniform(400, 1500), rng.lognormal(np.log(0.004), 0.5)
        else:
            raise ValueError(name)
        fits[q] = fit_curves(qubit_curves(rng, T1, Tw, fa))
    return fits
