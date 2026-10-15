"""Leakage with neighbour damage, and a leakage-aware decoder (SIMULATION).

Physics model (per shot, gate by gate, using stim.FlipSimulator):
  * measurement-induced leakage: an ancilla leaks with probability p_leak at
    each of its mid-circuit measurements and then STAYS leaked (IBM Heron
    resets are measure + conditional X and do not clear |2>);
  * a leaked qubit in a CZ applies a random phase to its partner: Z on the
    partner with probability 1/2 (standard leakage model for CZ gates);
  * a leaked measurement returns an IQ point from a |2> blob; others return
    IQ points from their true |0>/|1> blob (readout errors = blob overlap).
Decoders on the SAME shots:
  hard      : standard detector error model;
  erasure   : IQ-flagged records carry no information (flip prob 0.5);
  erasure+N : additionally, from the first flagged measurement of an
              ancilla on, every CZ partner of that ancilla gets Z_ERROR(1/2)
              in the decoder's model (the decoder expects the damage).
"""
import numpy as np
import pymatching
import stim

from leaksim import IQModel, ancilla_records, mcnemar, readout_records  # noqa: F401


def _ops(circ):
    return list(circ.flattened())


def simulate(circ, shots, p_leak, iq, rng, return_states=False):
    """det, obs, flags for one arm, with leakage propagating through CZs.
    return_states=True returns the true per-record states (0/1/2) instead."""
    ops = _ops(circ)
    nq = circ.num_qubits
    anc_q = set(readout_records(circ)[ancilla_records(circ)].tolist())
    sim = stim.FlipSimulator(batch_size=shots, num_qubits=nq,
                             seed=int(rng.integers(1 << 30)))
    leaked = np.zeros((nq, shots), bool)
    state_leak = []                            # per record: leaked mask
    for ins in ops:
        name = ins.name
        if name in ("DETECTOR", "OBSERVABLE_INCLUDE"):
            continue
        if name == "M":
            for t in ins.targets_copy():
                q = t.value
                if q in anc_q and p_leak > 0:
                    leaked[q] |= rng.random(shots) < p_leak
                state_leak.append(leaked[q].copy())
                sim.do(stim.CircuitInstruction("M", [q], [0.0]))
            continue
        sim.do(ins)
        if name == "CZ" and leaked.any():
            tg = [t.value for t in ins.targets_copy()]
            for a, b in zip(tg[0::2], tg[1::2], strict=True):
                for src, dst in ((a, b), (b, a)):
                    m = leaked[src] & (rng.random(shots) < 0.5)
                    if m.any():
                        mask = np.zeros((nq, shots), bool)
                        mask[dst] = m
                        sim.broadcast_pauli_errors(pauli="Z", mask=mask)
    ref = np.array(circ.reference_sample(), bool)          # noiseless record
    vals = (sim.get_measurement_flips() ^ ref[:, None]).T  # shots x records
    lk = np.stack(state_leak, axis=1)
    states = np.where(lk, 2, vals.astype(int))
    if return_states:
        return states
    pts = iq.sample(rng, states)
    hard = iq.hard_bit(pts)
    flags = iq.leak_flag(pts)
    anc = set(ancilla_records(circ).tolist())
    flags[:, [i for i in range(flags.shape[1]) if i not in anc]] = False
    det, obs = circ.compile_m2d_converter().convert(
        measurements=hard, separate_observables=True)
    return det, obs[:, 0], flags


def _dem(c):
    return c.detector_error_model(decompose_errors=True,
                                  ignore_decomposition_failures=True,
                                  approximate_disjoint_errors=True)


def aware_circuit(circ, flagged_records, neighbours=True):
    """Decoder model for one flag pattern: flagged records -> M(0.5); with
    neighbours=True also Z_ERROR(0.5) on CZ partners of each flagged ancilla
    after its first flagged measurement."""
    flagged = set(flagged_records)
    qrec = readout_records(circ)
    first = {}                                 # qubit -> first flagged record
    for r in sorted(flagged):
        first.setdefault(int(qrec[r]), r)
    out, k, active = stim.Circuit(), 0, set()
    for ins in _ops(circ):
        if ins.name == "M":
            for t in ins.targets_copy():
                p = 0.5 if k in flagged else (ins.gate_args_copy() or [0.0])[0]
                out.append("M", [t.value], p)
                if first.get(t.value) == k:
                    active.add(t.value)
                k += 1
            continue
        out.append(ins)
        if neighbours and ins.name == "CZ" and active:
            tg = [t.value for t in ins.targets_copy()]
            for a, b in zip(tg[0::2], tg[1::2], strict=True):
                if a in active:
                    out.append("Z_ERROR", [b], 0.5)
                if b in active:
                    out.append("Z_ERROR", [a], 0.5)
    return out


def decode_three(circ, det, obs, flags):
    m = pymatching.Matching.from_detector_error_model(_dem(circ))
    hard = m.decode_batch(det)[:, 0].astype(bool) != obs
    er, ern = hard.copy(), hard.copy()
    pats = {}
    for s in np.nonzero(flags.any(1))[0]:
        pats.setdefault(tuple(np.nonzero(flags[s])[0]), []).append(s)
    for pat, sh in pats.items():
        for out, nb in ((er, False), (ern, True)):
            mm = pymatching.Matching.from_detector_error_model(
                _dem(aware_circuit(circ, pat, neighbours=nb)))
            out[sh] = mm.decode_batch(det[sh])[:, 0].astype(bool) != obs[sh]
    return hard, er, ern, len(pats)
