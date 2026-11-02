"""The one-out-of-many proof Pi_oom' (Section 4, Figs. 9 and 8, App. C.1).

Statement:  T in Z_q^{nd x N} (one column per ring member) and A in Z_q^{nd x md}.
Witness:    s in {-1,1}^{md} and an index i with  T e_i = A s  (mod q),
            written with signed selectors v_1 (x) ... (x) v_{k+1} = e_i (tensor.py).

Proof flow (prover and verifier walk through the same transcript):

    commit     t_A = A1 s1 + A2 s2,   s1 = (s, u_1, ..., u_k, v_{k+1})           Fig. 9 top
    level 0    phi_1..phi_ell <- H;  T_1 = fold(T);  a_i = A^T phi_i;
               w_1 = T_1 (v_2 (x) ... (x) v_{k+1});  send t_1 = b_1^T s2 + w_1      Fig. 9
    level j    phi_{j,1..ell} <- H;  T_{j+1} = fold(T_j);  w_{j+1};  send t_{j+1}  Fig. 8
    base case  Pi_k proves R_k in zero knowledge                                 base_case.py

reduce_prove / reduce_verify are the recursion; prove / verify add the base case and the
rejection-sampling restart loop. Every restart throws away ALL randomness (s2, masks, the
bimodal sign b) and starts again from the statement, so an output proof never depends on a
rejected attempt (Fiat-Shamir with aborts).

Bimodal mode (Sec. 3.2 + Lemma C.1): with k even the relation is sign-invariant, so the prover
picks b in {-1,1} per attempt and proves the statement for the witness (b s, b u_1, ..., b v_{k+1}).
All w_j are recomputed from the signed selectors, so they stay consistent automatically.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from . import abdlop, base_case
from .base_case import AttemptStats, BaseCaseProof, BaseStatement, BaseWitness, VerifyResult
from .fiat_shamir import OomTranscript
from .fold import first_round_a
from .oom_params import ProofParams
from .recursion import level_step, unpack_w
from .ring import Ring, matmul_mod, reduce
from .sampling import random_sign, ternary_array
from .tensor import describe, index_to_digits, kron_all, selector_poly, selectors

MAX_ATTEMPTS = 200


@dataclass
class Statement:
    """Public statement of Section 4: T (ring as columns) and A (rot of the key matrix), mod q."""

    T: np.ndarray        # (rows, N)
    A_rot: np.ndarray    # (rows, m*d)

    def validate(self, pp: ProofParams) -> None:
        ps = pp.ps
        rows = ps.n * ps.d
        if self.T.shape != (rows, ps.ring_size):
            raise ValueError(f"T must have shape {(rows, ps.ring_size)}, got {self.T.shape}")
        if self.A_rot.shape != (rows, ps.m * ps.d):
            raise ValueError(f"A must have shape {(rows, ps.m * ps.d)}, got {self.A_rot.shape}")


@dataclass
class Witness:
    s: np.ndarray        # (m, d) entries in {-1, 1}
    index: int           # position in the ring, 0 <= index < N


@dataclass
class OomProof:
    tA: np.ndarray               # (n_com, d)
    t_w: np.ndarray              # (k, d): t_1, ..., t_k
    base: BaseCaseProof


@dataclass
class TraceLevel:
    level: int
    figure: str
    T_shape: tuple[int, int]
    T_next_shape: tuple[int, int]
    w_blocks: np.ndarray | None      # (ell, base) prover only
    selector: str
    phis: np.ndarray | None = None   # (ell, rows) the public challenges of this level


@dataclass
class ProveTrace:
    """What demo.py prints for part (3): the tensor decomposition through the recursion."""

    decomposition: str = ""
    levels: list[TraceLevel] = field(default_factory=list)
    attempts: int = 0
    stats: AttemptStats = field(default_factory=AttemptStats)
    seconds: float = 0.0
    sign: int = 1                    # bimodal sign b of the accepted attempt (SECRET; demo only)


# =====================================================================================
# Recursion (Figs. 9 and 8)
# =====================================================================================


def _base_statement(pp: ProofParams, stmt: Statement, tr: OomTranscript, tA: np.ndarray,
                    commit_level, trace: ProveTrace | None = None, wit_selectors=None):
    """Shared loop for prover and verifier.

    commit_level(level, T_next, w) must return the commitment t_{level+1} (the prover computes it
    from w and s2; the verifier reads it from the proof). Returns the BaseStatement plus the
    prover's packed w's (or None for the verifier).
    """
    ps, q, k = pp.ps, pp.q, pp.k
    tr.absorb_array("tA", tA)
    T = stmt.T
    phis_all, ws, ts = [], [], []
    a = None
    for level in range(k):
        phis = tr.phis(level, T.shape[0])
        if level == 0:
            a = first_round_a(stmt.A_rot, phis, ps.m, ps.d, q)
        rest = None if wit_selectors is None else wit_selectors[level + 1:]
        T_next, w = level_step(T, phis, rest, ps.base, ps.d, q)
        t = commit_level(level, T_next, w)
        tr.absorb_array(f"t_{level + 1}", t)
        if trace is not None:
            still = f"v_{k + 1}" if level + 1 == k else f"v_{level + 2}, ..., v_{k + 1}"
            trace.levels.append(TraceLevel(
                level=level, figure="Fig. 9" if level == 0 else "Fig. 8",
                T_shape=T.shape, T_next_shape=T_next.shape,
                w_blocks=None if w is None else unpack_w(w, ps.ell, ps.base),
                selector=f"v_{level + 1} folded away; still inside w_{level + 1}: {still}",
                phis=phis))
        phis_all.append(phis)
        ws.append(w)
        ts.append(t)
        T = T_next
    st = BaseStatement(T_k=T, a=a, phis=phis_all, tA=tA, t_w=np.stack(ts))
    return st, (None if wit_selectors is None else np.stack(ws))


def reduce_prove(pp: ProofParams, ck: abdlop.CommitmentKey, stmt: Statement, wit: Witness,
                 tr: OomTranscript, sign: int = 1, rng=None, trace: ProveTrace | None = None):
    """Commit and run the k folds as the prover. Returns (tA, BaseStatement, BaseWitness)."""
    ps, d = pp.ps, pp.d
    sel = selectors(wit.index, d, ps.base, pp.k, sign)
    us = [selector_poly(v, _ring(pp)).centered() for v in sel]
    s1 = np.concatenate([sign * wit.s, np.stack(us)]).astype(np.int64)
    s2 = ternary_array((pp.m2, d), rng)
    tA = abdlop.commit_ajtai(ck, s1, s2)

    def commit_level(level, T_next, w):
        return abdlop.commit_row(ck, level, s2, w)

    st, ws = _base_statement(pp, stmt, tr, tA, commit_level, trace, sel)
    return tA, st, BaseWitness(s1=s1, s2=s2, w=ws)


def reduce_verify(pp: ProofParams, stmt: Statement, tr: OomTranscript, proof: OomProof) -> BaseStatement:
    """Recompute every T_j, a_i and phi from public data only (no witness)."""
    if proof.t_w.shape != (pp.k, pp.d):
        raise ValueError(f"expected {pp.k} level commitments")

    def commit_level(level, T_next, w):
        return proof.t_w[level]

    st, _ = _base_statement(pp, stmt, tr, proof.tA, commit_level)
    return st


def _ring(pp: ProofParams) -> Ring:
    return Ring(pp.d, pp.q)


# =====================================================================================
# Full proof with restarts
# =====================================================================================


def prove(pp: ProofParams, ck: abdlop.CommitmentKey, stmt: Statement, wit: Witness,
          tr: OomTranscript, rng=None, trace: ProveTrace | None = None) -> OomProof:
    """Pi_oom': keep trying until every rejection step accepts (expected pp.expected_repetitions).

    `tr` must already contain the statement (fiat_shamir.OomTranscript.absorb_statement).
    """
    stmt.validate(pp)
    _check_witness(pp, wit)
    trace = trace if trace is not None else ProveTrace()
    trace.decomposition = describe(wit.index, pp.d, pp.ps.base, pp.k)
    start = time.perf_counter()
    for attempt in range(1, MAX_ATTEMPTS + 1):
        sign = random_sign(rng) if pp.rejection == "bimodal" else 1
        attempt_tr = tr.fork()
        trace.levels.clear()
        tA, st, bw = reduce_prove(pp, ck, stmt, wit, attempt_tr, sign, rng, trace)
        pf = base_case.prove(pp, ck, attempt_tr, st, bw, rng, trace.stats)
        if pf is not None:
            trace.attempts = attempt
            trace.sign = sign
            trace.seconds = time.perf_counter() - start
            return OomProof(tA=tA, t_w=st.t_w, base=pf)
    raise RuntimeError(f"no accepted attempt after {MAX_ATTEMPTS} tries; parameters are off")


def verify(pp: ProofParams, ck: abdlop.CommitmentKey, stmt: Statement, proof: OomProof,
           tr: OomTranscript) -> VerifyResult:
    """Verifier of Pi_oom'. `tr` must already contain the statement."""
    try:
        stmt.validate(pp)
        if not isinstance(proof, OomProof):
            return VerifyResult.reject("not a proof object")
        if proof.tA.shape != (pp.n_com, pp.d):
            return VerifyResult.reject("malformed proof: t_A has the wrong shape")
        for arr in (proof.tA, proof.t_w):
            if arr.min() < 0 or arr.max() >= pp.q:
                return VerifyResult.reject("malformed proof: commitment not reduced mod q")
        st = reduce_verify(pp, stmt, tr, proof)
    except ValueError as exc:
        return VerifyResult.reject(f"malformed input: {exc}")
    return base_case.verify(pp, ck, tr, st, proof.base)


def _check_witness(pp: ProofParams, wit: Witness) -> None:
    ps = pp.ps
    if wit.s.shape != (ps.m, ps.d) or not np.all(np.abs(wit.s) == 1):
        raise ValueError("secret s must be an (m, d) array with entries in {-1, 1}")
    if not 0 <= wit.index < ps.ring_size:
        raise ValueError(f"index must be in [0, {ps.ring_size})")


def statement_holds(pp: ProofParams, stmt: Statement, wit: Witness) -> bool:
    """T e_i == A s mod q (the relation before any proof). Used by tests and the signer."""
    q = pp.q
    lhs = stmt.T[:, wit.index] % q
    rhs = matmul_mod(stmt.A_rot, reduce(wit.s.reshape(-1), q)[:, None], q)[:, 0]
    return bool(np.array_equal(lhs, rhs))


__all__ = ["Statement", "Witness", "OomProof", "ProveTrace", "prove", "verify",
           "reduce_prove", "reduce_verify", "statement_holds", "index_to_digits", "kron_all"]
