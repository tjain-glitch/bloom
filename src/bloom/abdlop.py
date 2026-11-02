"""ABDLOP commitment (paper Section 2.6, from LNP22).

    t_A = A1 s1 + A2 s2          Ajtai part: commits to the SHORT vector s1 (keys, selectors)
    t_B = B  s2 + m              BDLOP part: commits to FULL-SIZE polynomials m (one row each)

s2 is short random ternary noise. Under Module-LWE, (A2; B) s2 looks uniform, so t_A and t_B
hide s1 and m (Lemma 2.11). Under Module-SIS you cannot open t_A to two different short s1
(Lemma 2.10). The size of t_A does not depend on len(s1), which is why the long secret key
goes in the Ajtai part and the few full-size polynomials (w_j, masks, garbage) go in the
BDLOP part, one row of B each.

Layout of the BDLOP rows used by our one-out-of-many proof (see oom_params.ProofParams):

    rows 0 .. k-1                 b_1..b_k   for w_1..w_k            (Figs. 8 and 9)
    next JL_ROWS/d rows           B_y        for the JL masks y3     (base case)
    next lam rows                 B_g        for the garbage g_i     (base case, Z_q lift)
    last row                      b          for the quadratic garbage t = b^T s2 + g1 (Fig. 7)

All public matrices are expanded from a public seed with SHAKE-256 ("nothing up my sleeve"),
so anyone can rebuild them and nobody knows a trapdoor for them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .oom_params import ProofParams
from .ring import matvec, polyvec_dot
from .sampling import expand_array


@dataclass(frozen=True)
class CommitmentKey:
    """Public ABDLOP matrices, as int64 arrays of shape (rows, cols, d)."""

    seed: bytes
    A1: np.ndarray   # (n_com, m1, d)
    A2: np.ndarray   # (n_com, m2, d)
    B: np.ndarray    # (n_bdlop_rows, m2, d)
    q: int

    # Named row ranges inside B.
    def rows_w(self, pp: ProofParams) -> slice:
        return slice(0, pp.k)

    def rows_y3(self, pp: ProofParams) -> slice:
        return slice(pp.k, pp.k + pp.n_y3)

    def rows_g(self, pp: ProofParams) -> slice:
        start = pp.k + pp.n_y3
        return slice(start, start + pp.lam)

    def row_quad(self, pp: ProofParams) -> int:
        return pp.n_bdlop_rows - 1


def keygen(pp: ProofParams, seed: bytes) -> CommitmentKey:
    """Expand the commitment key from a public seed."""
    if len(seed) < 16:
        raise ValueError("commitment seed must be at least 16 bytes")
    d, q = pp.d, pp.q
    return CommitmentKey(
        seed=seed,
        A1=expand_array(seed, "abdlop-A1", (pp.n_com, pp.m1, d), q),
        A2=expand_array(seed, "abdlop-A2", (pp.n_com, pp.m2, d), q),
        B=expand_array(seed, "abdlop-B", (pp.n_bdlop_rows, pp.m2, d), q),
        q=q,
    )


def commit_ajtai(ck: CommitmentKey, s1: np.ndarray, s2: np.ndarray) -> np.ndarray:
    """t_A = A1 s1 + A2 s2 (inputs reduced mod q, shape (len, d))."""
    q = ck.q
    return (matvec(ck.A1, s1 % q, q) + matvec(ck.A2, s2 % q, q)) % q


def commit_rows(ck: CommitmentKey, rows: slice, s2: np.ndarray, messages: np.ndarray) -> np.ndarray:
    """t_i = b_i^T s2 + m_i for the BDLOP rows in `rows`; messages has shape (count, d)."""
    q = ck.q
    B = ck.B[rows]
    if B.shape[0] != messages.shape[0]:
        raise ValueError(f"{B.shape[0]} rows but {messages.shape[0]} messages")
    return (matvec(B, s2 % q, q) + messages) % q


def commit_row(ck: CommitmentKey, row: int, s2: np.ndarray, message: np.ndarray) -> np.ndarray:
    """Single BDLOP row: t = b_row^T s2 + message."""
    return (polyvec_dot(ck.B[row], s2 % ck.q, ck.q) + message) % ck.q


def opens(ck: CommitmentKey, tA: np.ndarray, tB: np.ndarray, rows: slice,
          s1: np.ndarray, s2: np.ndarray, messages: np.ndarray) -> bool:
    """Exact opening check (debug tool only: it needs the secret s1, s2, m)."""
    return bool(np.array_equal(commit_ajtai(ck, s1, s2), tA)
                and np.array_equal(commit_rows(ck, rows, s2, messages), tB))
