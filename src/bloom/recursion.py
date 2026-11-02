"""Recursion levels of Figs. 9 and 8: packing w_j and the Lemma 4.1 linking checks.

Each level j commits to one polynomial w_j whose coefficient vector is

    w_j = w_{j,0} || w_{j,1} || ... || w_{j,ell-1},     w_{j,i} in Z_q^base,

so ell * base must be at most d (we use ell = 8, base = 8, d = 64; see params.py for why the
paper's ell = 16 cannot be packed this way).

Lemma 4.1: for a selector polynomial u with coefficient vector v || 0^{d-base},

    constant coefficient of  X^{i*base} * u * sigma(w)   =   < v, w_i >

because multiplying u by X^{i*base} slides v under the i-th block of w (no wrap-around
since (i+1)*base <= d), and Lemma 2.2 turns the constant coefficient of sigma(a) b into <a, b>.
The base case proves the two linking relations below in zero knowledge; this module computes
them in the clear, which is what the OpeningChecker (debug tool) and the tests use.

    level 0 (Fig. 9):   const( X^{i base} u_1 sigma(w_1) - sigma(a_i)^T s )          = 0
    level j (Fig. 8):   const( X^{i base} u_{j+1} sigma(w_{j+1}) - sigma(phi_{j,i}) w_j ) = 0
"""

from __future__ import annotations

import numpy as np

from .fold import apply_selector, first_round_a, fold
from .ring import conj_array, polymul, polyvec_dot, reduce
from .tensor import kron_all


def pack_w(w: np.ndarray, d: int, q: int) -> np.ndarray:
    """w in Z_q^{ell*base} -> coefficient array of one polynomial (zero padded to d)."""
    if w.shape[0] > d:
        raise ValueError(f"cannot pack {w.shape[0]} values into a degree-{d} polynomial")
    out = np.zeros(d, dtype=np.int64)
    out[: w.shape[0]] = reduce(w, q)
    return out


def unpack_w(poly: np.ndarray, ell: int, base: int) -> np.ndarray:
    """Inverse of pack_w, returned as shape (ell, base): row i is w_{j,i}."""
    return poly[: ell * base].reshape(ell, base)


def shifted_selector(u: np.ndarray, i: int, base: int, q: int) -> np.ndarray:
    """X^{i*base} * u as a coefficient array (no wrap-around for i < ell)."""
    d = u.shape[0]
    shift = np.zeros(d, dtype=np.int64)
    shift[(i * base) % d] = 1 if (i * base) < d else q - 1
    return polymul(shift, u, q)


def link_first_value(u1: np.ndarray, w1: np.ndarray, a_i: np.ndarray, s: np.ndarray,
                     i: int, base: int, q: int) -> int:
    """const( X^{i base} u_1 sigma(w_1) - sigma(a_i)^T s ), which must be 0 (Fig. 9 / Lemma 4.1)."""
    left = polymul(shifted_selector(u1, i, base, q), conj_array(w1, q), q)
    right = polyvec_dot(conj_array(a_i, q), reduce(s, q), q)
    return int((left[0] - right[0]) % q)


def link_next_value(u_next: np.ndarray, w_next: np.ndarray, phi: np.ndarray, w_prev: np.ndarray,
                    i: int, base: int, q: int) -> int:
    """const( X^{i base} u_{j+1} sigma(w_{j+1}) - sigma(phi_{j,i}) w_j ), which must be 0 (Fig. 8)."""
    left = polymul(shifted_selector(u_next, i, base, q), conj_array(w_next, q), q)
    right = polymul(conj_array(reduce(phi, q), q), w_prev, q)
    return int((left[0] - right[0]) % q)


def level_step(T: np.ndarray, phis: np.ndarray, rest: list[np.ndarray] | None,
               base: int, d: int, q: int) -> tuple[np.ndarray, np.ndarray | None]:
    """One level of Fig. 9 / Fig. 8: fold T with the challenges and (prover only) compute w.

    rest = [v_{j+2}, ..., v_{k+1}] for the prover, or None for the verifier, who never sees
    the selectors. Returns (T_{j+1}, packed w_{j+1} or None).
    """
    T_next = fold(T, phis, base, q)
    if rest is None:
        return T_next, None
    w = pack_w(apply_selector(T_next, kron_all(rest), q), d, q)
    return T_next, w


__all__ = [
    "pack_w", "unpack_w", "shifted_selector", "link_first_value", "link_next_value",
    "level_step", "first_round_a", "fold",
]
