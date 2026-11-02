"""One folding step of the recursion: T_j -> T_{j+1} (Section 4, Figs. 8 and 9).

Notation (0-indexed levels; level 0 is Fig. 9, levels 1..k-1 are Fig. 8):

    T_j      integer matrix with `rows` rows and base * L columns
    T_{j,r}  its r-th block of L consecutive columns, r = 0..base-1
    phi_i    the i-th of ell challenge vectors (length `rows`) sent by the verifier

    T_{j+1,i} := [ phi_i^T T_{j,0} ; ... ; phi_i^T T_{j,base-1} ]     (base x L)
    T_{j+1}   := [ T_{j+1,0} ; ... ; T_{j+1,ell-1} ]                  (ell*base x L)

The identity that makes this work (Section 4, the display before Lemma 4.1):

    < T_j (v (x) rest), phi_i >  =  < v, T_{j+1,i} rest >  =  < v, w_{j+1,i} >

so checking the ell inner products <v, w_{j+1,i}> replaces one tensor factor by a short
vector w_{j+1} = T_{j+1} rest of length ell*base = d, which fits in one polynomial.

All arithmetic is mod q on int64 arrays (see ring.matmul_mod for the overflow handling).
"""

from __future__ import annotations

import numpy as np

from .ring import matmul_mod, reduce


def fold(T: np.ndarray, phis: np.ndarray, base: int, q: int) -> np.ndarray:
    """T_{j+1} from T_j and the ell challenge vectors phis (shape (ell, rows))."""
    rows, cols = T.shape
    ell = phis.shape[0]
    if phis.shape[1] != rows:
        raise ValueError(f"challenge length {phis.shape[1]} does not match {rows} rows")
    if cols % base:
        raise ValueError(f"{cols} columns cannot be split into {base} blocks")
    L = cols // base
    # (ell x rows) @ (rows x base*L) -> (ell x base*L); row i holds phi_i^T T_{j,r} for all r.
    folded = matmul_mod(phis, T, q)
    # Split each row into its base blocks of length L: row (i, r) of T_{j+1} is block r of row i.
    return folded.reshape(ell * base, L)


def first_round_a(A_rot: np.ndarray, phis: np.ndarray, m: int, d: int, q: int) -> np.ndarray:
    """a_i = A^T phi_i (Fig. 9), returned as polynomials: shape (ell, m, d).

    A_rot is the rotation matrix A = rot(A) in Z_q^{nd x md} from eq. (28). The coefficient
    vector of a_i is exactly A^T phi_i, so <s, A^T phi_i> = const coeff of sigma(a_i)^T s.
    """
    a = matmul_mod(phis, A_rot, q)          # (ell, m*d) = (A^T phi_i)^T for each i
    return a.reshape(phis.shape[0], m, d)


def apply_selector(T: np.ndarray, rest: np.ndarray, q: int) -> np.ndarray:
    """w = T @ rest mod q, where rest = v_{j+2} (x) ... (x) v_{k+1} (signed one-hot)."""
    return matmul_mod(T, reduce(rest, q)[:, None], q)[:, 0]


def fold_identity_holds(T: np.ndarray, v_first: np.ndarray, rest: np.ndarray,
                        phis: np.ndarray, base: int, q: int) -> bool:
    """Check <T (v (x) rest), phi_i> == <v, w_i> for every i (used in tests and the demo).

    This is the algebra behind one recursion level, with no commitments involved.
    """
    full = np.kron(v_first, rest)
    lhs = matmul_mod(phis, matmul_mod(T, reduce(full, q)[:, None], q), q)[:, 0]
    w = apply_selector(fold(T, phis, base, q), rest, q).reshape(phis.shape[0], base)
    rhs = matmul_mod(w, reduce(v_first, q)[:, None], q)[:, 0]
    return bool(np.all(lhs == rhs))
