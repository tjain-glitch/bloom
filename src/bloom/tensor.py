"""Signer index <-> tensor-product selectors (paper eq. 28-29, App. C.1).

A ring has N = d * base^k members (base is the paper's fraktur-d, 8 in Fig. 10). The signer
index i in [0, N) is written in mixed radix

    i = ((i_1 * base + i_2) * base + ... + i_k) * d + i_{k+1},   0 <= i_j < base,  0 <= i_{k+1} < d

and the one-hot vector e_i of length N factors as the Kronecker product

    e_i = v_1 (x) v_2 (x) ... (x) v_k (x) v_{k+1},   v_j = e_{i_j} in {0,1}^base,  v_{k+1} = e_{i_{k+1}} in {0,1}^d.

v_1 is the most significant digit: (v_1 (x) rest)[r * len(rest) + t] = v_1[r] * rest[t].
This matches how fold.py cuts T into column blocks T_{0,1} | ... | T_{0,base}.

In the ring signature (App. C.1) the selectors are signed: every v_j may be multiplied by the
same sign b, and the committed polynomials are the signed monomials u_j = b * X^{i_j}, whose
coefficient vector is v_j || 0^{d - base} (Section 4, footnote 13).
"""

from __future__ import annotations

from functools import reduce

import numpy as np

from .ring import Poly, Ring


def ring_size(d: int, base: int, k: int) -> int:
    """N = d * base^k (Section 4)."""
    return d * base**k


def index_to_digits(i: int, d: int, base: int, k: int) -> list[int]:
    """Signer index -> digits [i_1, ..., i_k, i_{k+1}] (i_1 most significant)."""
    n = ring_size(d, base, k)
    if not 0 <= i < n:
        raise ValueError(f"index {i} outside the ring [0, {n})")
    last = i % d
    i //= d
    digits = []
    for _ in range(k):
        digits.append(i % base)
        i //= base
    return list(reversed(digits)) + [last]


def digits_to_index(digits: list[int], d: int, base: int) -> int:
    """Inverse of index_to_digits."""
    *head, last = digits
    i = 0
    for dig in head:
        if not 0 <= dig < base:
            raise ValueError(f"digit {dig} out of range for base {base}")
        i = i * base + dig
    if not 0 <= last < d:
        raise ValueError(f"last digit {last} out of range for d={d}")
    return i * d + last


def selectors(i: int, d: int, base: int, k: int, sign: int = 1) -> list[np.ndarray]:
    """The unit vectors v_1, ..., v_{k+1} for index i, each multiplied by `sign` (+1 or -1)."""
    if sign not in (1, -1):
        raise ValueError("sign must be +1 or -1")
    digits = index_to_digits(i, d, base, k)
    vs = []
    for j, dig in enumerate(digits):
        length = d if j == k else base
        v = np.zeros(length, dtype=np.int64)
        v[dig] = sign
        vs.append(v)
    return vs


def kron_all(vectors: list[np.ndarray]) -> np.ndarray:
    """v_1 (x) v_2 (x) ... (x) v_last as a flat integer vector (eq. 29)."""
    if not vectors:
        return np.ones(1, dtype=np.int64)
    return reduce(np.kron, vectors)


def selector_poly(v: np.ndarray, ring: Ring) -> Poly:
    """u_j: the polynomial whose coefficient vector is v_j || 0^{d - len(v_j)}."""
    if len(v) > ring.d:
        raise ValueError("selector longer than the ring degree")
    c = np.zeros(ring.d, dtype=np.int64)
    c[: len(v)] = v
    return ring.from_array(c)


def describe(i: int, d: int, base: int, k: int) -> str:
    """Human-readable decomposition, used by the demo's recursion trace."""
    digits = index_to_digits(i, d, base, k)
    parts = [f"v_{j + 1} = e_{dig} (length {base})" for j, dig in enumerate(digits[:-1])]
    parts.append(f"v_{k + 1} = e_{digits[-1]} (length {d})")
    radix = " ".join(str(x) for x in digits)
    return f"index {i} = digits ({radix}) in radix ({', '.join([str(base)] * k + [str(d)])})\n  " + "\n  ".join(parts)
