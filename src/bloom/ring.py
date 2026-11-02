"""Polynomial arithmetic in R_q = Z_q[X]/(X^d + 1)  (paper Section 2.2).

Coefficients are stored as numpy int64 arrays, always reduced into [0, q). Use .centered()
when you need the representative in (-q/2, q/2], e.g. for norms.

Speed note (changed on mitch-base-implementation). The first version stored Python ints
(dtype=object) because q ~ 2^32 means a product of two coefficients does not fit in int64.
That was exact but far too slow for the LNP22 base case, which multiplies thousands of
polynomials per signature. We now keep int64 storage and avoid overflow by splitting one
operand into 16-bit halves ("limbs"):

    a * b = a * b_lo + 2^16 * (a * b_hi)      with b_lo, b_hi < 2^16

Each partial product is below 2^32 * 2^16 = 2^48, so a sum of up to 2^14 of them still fits
in a signed 64-bit integer. This works for any q < 2^32, which covers every parameter set in
the paper. Moduli >= 2^32 are rejected rather than silently overflowing.

The second half of the file has array-level helpers (polymul, matvec, matmul_mod, ...) that
work on plain int64 arrays of shape (..., d). The proof engine in base_case.py uses those
directly, because building thousands of Poly objects would dominate the running time.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

# Largest modulus the int64 limb trick supports (see module docstring).
MAX_Q = 1 << 32
# Inner-product length that is still overflow-free with 16-bit limbs: 2^48 * 2^14 = 2^62.
_MAX_TERMS = 1 << 14


# =====================================================================================
# Array-level helpers (no Poly objects). All inputs are int64 arrays reduced mod q.
# =====================================================================================


def negacyclic_mul(a: np.ndarray, b: np.ndarray, q: int | None = None) -> np.ndarray:
    """Coefficients of a*b in Z[X]/(X^d + 1), reduced mod q if q is given.

    Exact reference version (works on Python ints / dtype=object). X^d = -1, so the top
    half of the full product wraps around with a sign flip. Used by the challenge-space
    norm check, which needs exact integer powers of c.
    """
    d = len(a)
    full = np.convolve(a, b)            # length 2d - 1
    res = full[:d].copy()
    res[: d - 1] -= full[d:]
    if q is not None:
        res %= q
    return res


def _check_q(q: int) -> None:
    if q >= MAX_Q:
        raise ValueError(f"q must be < 2^32 for the int64 fast path, got {q}")


def reduce(a, q: int) -> np.ndarray:
    """Any integer array (int64 or Python ints) -> int64 array in [0, q)."""
    arr = np.asarray(a)
    if arr.dtype == object:
        return np.array([int(x) % q for x in arr.ravel()], dtype=np.int64).reshape(arr.shape)
    return np.mod(arr.astype(np.int64, copy=False), q)


def rot_batch(b: np.ndarray, q: int) -> np.ndarray:
    """Negacyclic matrices: shape (..., d) -> (..., d, d).

    M[..., i, j] is coefficient i of b * X^j, so M @ coeffs(a) = coeffs(b * a). This is the
    rot() map of Section 4 (eq. 28), applied to a whole batch at once.
    """
    d = b.shape[-1]
    i = np.arange(d)[:, None]
    j = np.arange(d)[None, :]
    idx = (i - j) % d
    neg = i < j                                   # these entries wrapped past X^d = -1
    m = b[..., idx]
    return np.where(neg, (q - m) % q, m)


def matmul_mod(a: np.ndarray, b: np.ndarray, q: int) -> np.ndarray:
    """Exact (a @ b) mod q for int64 arrays with entries in [0, q), q < 2^32.

    `a` is split into 16-bit limbs so no partial sum can overflow. Long inner dimensions
    are processed in chunks of 2^14.
    """
    _check_q(q)
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    k = a.shape[-1]
    if k > _MAX_TERMS:
        out = None
        for s in range(0, k, _MAX_TERMS):
            part = matmul_mod(a[..., s:s + _MAX_TERMS], b[..., s:s + _MAX_TERMS, :], q)
            out = part if out is None else (out + part) % q
        return out
    lo = a & 0xFFFF
    hi = a >> 16
    r_lo = np.matmul(lo, b) % q
    r_hi = np.matmul(hi, b) % q
    return (r_lo + (r_hi << 16) % q) % q


def polymul(a: np.ndarray, b: np.ndarray, q: int) -> np.ndarray:
    """Elementwise product in R_q of two (broadcastable) arrays of shape (..., d)."""
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    if a.ndim == 1 and b.ndim == 1:
        return _polymul_1d(a, b, q)
    a, b = np.broadcast_arrays(a, b)
    return _rot_apply(rot_batch(b, q), a, q)


def _polymul_1d(a: np.ndarray, b: np.ndarray, q: int) -> np.ndarray:
    """Single product via two integer convolutions (one per 16-bit limb of b)."""
    d = a.shape[0]
    out = np.zeros(d, dtype=np.int64)
    for shift, limb in ((0, b & 0xFFFF), (16, b >> 16)):
        full = np.convolve(a, limb)               # each entry < d * 2^48
        res = full[:d].copy()
        res[: d - 1] -= full[d:]
        res %= q
        out = (out + ((res << shift) % q)) % q
    return out


def _rot_apply(rb: np.ndarray, a: np.ndarray, q: int) -> np.ndarray:
    """(rb @ a) mod q for rb (..., d, d) and a (..., d), splitting a into limbs."""
    lo = a & 0xFFFF
    hi = a >> 16
    r_lo = np.einsum("...ij,...j->...i", rb, lo) % q
    r_hi = np.einsum("...ij,...j->...i", rb, hi) % q
    return (r_lo + (r_hi << 16) % q) % q


def polyvec_dot(a: np.ndarray, b: np.ndarray, q: int) -> np.ndarray:
    """sum_j a[j] * b[j] in R_q, for arrays of shape (n, d). Returns shape (d,)."""
    if a.shape[0] == 0:
        return np.zeros(a.shape[-1], dtype=np.int64)
    return matvec(a[None, :, :], b, q)[0]


def matvec(A: np.ndarray, v: np.ndarray, q: int) -> np.ndarray:
    """Matrix-vector product over R_q.

    A has shape (rows, cols, d) and v has shape (cols, d); the result has shape (rows, d).
    Implemented as sum_j rot(v_j) @ A[:, j], i.e. one big integer contraction.
    """
    rows, cols, d = A.shape
    if v.shape != (cols, d):
        raise ValueError(f"cannot multiply {A.shape[:2]} matrix by vector of shape {v.shape}")
    if cols * d > _MAX_TERMS:
        out = np.zeros((rows, d), dtype=np.int64)
        step = max(1, _MAX_TERMS // d)
        for s in range(0, cols, step):
            out = (out + matvec(A[:, s:s + step], v[s:s + step], q)) % q
        return out
    rv = rot_batch(v, q)                          # (cols, d, d)
    lo = A & 0xFFFF
    hi = A >> 16
    r_lo = np.einsum("cij,rcj->ri", rv, lo) % q
    r_hi = np.einsum("cij,rcj->ri", rv, hi) % q
    return (r_lo + (r_hi << 16) % q) % q


def scalar_mulmod(a: np.ndarray, k: int, q: int) -> np.ndarray:
    """(k * a) mod q for an int64 array a in [0, q) and any integer k."""
    k %= q
    lo, hi = k & 0xFFFF, k >> 16
    return ((a * lo) % q + (((a * hi) % q) << 16) % q) % q


def conj_array(a: np.ndarray, q: int) -> np.ndarray:
    """sigma_{-1} on a batch (..., d): X^i -> X^(-i) = -X^(d-i)."""
    out = np.empty_like(a)
    out[..., 0] = a[..., 0]
    out[..., 1:] = (q - a[..., :0:-1]) % q
    return out


def centered(a: np.ndarray, q: int) -> np.ndarray:
    """Representatives in (-q/2, q/2] of an int64 array in [0, q)."""
    return np.where(a > q // 2, a - q, a)


def norm2_sq(a: np.ndarray, q: int) -> int:
    """Exact squared Euclidean norm of the centred coefficients (Python int)."""
    c = centered(np.asarray(a, dtype=np.int64), q).astype(object).ravel()
    return int(np.dot(c, c)) if c.size else 0


# =====================================================================================
# Object API: Ring, Poly, PolyVec, PolyMat
# =====================================================================================


class Ring:
    """The ring R_q = Z_q[X]/(X^d + 1). Two rings are equal iff d and q match."""

    __slots__ = ("d", "q")

    def __init__(self, d: int, q: int) -> None:
        if d < 2 or d & (d - 1):
            raise ValueError(f"d must be a power of two >= 2, got {d}")
        if q < 2:
            raise ValueError(f"q must be >= 2, got {q}")
        _check_q(q)
        self.d = d
        self.q = q

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Ring) and self.d == other.d and self.q == other.q

    def __hash__(self) -> int:
        return hash((self.d, self.q))

    def __repr__(self) -> str:
        return f"Ring(d={self.d}, q={self.q})"

    @property
    def bytes_per_coeff(self) -> int:
        return ((self.q - 1).bit_length() + 7) // 8

    def __call__(self, coeffs: Iterable[int]) -> Poly:
        return Poly(self, coeffs)

    def zero(self) -> Poly:
        return Poly._raw(self, np.zeros(self.d, dtype=np.int64))

    def one(self) -> Poly:
        return self.monomial(0)

    def monomial(self, i: int, coeff: int = 1) -> Poly:
        """coeff * X^i, for any integer i (uses X^d = -1, so X^(2d) = 1)."""
        i %= 2 * self.d
        sign = 1
        if i >= self.d:
            i -= self.d
            sign = -1
        c = np.zeros(self.d, dtype=np.int64)
        c[i] = (sign * coeff) % self.q
        return Poly._raw(self, c)

    def from_array(self, arr: np.ndarray) -> Poly:
        """Wrap a length-d integer array (reduced mod q here)."""
        arr = reduce(arr, self.q)
        if arr.shape != (self.d,):
            raise ValueError(f"expected shape ({self.d},), got {arr.shape}")
        return Poly._raw(self, arr)

    def poly_from_bytes(self, data: bytes) -> Poly:
        """Inverse of Poly.to_bytes. Rejects wrong lengths and out-of-range coefficients."""
        w = self.bytes_per_coeff
        if len(data) != w * self.d:
            raise ValueError(f"expected {w * self.d} bytes, got {len(data)}")
        coeffs = [int.from_bytes(data[i * w:(i + 1) * w], "little") for i in range(self.d)]
        if any(c >= self.q for c in coeffs):
            raise ValueError("coefficient out of range")
        return Poly(self, coeffs)


class Poly:
    """An element of R_q."""

    __slots__ = ("ring", "c")

    def __init__(self, ring: Ring, coeffs: Iterable[int]) -> None:
        # int() rejects floats with a fractional part only if we check; we refuse floats
        # outright so 1.5 never silently becomes 1.
        vals = []
        for x in coeffs:
            if isinstance(x, (float, np.floating)):
                raise TypeError("polynomial coefficients must be integers, got a float")
            vals.append(int(x))
        if len(vals) != ring.d:
            raise ValueError(f"expected {ring.d} coefficients, got {len(vals)}")
        self.ring = ring
        self.c = np.array([v % ring.q for v in vals], dtype=np.int64)

    @classmethod
    def _raw(cls, ring: Ring, arr: np.ndarray) -> Poly:
        """Build from an already-reduced int64 array without re-validating (internal)."""
        p = cls.__new__(cls)
        p.ring = ring
        p.c = arr
        return p

    def _check(self, other: Poly) -> None:
        if not isinstance(other, Poly):
            raise TypeError(f"expected Poly, got {type(other).__name__}")
        if other.ring != self.ring:
            raise ValueError(f"ring mismatch: {self.ring} vs {other.ring}")

    # --- arithmetic ---------------------------------------------------------------

    def __add__(self, other: Poly | int) -> Poly:
        if isinstance(other, (int, np.integer)):
            return self + self.ring.monomial(0, int(other))
        self._check(other)
        return Poly._raw(self.ring, (self.c + other.c) % self.ring.q)

    __radd__ = __add__

    def __sub__(self, other: Poly | int) -> Poly:
        if isinstance(other, (int, np.integer)):
            return self - self.ring.monomial(0, int(other))
        self._check(other)
        return Poly._raw(self.ring, (self.c - other.c) % self.ring.q)

    def __rsub__(self, other: int) -> Poly:
        return (-self) + other

    def __neg__(self) -> Poly:
        return Poly._raw(self.ring, (-self.c) % self.ring.q)

    def __mul__(self, other: Poly | int) -> Poly:
        if isinstance(other, (int, np.integer)) and not isinstance(other, bool):
            return Poly._raw(self.ring, scalar_mulmod(self.c, int(other), self.ring.q))
        if not isinstance(other, Poly):
            raise TypeError(f"can't multiply Poly by {type(other).__name__}")
        self._check(other)
        return Poly._raw(self.ring, polymul(self.c, other.c, self.ring.q))

    __rmul__ = __mul__

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Poly) and self.ring == other.ring and bool(np.all(self.c == other.c))

    def __hash__(self) -> int:
        return hash((self.ring, tuple(int(x) for x in self.c)))

    def __repr__(self) -> str:
        nz = [(i, int(v)) for i, v in enumerate(self.centered()) if v]
        if len(nz) > 6:
            return f"Poly(d={self.ring.d}, q={self.ring.q}, {len(nz)} nonzero coeffs)"
        terms = " + ".join(f"{v}*X^{i}" for i, v in nz) or "0"
        return f"Poly({terms})"

    # --- automorphisms ------------------------------------------------------------

    def sigma(self, i: int) -> Poly:
        """The automorphism sigma_i: X -> X^i, for odd i (Section 2.2)."""
        d = self.ring.d
        if i % 2 == 0:
            raise ValueError("sigma_i is only an automorphism for odd i")
        out = np.zeros(d, dtype=np.int64)
        for j, cj in enumerate(self.c):
            e = (i * j) % (2 * d)
            if e < d:
                out[e] += cj
            else:
                out[e - d] -= cj
        return Poly._raw(self.ring, out % self.ring.q)

    def conj(self) -> Poly:
        """sigma_{-1}. Lemma 2.2: the constant coeff of conj(u) * v is <u, v>."""
        return Poly._raw(self.ring, conj_array(self.c, self.ring.q))

    # --- views and norms -------------------------------------------------------------

    @property
    def const(self) -> int:
        return int(self.c[0])

    def centered(self) -> np.ndarray:
        """Coefficients as integers in (-q/2, q/2]."""
        return centered(self.c, self.ring.q)

    def norm_inf(self) -> int:
        return int(np.max(np.abs(self.centered())))

    def norm2_sq(self) -> int:
        return norm2_sq(self.c, self.ring.q)

    def lift(self, ring: Ring) -> Poly:
        """Move to another modulus via the centred representative (e.g. mod p -> mod q)."""
        if ring.d != self.ring.d:
            raise ValueError("can only lift between rings of the same degree")
        return Poly._raw(ring, reduce(self.centered(), ring.q))

    def rot(self) -> np.ndarray:
        """d x d integer matrix M with M @ coeffs(b) == coeffs(self * b) mod q.

        Column j is the coefficient vector of self * X^j. Section 4 uses this to turn
        A*s into a plain matrix-vector product over Z_q. Returned with dtype=object so a
        plain `M.dot(v)` is exact; use rot_batch() for the fast int64 version.
        """
        return rot_batch(self.c, self.ring.q).astype(object)

    def to_bytes(self) -> bytes:
        w = self.ring.bytes_per_coeff
        return b"".join(int(x).to_bytes(w, "little") for x in self.c)


class PolyVec:
    """A column vector of polynomials in R_q^k."""

    __slots__ = ("ring", "polys")

    def __init__(self, ring: Ring, polys: Sequence[Poly]) -> None:
        for p in polys:
            if not isinstance(p, Poly) or p.ring != ring:
                raise ValueError("all entries must be Polys in the given ring")
        self.ring = ring
        self.polys = list(polys)

    @classmethod
    def zeros(cls, ring: Ring, k: int) -> PolyVec:
        return cls(ring, [ring.zero() for _ in range(k)])

    @classmethod
    def from_coeffs(cls, ring: Ring, coeffs: Sequence[int]) -> PolyVec:
        """Split a flat coefficient vector of length k*d into k polynomials."""
        d = ring.d
        if len(coeffs) % d:
            raise ValueError(f"length {len(coeffs)} is not a multiple of d={d}")
        return cls(ring, [Poly(ring, coeffs[i:i + d]) for i in range(0, len(coeffs), d)])

    @classmethod
    def from_array(cls, ring: Ring, arr: np.ndarray) -> PolyVec:
        """Wrap an integer array of shape (k, d)."""
        arr = reduce(arr, ring.q)
        if arr.ndim != 2 or arr.shape[1] != ring.d:
            raise ValueError(f"expected shape (k, {ring.d}), got {arr.shape}")
        return cls(ring, [Poly._raw(ring, row.copy()) for row in arr])

    @staticmethod
    def concat(*vecs: PolyVec) -> PolyVec:
        if not vecs:
            raise ValueError("need at least one vector")
        ring = vecs[0].ring
        return PolyVec(ring, [p for v in vecs for p in v.polys])

    def _check(self, other: PolyVec) -> None:
        if not isinstance(other, PolyVec):
            raise TypeError(f"expected PolyVec, got {type(other).__name__}")
        if other.ring != self.ring or len(other) != len(self):
            raise ValueError("vector ring/length mismatch")

    @property
    def arr(self) -> np.ndarray:
        """The coefficients as an int64 array of shape (k, d)."""
        if not self.polys:
            return np.zeros((0, self.ring.d), dtype=np.int64)
        return np.stack([p.c for p in self.polys])

    def __len__(self) -> int:
        return len(self.polys)

    def __getitem__(self, i: int) -> Poly:
        return self.polys[i]

    def __iter__(self):
        return iter(self.polys)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, PolyVec) and self.ring == other.ring and self.polys == other.polys

    def __repr__(self) -> str:
        return f"PolyVec(len={len(self)}, {self.ring})"

    def __add__(self, other: PolyVec) -> PolyVec:
        self._check(other)
        return PolyVec(self.ring, [a + b for a, b in zip(self, other)])

    def __sub__(self, other: PolyVec) -> PolyVec:
        self._check(other)
        return PolyVec(self.ring, [a - b for a, b in zip(self, other)])

    def __neg__(self) -> PolyVec:
        return PolyVec(self.ring, [-a for a in self])

    def scale(self, c: Poly | int) -> PolyVec:
        """Multiply every entry by the same polynomial (e.g. a challenge c) or integer."""
        return PolyVec(self.ring, [a * c for a in self])

    def dot(self, other: PolyVec) -> Poly:
        """sum_i self[i] * other[i] in R_q (no conjugation)."""
        self._check(other)
        return Poly._raw(self.ring, polyvec_dot(self.arr, other.arr, self.ring.q))

    def conj(self) -> PolyVec:
        return PolyVec(self.ring, [a.conj() for a in self])

    def coeffs(self) -> np.ndarray:
        """Flat coefficient vector of length k*d (the paper's arrow-vector notation)."""
        return self.arr.reshape(-1)

    def centered(self) -> np.ndarray:
        return centered(self.coeffs(), self.ring.q)

    def norm_inf(self) -> int:
        return max((p.norm_inf() for p in self), default=0)

    def norm2_sq(self) -> int:
        return norm2_sq(self.coeffs(), self.ring.q)

    def lift(self, ring: Ring) -> PolyVec:
        return PolyVec(ring, [p.lift(ring) for p in self])

    def to_bytes(self) -> bytes:
        return b"".join(p.to_bytes() for p in self)


class PolyMat:
    """A rows x cols matrix over R_q, stored row-major as lists of Polys."""

    __slots__ = ("ring", "rows", "_arr")

    def __init__(self, ring: Ring, rows: Sequence[Sequence[Poly]]) -> None:
        if not rows or not rows[0]:
            raise ValueError("matrix must be non-empty")
        width = len(rows[0])
        for r in rows:
            if len(r) != width:
                raise ValueError("ragged matrix")
            for p in r:
                if not isinstance(p, Poly) or p.ring != ring:
                    raise ValueError("all entries must be Polys in the given ring")
        self.ring = ring
        self.rows = [list(r) for r in rows]
        self._arr = None

    @classmethod
    def from_array(cls, ring: Ring, arr: np.ndarray) -> PolyMat:
        """Wrap an integer array of shape (rows, cols, d)."""
        arr = reduce(arr, ring.q)
        m = cls(ring, [[Poly._raw(ring, arr[i, j].copy()) for j in range(arr.shape[1])]
                       for i in range(arr.shape[0])])
        m._arr = arr
        return m

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.rows), len(self.rows[0])

    @property
    def arr(self) -> np.ndarray:
        """The coefficients as an int64 array of shape (rows, cols, d). Cached."""
        if self._arr is None:
            self._arr = np.stack([np.stack([p.c for p in row]) for row in self.rows])
        return self._arr

    def __getitem__(self, ij: tuple[int, int]) -> Poly:
        i, j = ij
        return self.rows[i][j]

    def __eq__(self, other: object) -> bool:
        return isinstance(other, PolyMat) and self.ring == other.ring and self.rows == other.rows

    def __repr__(self) -> str:
        return f"PolyMat(shape={self.shape}, {self.ring})"

    def __matmul__(self, v: PolyVec) -> PolyVec:
        if not isinstance(v, PolyVec):
            raise TypeError(f"expected PolyVec, got {type(v).__name__}")
        if v.ring != self.ring or len(v) != self.shape[1]:
            raise ValueError(f"cannot multiply {self.shape} matrix by length-{len(v)} vector")
        return PolyVec.from_array(self.ring, matvec(self.arr, v.arr, self.ring.q))

    def rot(self) -> np.ndarray:
        """Block matrix of rot() of each entry, shape (rows*d, cols*d).

        rot(A) @ coeffs(s) == coeffs(A @ s) mod q, which is the A in eq. (28).
        Returned with dtype=object so a plain `.dot` is exact.
        """
        return np.block([[p.rot() for p in row] for row in self.rows])
