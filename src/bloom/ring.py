"""Polynomial arithmetic in R_q = Z_q[X]/(X^d + 1)  (paper Section 2.2).

Coefficients are stored as numpy arrays with dtype=object, i.e. plain Python ints. That's
slower than int64, but q ~ 2^32 means a product of two coefficients doesn't fit in int64,
and correctness matters more than speed at this stage. If benchmarks end up too slow, the
plan is to switch to CRT (work mod q1 and q2 separately, both < 2^17) behind the same API.

Coefficients are always kept reduced into [0, q). Use .centered() when you need the
representative in (-q/2, q/2], e.g. for norms.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np


def negacyclic_mul(a: np.ndarray, b: np.ndarray, q: int | None = None) -> np.ndarray:
    """Coefficients of a*b in Z[X]/(X^d + 1), reduced mod q if q is given.

    X^d = -1, so the top half of the full product wraps around with a sign flip.
    """
    d = len(a)
    full = np.convolve(a, b)            # length 2d - 1
    res = full[:d].copy()
    res[: d - 1] -= full[d:]
    if q is not None:
        res %= q
    return res


class Ring:
    """The ring R_q = Z_q[X]/(X^d + 1). Two rings are equal iff d and q match."""

    __slots__ = ("d", "q")

    def __init__(self, d: int, q: int) -> None:
        if d < 2 or d & (d - 1):
            raise ValueError(f"d must be a power of two >= 2, got {d}")
        if q < 2:
            raise ValueError(f"q must be >= 2, got {q}")
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
        return Poly(self, [0] * self.d)

    def one(self) -> Poly:
        return self.monomial(0)

    def monomial(self, i: int, coeff: int = 1) -> Poly:
        """coeff * X^i, for any integer i (uses X^d = -1, so X^(2d) = 1)."""
        i %= 2 * self.d
        sign = 1
        if i >= self.d:
            i -= self.d
            sign = -1
        c = [0] * self.d
        c[i] = sign * coeff
        return Poly(self, c)

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
        arr = np.array([int(x) for x in coeffs], dtype=object)
        if arr.shape != (ring.d,):
            raise ValueError(f"expected {ring.d} coefficients, got {arr.shape}")
        self.ring = ring
        self.c = arr % ring.q

    @classmethod
    def _raw(cls, ring: Ring, arr: np.ndarray) -> Poly:
        """Build from an already-reduced object array without re-validating (internal)."""
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
        if isinstance(other, int):
            return self + self.ring.monomial(0, other)
        self._check(other)
        return Poly._raw(self.ring, (self.c + other.c) % self.ring.q)

    __radd__ = __add__

    def __sub__(self, other: Poly | int) -> Poly:
        if isinstance(other, int):
            return self - self.ring.monomial(0, other)
        self._check(other)
        return Poly._raw(self.ring, (self.c - other.c) % self.ring.q)

    def __rsub__(self, other: int) -> Poly:
        return (-self) + other

    def __neg__(self) -> Poly:
        return Poly._raw(self.ring, (-self.c) % self.ring.q)

    def __mul__(self, other: Poly | int) -> Poly:
        if isinstance(other, int):
            return Poly._raw(self.ring, (self.c * other) % self.ring.q)
        self._check(other)
        return Poly._raw(self.ring, negacyclic_mul(self.c, other.c, self.ring.q))

    __rmul__ = __mul__

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Poly) and self.ring == other.ring and bool(np.all(self.c == other.c))

    def __hash__(self) -> int:
        return hash((self.ring, tuple(self.c)))

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
        out = np.zeros(d, dtype=object)
        for j, cj in enumerate(self.c):
            e = (i * j) % (2 * d)
            if e < d:
                out[e] += cj
            else:
                out[e - d] -= cj
        return Poly._raw(self.ring, out % self.ring.q)

    def conj(self) -> Poly:
        """sigma_{-1}. Lemma 2.2: the constant coeff of conj(u) * v is <u, v>."""
        return self.sigma(-1)

    # --- views and norms -------------------------------------------------------------

    @property
    def const(self) -> int:
        return int(self.c[0])

    def centered(self) -> np.ndarray:
        """Coefficients as integers in (-q/2, q/2]."""
        q = self.ring.q
        c = self.c.copy()
        c[c > q // 2] -= q
        return c

    def norm_inf(self) -> int:
        return int(max(abs(x) for x in self.centered()))

    def norm2_sq(self) -> int:
        return int(sum(x * x for x in self.centered()))

    def lift(self, ring: Ring) -> Poly:
        """Move to another modulus via the centred representative (e.g. mod p -> mod q)."""
        if ring.d != self.ring.d:
            raise ValueError("can only lift between rings of the same degree")
        return Poly(ring, self.centered())

    def rot(self) -> np.ndarray:
        """d x d integer matrix M with M @ coeffs(b) == coeffs(self * b) mod q.

        Column j is the coefficient vector of self * X^j. Section 4 uses this to turn
        A*s into a plain matrix-vector product over Z_q.
        """
        d = self.ring.d
        M = np.zeros((d, d), dtype=object)
        col = self.c.copy()
        for j in range(d):
            M[:, j] = col
            # multiply by X: shift up by one, top coefficient wraps to the bottom negated
            col = np.concatenate(([(-col[-1]) % self.ring.q], col[:-1]))
        return M

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
        acc = np.zeros(self.ring.d, dtype=object)
        for a, b in zip(self, other):
            acc += negacyclic_mul(a.c, b.c)
        return Poly._raw(self.ring, acc % self.ring.q)

    def conj(self) -> PolyVec:
        return PolyVec(self.ring, [a.conj() for a in self])

    def coeffs(self) -> np.ndarray:
        """Flat coefficient vector of length k*d (the paper's arrow-vector notation)."""
        if not self.polys:
            return np.zeros(0, dtype=object)
        return np.concatenate([p.c for p in self])

    def centered(self) -> np.ndarray:
        if not self.polys:
            return np.zeros(0, dtype=object)
        return np.concatenate([p.centered() for p in self])

    def norm_inf(self) -> int:
        return max((p.norm_inf() for p in self), default=0)

    def norm2_sq(self) -> int:
        return sum(p.norm2_sq() for p in self)

    def lift(self, ring: Ring) -> PolyVec:
        return PolyVec(ring, [p.lift(ring) for p in self])

    def to_bytes(self) -> bytes:
        return b"".join(p.to_bytes() for p in self)


class PolyMat:
    """A rows x cols matrix over R_q, stored row-major as lists of Polys."""

    __slots__ = ("ring", "rows")

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

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.rows), len(self.rows[0])

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
        return PolyVec(self.ring, [PolyVec(self.ring, row).dot(v) for row in self.rows])

    def rot(self) -> np.ndarray:
        """Block matrix of rot() of each entry, shape (rows*d, cols*d).

        rot(A) @ coeffs(s) == coeffs(A @ s) mod q, which is the A in eq. (28).
        """
        return np.block([[p.rot() for p in row] for row in self.rows])
