"""Bit-packed encoding of signatures, to measure actual sizes (p. 29, Fig. 10).

Jie's email asks us to keep measured sizes separate from the paper's estimates. This module
produces the measured number: len(encode(signature)). bench/size.py prints it next to Fig. 10.

Format (all little-endian, fields in this fixed order; shapes come from the parameters):

    b"BLM1"                     magic + version
    mod-q arrays                t_A, t_1..t_k, t_y3, t_g, h, t        32 bits per coefficient
    c                           challenge, 5 bits per coefficient (|c_i| <= kappa = 8)
    z3, z1, z2                  one width byte w, then each coefficient as a w-bit two's
                                complement integer (w = bits of the largest |value| + 1)

The z's are the only part where a cleverer encoding (entropy coding, as LNP22 Sec. 6.1
assumes) would save much; size_breakdown() reports both the packed size and that estimate.
"""

from __future__ import annotations

import math

import numpy as np

from .base_case import BaseCaseProof
from .oom import OomProof
from .oom_params import JL_ROWS, ProofParams

MAGIC = b"BLM1"


def _pack_bits(values: np.ndarray, width: int) -> bytes:
    v = np.asarray(values, dtype=np.int64).ravel()
    bits = ((v[:, None] >> np.arange(width, dtype=np.int64)) & 1).astype(np.uint8)
    return np.packbits(bits.ravel(), bitorder="little").tobytes()


def _unpack_bits(data: bytes, count: int, width: int) -> np.ndarray:
    bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8), bitorder="little")[: count * width]
    if bits.size != count * width:
        raise ValueError("truncated field")
    bits = bits.reshape(count, width).astype(np.int64)
    return (bits << np.arange(width, dtype=np.int64)).sum(axis=1)


def _nbytes(count: int, width: int) -> int:
    return (count * width + 7) // 8


def _signed_width(arr: np.ndarray) -> int:
    m = int(np.max(np.abs(arr))) if arr.size else 0
    return max(2, m.bit_length() + 1)


class _Reader:
    def __init__(self, data: bytes) -> None:
        self.data, self.pos = data, 0

    def take(self, n: int) -> bytes:
        if self.pos + n > len(self.data):
            raise ValueError("signature too short")
        out = self.data[self.pos:self.pos + n]
        self.pos += n
        return out


def _modq_fields(pf: OomProof) -> list[np.ndarray]:
    b = pf.base
    return [pf.tA, pf.t_w, b.t_y3, b.t_g, b.h, b.t_quad]


def encode(pf: OomProof, pp: ProofParams) -> bytes:
    out = [MAGIC]
    for arr in _modq_fields(pf):
        out.append(_pack_bits(arr, 32))
    kappa = pp.ps.kappa
    out.append(_pack_bits(_centered_c(pf.base.c, pp) + kappa, _c_width(pp)))
    for arr in (pf.base.z3, pf.base.z1, pf.base.z2):
        w = _signed_width(arr)
        out.append(bytes([w]))
        out.append(_pack_bits(np.asarray(arr) & ((1 << w) - 1), w))
    return b"".join(out)


def _c_width(pp: ProofParams) -> int:
    return (2 * pp.ps.kappa).bit_length()


def _centered_c(c: np.ndarray, pp: ProofParams) -> np.ndarray:
    q = pp.q
    return np.where(c > q // 2, c - q, c)


def decode(data: bytes, pp: ProofParams) -> OomProof:
    """Inverse of encode. Raises ValueError on anything malformed (wrong length, bad magic)."""
    if not isinstance(data, (bytes, bytearray)):
        raise TypeError("signature must be bytes")
    r = _Reader(bytes(data))
    if r.take(4) != MAGIC:
        raise ValueError("not a BLOOM signature (bad magic)")
    d, q = pp.d, pp.q
    shapes = [(pp.n_com, d), (pp.k, d), (pp.n_y3, d), (pp.lam, d), (pp.lam, d), (d,)]
    modq = []
    for shape in shapes:
        count = int(np.prod(shape))
        arr = _unpack_bits(r.take(_nbytes(count, 32)), count, 32).reshape(shape)
        if arr.max() >= q:
            raise ValueError("coefficient not reduced mod q")
        modq.append(arr)
    cw = _c_width(pp)
    c = _unpack_bits(r.take(_nbytes(d, cw)), d, cw) - pp.ps.kappa
    c = c % q
    zs = []
    for shape in ((JL_ROWS,), (pp.m1, d), (pp.m2, d)):
        w = r.take(1)[0]
        if not 2 <= w <= 40:
            raise ValueError("bad width byte")
        count = int(np.prod(shape))
        raw = _unpack_bits(r.take(_nbytes(count, w)), count, w)
        vals = np.where(raw >= 1 << (w - 1), raw - (1 << w), raw)
        zs.append(vals.reshape(shape))
    if r.pos != len(r.data):
        raise ValueError("trailing bytes after signature")
    tA, t_w, t_y3, t_g, h, t_quad = modq
    base = BaseCaseProof(t_y3=t_y3, t_g=t_g, z3=zs[0], h=h, t_quad=t_quad, c=c, z1=zs[1], z2=zs[2])
    return OomProof(tA=tA, t_w=t_w, base=base)


def size_breakdown(pf: OomProof, pp: ProofParams) -> dict[str, float]:
    """Bytes per component as encoded, plus an entropy estimate for the Gaussian parts.

    The estimate uses log2(sqrt(2 pi e) * s) bits per coefficient for a discrete Gaussian of
    width s, which is how LNP22 Sec. 6.1 counts the z's; it's not what we actually send.
    """
    d = pp.d
    b = pf.base
    out = {
        "t_A": pp.n_com * d * 4,
        "t_1..t_k": pp.k * d * 4,
        "t_y3": pp.n_y3 * d * 4,
        "t_g": pp.lam * d * 4,
        "h": pp.lam * d * 4,
        "t (quadratic garbage)": d * 4,
        "c": _nbytes(d, _c_width(pp)),
    }
    for name, arr in (("z3", b.z3), ("z1", b.z1), ("z2", b.z2)):
        out[name] = 1 + _nbytes(arr.size, _signed_width(arr))
    out["total (encoded)"] = len(encode(pf, pp))
    ent = lambda s, n: n * math.log2(math.sqrt(2 * math.pi * math.e) * s) / 8  # noqa: E731
    out["z entropy estimate (z1+z2+z3)"] = (ent(pp.std1, pp.m1 * d) + ent(pp.std2, pp.m2 * d)
                                            + ent(pp.std3, JL_ROWS))
    return out
