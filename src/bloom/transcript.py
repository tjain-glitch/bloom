"""Fiat-Shamir transcript.

The prover and verifier both feed the same public data into a Transcript in the same order,
and pull challenges out of it. As long as they absorb identical bytes, they get identical
challenges, which is what turns the interactive protocols in the paper into signatures.

Design:
  - one running SHAKE-256 state
  - every absorbed item is framed as (label, type tag, length, bytes), so two different
    sequences of items can never encode to the same byte string
  - squeezing a challenge copies the state, derives the challenge, then absorbs the result
    back in, so two challenges with the same label still come out different
"""

from __future__ import annotations

import hashlib

from .ring import Poly, PolyMat, PolyVec, Ring
from .sampling import XofRandom, challenge_poly

_DOMAIN = b"bloom-transcript-v1"


def _ring_header(ring: Ring) -> bytes:
    return ring.d.to_bytes(4, "little") + ring.q.to_bytes(8, "little")


def encode(value: object) -> tuple[bytes, bytes]:
    """Return (type tag, bytes) for anything we allow in a transcript."""
    if isinstance(value, (bytes, bytearray)):
        return b"B", bytes(value)
    if isinstance(value, str):
        return b"S", value.encode()
    if isinstance(value, bool):  # bool before int, since bool is a subclass of int
        return b"b", b"\x01" if value else b"\x00"
    if isinstance(value, int):
        if value < 0:
            raise ValueError("encode non-negative ints only (reduce mod q first)")
        return b"I", value.to_bytes((value.bit_length() + 7) // 8 or 1, "little")
    if isinstance(value, Poly):
        return b"P", _ring_header(value.ring) + value.to_bytes()
    if isinstance(value, PolyVec):
        return b"V", _ring_header(value.ring) + len(value).to_bytes(4, "little") + value.to_bytes()
    if isinstance(value, PolyMat):
        rows, cols = value.shape
        body = b"".join(p.to_bytes() for row in value.rows for p in row)
        return b"M", _ring_header(value.ring) + rows.to_bytes(4, "little") + cols.to_bytes(4, "little") + body
    raise TypeError(f"can't absorb a {type(value).__name__}")


class Transcript:
    def __init__(self, protocol: str) -> None:
        self._h = hashlib.shake_256()
        self._frame(b"domain", b"-", _DOMAIN)
        self._frame(b"protocol", b"S", protocol.encode())

    def _frame(self, label: bytes, tag: bytes, data: bytes) -> None:
        self._h.update(len(label).to_bytes(2, "little") + label)
        self._h.update(tag)
        self._h.update(len(data).to_bytes(8, "little") + data)

    def absorb(self, label: str, value: object) -> None:
        tag, data = encode(value)
        self._frame(label.encode(), tag, data)

    def challenge_bytes(self, label: str, n: int = 32) -> bytes:
        if n < 1:
            raise ValueError("n must be positive")
        h = self._h.copy()
        h.update(b"squeeze|" + label.encode())
        out = h.digest(n)
        self._frame(b"challenge|" + label.encode(), b"B", out)
        return out

    def rng(self, label: str) -> XofRandom:
        """A deterministic random stream bound to everything absorbed so far."""
        return XofRandom(self.challenge_bytes(label, 32))

    def challenge_poly(self, label: str, ring: Ring, kappa: int, eta: int, chal_pow: int) -> Poly:
        """A challenge c from the Section 2.5 challenge space."""
        return challenge_poly(ring, kappa, eta, chal_pow, self.rng(label))

    def challenge_ints(self, label: str, count: int, modulus: int) -> list[int]:
        """`count` uniform integers mod `modulus`, e.g. the gamma vectors in Section 4."""
        r = self.rng(label)
        return [r.randrange(modulus) for _ in range(count)]

    def fork(self) -> Transcript:
        """An independent copy, for when the verifier needs to replay part of the protocol."""
        t = Transcript.__new__(Transcript)
        t._h = self._h.copy()
        return t
