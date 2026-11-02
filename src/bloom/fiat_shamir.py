"""Protocol-specific Fiat-Shamir for the BLOOM ring signature (Section 4, App. C.2).

transcript.py is the generic hash state. This module fixes WHAT goes into it and in which
order, because that order is the security argument for statement binding: every challenge
must depend on everything the verifier has seen so far.

Order (prover and verifier must call these in exactly this order):

    1. absorb_statement:  protocol id, all parameters, commitment-key seed, key matrix A,
                          the FULL ordered ring of public keys T, the message mu
    2. absorb t_A                                        (Fig. 9, first message)
    3. for each level j = 0..k-1:  phis_j <- H;  absorb t_{j+1}   (Figs. 9 and 8)
    4. base case (base_case.py):  absorb t_y3, t_g;  R <- H;  absorb z3;  gamma <- H;
                                  absorb h;  mu_rq <- H;  absorb t, w, v;  c <- H

A team-reported bug from the consultation minutes is the reason step 1 is so explicit: a
transcript that skipped the ring and the message accepted modified statements every time.
tests/test_statement_binding.py checks that each item really is bound.
"""

from __future__ import annotations

import hashlib

import numpy as np

from .oom_params import ProofParams
from .ring import Poly, Ring
from .sampling import _uniform_from_bytes, binomial_array, challenge_poly
from .transcript import Transcript

PROTOCOL_ID = "bloom-ringsig-v1"


def _array_bytes(arr: np.ndarray) -> bytes:
    """Canonical encoding: shape header, then little-endian uint32 entries (all < 2^32)."""
    arr = np.asarray(arr, dtype=np.int64)
    if arr.size and (arr.min() < 0 or arr.max() >= 1 << 32):
        raise ValueError("absorb arrays reduced into [0, 2^32)")
    header = len(arr.shape).to_bytes(1, "little") + b"".join(x.to_bytes(4, "little") for x in arr.shape)
    return header + arr.astype("<u4").tobytes()


class OomTranscript:
    """Thin wrapper that names every absorb/squeeze of the protocol."""

    def __init__(self, pp: ProofParams) -> None:
        self.pp = pp
        self.ring = Ring(pp.d, pp.q)
        self._t = Transcript(PROTOCOL_ID)

    def fork(self) -> "OomTranscript":
        """Independent copy (each signing attempt restarts from the statement state)."""
        other = OomTranscript.__new__(OomTranscript)
        other.pp, other.ring, other._t = self.pp, self.ring, self._t.fork()
        return other

    # --- absorbing -----------------------------------------------------------------------

    def absorb_statement(self, ck_seed: bytes, A_keys: np.ndarray, ring_T: np.ndarray, message: bytes) -> None:
        pp, ps = self.pp, self.pp.ps
        self._t.absorb("params", repr((ps.name, ps.d, ps.p, ps.q1, ps.q2, ps.n, ps.m, ps.base, ps.ell,
                                       ps.k, ps.kappa, ps.eta, ps.chal_pow, pp.n_com, pp.m2, pp.lam,
                                       pp.rejection, pp.gamma1, pp.gamma2, pp.gamma3)))
        self._t.absorb("ck-seed", ck_seed)
        self._t.absorb("key-matrix-A", _array_bytes(A_keys))
        self._t.absorb("ring-T", _array_bytes(ring_T))
        if not isinstance(message, (bytes, bytearray)):
            raise TypeError("message must be bytes")
        self._t.absorb("message", bytes(message))

    def absorb_array(self, label: str, arr: np.ndarray) -> None:
        self._t.absorb(label, _array_bytes(arr))

    # --- squeezing -----------------------------------------------------------------------

    def _seed(self, label: str) -> bytes:
        return self._t.challenge_bytes(label, 32)

    def uniform(self, label: str, shape: tuple[int, ...]) -> np.ndarray:
        """Uniform challenge array mod q (phis, gamma, mu)."""
        seed = self._seed(label)
        counter = [0]

        def next_bytes(n: int) -> bytes:
            counter[0] += 1
            return hashlib.shake_256(b"bloom-chal|" + seed + counter[0].to_bytes(4, "little")).digest(n)

        count = int(np.prod(shape))
        return _uniform_from_bytes(next_bytes, count, self.pp.q).reshape(shape)

    def phis(self, level: int, rows: int) -> np.ndarray:
        """The ell challenge vectors of recursion level `level` (Figs. 9 and 8)."""
        return self.uniform(f"phi-level-{level}", (self.pp.ps.ell, rows))

    def jl_matrix(self, cols: int) -> np.ndarray:
        """JL projection R in {-1,0,1}^{256 x cols} (LNP22 approximate shortness)."""
        from .oom_params import JL_ROWS
        return binomial_array((JL_ROWS, cols), self._seed("jl-matrix"), "jl")

    def challenge_c(self) -> Poly:
        """c from the challenge space C of Section 2.5 (Fig. 4: kappa = 8, eta = 140)."""
        ps = self.pp.ps
        return self._t.challenge_poly("c", self.ring, ps.kappa, ps.eta, ps.chal_pow)


__all__ = ["OomTranscript", "PROTOCOL_ID", "challenge_poly"]
