"""Parameter sets.

Everything here comes from the paper:
  - ring signature parameters: Section 4, Appendix C.2 / Fig. 10
  - challenge space parameters: Section 2.5 / Fig. 4

Notation (the paper uses a fraktur d for the second one, which is easy to misread):
  d     ring degree, R = Z[X]/(X^d + 1). Also the length of the LAST unit vector v_{k+1}.
  base  the paper's fraktur-d: length of the unit vectors v_1, ..., v_k.
  ell   the paper's fraktur-l: number of challenge vectors per recursion step.

So a ring has N = d * base^k members.

Gaussian widths for the proofs (s1, s2) aren't given explicitly in the paper. They depend on
the proof layer, so they'll be added once that layer exists.
"""

from __future__ import annotations

from dataclasses import dataclass


def is_prime(n: int) -> bool:
    """Trial division. Only used on ~17-bit moduli, so this is plenty fast."""
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True


@dataclass(frozen=True)
class ParamSet:
    name: str
    d: int          # ring degree, and length of v_{k+1}
    p: int          # ring signature modulus (public keys live mod p)
    q1: int         # proof system modulus q = q1 * q2
    q2: int
    n: int          # rows of A
    m: int          # columns of A, so secret keys are in {-1,1}^(m*d)
    base: int       # length of v_1..v_k (fraktur d in the paper)
    ell: int        # challenge vectors per recursion step (fraktur l), need ell * base <= d
    k: int          # recursion depth
    kappa: int      # challenge coefficients lie in [-kappa, kappa]
    eta: int        # challenge "operator norm" bound
    chal_pow: int   # the power-of-two exponent in the eta check of Fig. 4 (paper uses 32)

    def __post_init__(self) -> None:
        if self.d < 2 or self.d & (self.d - 1):
            raise ValueError(f"d must be a power of two, got {self.d}")
        if not 1 <= self.base <= self.d:
            raise ValueError(f"base must be in [1, d], got {self.base}")
        if self.ell < 1 or self.ell * self.base > self.d:
            # the ell vectors w_i of length `base` get packed into one polynomial of degree d
            raise ValueError(f"need ell * base <= d, got {self.ell} * {self.base} > {self.d}")
        if self.k < 0 or self.k % 2:
            # Lemma C.1: the relation is only sign-invariant (so bimodal sampling is free)
            # when k is even.
            raise ValueError(f"k must be a non-negative even number, got {self.k}")
        for name, prime in (("q1", self.q1), ("q2", self.q2)):
            if not is_prime(prime):
                raise ValueError(f"{name}={prime} is not prime")
            # Lemma 2.1 needs q_i = 5 mod 8 so that sigma_{-1}-invariant challenge
            # differences are invertible.
            if prime % 8 != 5:
                raise ValueError(f"{name}={prime} must be 5 mod 8")
        if self.q % self.p:
            raise ValueError("p must divide q (Appendix C.2)")
        if self.chal_pow < 1 or self.chal_pow & (self.chal_pow - 1):
            raise ValueError("chal_pow must be a power of two")

    @property
    def q(self) -> int:
        return self.q1 * self.q2

    @property
    def ring_size(self) -> int:
        """N = d * base^k, the number of users the ring signature hides among."""
        return self.d * self.base ** self.k

    @property
    def pk_bytes(self) -> int:
        """A public key is n polynomials mod p."""
        bytes_per_coeff = ((self.p - 1).bit_length() + 7) // 8
        return self.n * self.d * bytes_per_coeff

    @property
    def sk_bytes(self) -> int:
        """A secret key is m*d signs, so one bit each."""
        return (self.m * self.d + 7) // 8


def ring_signature_params(k: int) -> ParamSet:
    """The Fig. 10 parameter set for recursion depth k (ring size 64 * 8^k)."""
    return ParamSet(
        name=f"ringsig-k{k}",
        d=64,
        p=65437,
        q1=65437,
        q2=65629,
        n=1,
        m=12,
        base=8,
        ell=8,  # deviation from the paper, see below
        k=k,
        kappa=8,
        eta=140,
        chal_pow=32,
    )


# Deviation: the paper says d = ell * base and then picks (d, base, ell) = (64, 8, 16), which
# don't satisfy that (16 * 8 = 128). The packing of the ell vectors w_i in Z_q^base into one
# polynomial of degree d only works for ell * base <= d, so we use ell = 8. Each recursion
# step then has soundness error about q1^-8 ~ 2^-128, which still meets the paper's 128-bit
# target.

RINGSIG_K0 = ring_signature_params(0)  # N = 2^6
RINGSIG_K2 = ring_signature_params(2)  # N = 2^12
RINGSIG_K4 = ring_signature_params(4)  # N = 2^18
RINGSIG_K6 = ring_signature_params(6)  # N = 2^24

ALL = {p.name: p for p in (RINGSIG_K0, RINGSIG_K2, RINGSIG_K4, RINGSIG_K6)}
