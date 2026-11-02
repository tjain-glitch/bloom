"""Proof-system parameters for the one-out-of-many proof (Section 4, App. C, Lemma 2.8).

params.py holds what the paper fixes for the ring signature (Fig. 10: d, p, q, n, m, base,
ell, k and the challenge space of Fig. 4). The paper does NOT give the remaining knobs a
working prover needs, so this module derives them and says where each formula comes from:

    n_com      rows of A1, A2 in the ABDLOP commitment (Section 2.6)
    m2         length of the commitment randomness s2
    lam        number of garbage polynomials in the Z_q -> R_q lift (LNP22 Sec. 4.4); the
               lift has soundness error q1^-lam, so lam = ceil(128 / log2 q1) = 9
    gamma_i    rejection-sampling slack: s_i = gamma_i * T_i (Lemma 2.8)

IMPORTANT (Jie's email, point 4): n_com and m2 are FUNCTIONAL choices. They are sized like
the LNP22/Graz examples so that the commitment is plausibly binding and hiding, but we have
not run a lattice estimator on them, so no concrete security level is claimed. The values
the paper does fix (d, q, p, m, n, kappa, eta) are used unchanged.

Two presets:
  paper_scale(k)  Fig. 10 ring-signature parameters (m = 12, N = 64 * 8^k)
  toy(k)          tiny m and commitment sizes so the test-suite runs in seconds. Toy keys are
                  NOT secure (m = 2); toy parameters exist only to exercise the code paths.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

from .params import ParamSet, ring_signature_params
from .sampling import bimodal_M, rej1_M

# Rows of the Johnson-Lindenstrauss projection used for approximate shortness (LNP22 Sec. 2.6):
# 256 rows give the 2^-128 failure bound of the modular JL lemma.
JL_ROWS = 256
# JL constants for R <- Bin_1^{256 x n} (LNP22 Sec. 2.6, from [GHL21]):
#  * honest prover: ||R x|| <= sqrt(337) ||x|| except with probability ~2^-128 (the Graz
#    notebook uses the same sqrt(337) when it sets the JL Gaussian width);
#  * soundness: if ||R x + y mod q|| <= B then ||x|| <= B / sqrt(30), for B small enough
#    relative to q (we use this in no_wraparound_margin()).
# These are the constants as we read them from LNP22/Graz; we did not re-derive them.
JL_UPPER_SQ = 337
JL_LOWER_SQ = 30

REJECTION_MODES = ("rej1", "bimodal")


@dataclass(frozen=True)
class ProofParams:
    ps: ParamSet                   # what the paper fixes (Fig. 4 and Fig. 10)
    n_com: int                     # rows of A1 and A2
    m2_extra: int                  # m2 = n_com + (number of BDLOP rows) + m2_extra
    lam: int                       # garbage polynomials for the Z_q lift
    rejection: str = "rej1"        # "rej1" (Lemma 2.8.1) or "bimodal" (Lemma 2.8.2, Sec. 3)
    gamma1: float = 40.0           # s1 slack (z1 = y1 + c s1); Rej1 repetition rate 1.42
    gamma2: float = 40.0           # s2 slack (z2 = y2 + c s2)
    gamma3: float = 20.0           # JL slack (z3 = y3 + R s3); always Rej1, see base_case.py.
                                   # Capped by the no-wrap-around margin (must stay > 1).
    label: str = field(default="")

    def __post_init__(self) -> None:
        if self.rejection not in REJECTION_MODES:
            raise ValueError(f"rejection must be one of {REJECTION_MODES}")
        if self.ps.k < 1:
            raise ValueError("the recursive proof needs k >= 1 (k = 0 is a direct LNP22 proof, not implemented)")
        if self.rejection == "bimodal" and self.ps.k % 2:
            # Lemma C.1: only for even k is the relation sign-invariant.
            raise ValueError("bimodal rejection on s1 needs an even k (Lemma C.1)")
        if self.ps.ell * self.ps.base != self.ps.d:
            raise ValueError("this implementation packs w_j into one polynomial: need ell * base == d")
        if JL_ROWS % self.ps.d:
            raise ValueError("JL_ROWS must be a multiple of d")
        if min(self.gamma1, self.gamma2, self.gamma3) <= 0:
            raise ValueError("gammas must be positive")
        if self.no_wraparound_margin() <= 1:
            raise ValueError("JL bound too loose: binary/norm proofs could wrap around mod q")

    # --- sizes -----------------------------------------------------------------------------

    @property
    def d(self) -> int:
        return self.ps.d

    @property
    def q(self) -> int:
        return self.ps.q

    @property
    def k(self) -> int:
        return self.ps.k

    @property
    def m1(self) -> int:
        """Ajtai part s1 = (s, u_1, ..., u_k, v_{k+1}): m + k + 1 polynomials (Fig. 9)."""
        return self.ps.m + self.k + 1

    @property
    def n_y3(self) -> int:
        """Polynomials holding the 256 JL masks y3."""
        return JL_ROWS // self.d

    @property
    def n_vars_bdlop(self) -> int:
        """BDLOP messages that the relations talk about: w_1..w_k, y3, g_1..g_lam."""
        return self.k + self.n_y3 + self.lam

    @property
    def n_bdlop_rows(self) -> int:
        """All BDLOP rows: the message rows above plus one row b for the quadratic garbage t."""
        return self.n_vars_bdlop + 1

    @property
    def m2(self) -> int:
        return self.n_com + self.n_bdlop_rows + self.m2_extra

    @property
    def n_vars(self) -> int:
        """Length of x = s1 || m, the vector every relation is written in."""
        return self.m1 + self.n_vars_bdlop

    @property
    def n3(self) -> int:
        """Polynomials in the JL vector s3 = ((s+1)/2, u_1, ..., u_k, v_{k+1})."""
        return self.ps.m + self.k + 1

    @property
    def num_zq_relations(self) -> int:
        """Number of 'constant coefficient is zero' relations in Pi_k (see base_case.py)."""
        p = self.ps
        return (1                          # binary: <x, x - 1> = 0
                + (self.k + 1)             # ||u_j||^2 = 1, ||v_{k+1}||^2 = 1
                + (p.d - p.base) * self.k  # u_j only uses the first `base` coefficients
                + p.d                      # T_k v_{k+1} = w_k
                + p.ell * self.k           # Lemma 4.1 linking checks, Fig. 9 and Fig. 8 levels
                + JL_ROWS)                 # z3 = y3 + R s3

    # --- Gaussian widths (Lemma 2.8) ----------------------------------------------------------

    @property
    def alpha1(self) -> float:
        """||s1|| exactly: m*d entries of size 1 in s, and k+1 signed monomials of norm 1."""
        return math.sqrt(self.ps.m * self.d + self.k + 1)

    @property
    def alpha2(self) -> float:
        """Upper bound on ||s2|| for ternary s2."""
        return math.sqrt(self.m2 * self.d)

    @property
    def alpha3(self) -> float:
        """Upper bound on ||s3||: (s+1)/2 is binary (m*d entries), plus k+1 unit vectors."""
        return math.sqrt(self.ps.m * self.d + self.k + 1)

    @property
    def T1(self) -> float:
        """Bound on ||c s1|| <= eta ||s1|| (Section 2.5)."""
        return self.ps.eta * self.alpha1

    @property
    def T2(self) -> float:
        return self.ps.eta * self.alpha2

    @property
    def T3(self) -> float:
        """Bound on ||R s3|| (JL upper bound); the prover restarts in the rare case it fails."""
        return math.sqrt(JL_UPPER_SQ) * self.alpha3

    @property
    def std1(self) -> float:
        return self.gamma1 * self.T1

    @property
    def std2(self) -> float:
        return self.gamma2 * self.T2

    @property
    def std3(self) -> float:
        return self.gamma3 * self.T3

    @property
    def M1(self) -> float:
        return bimodal_M(self.gamma1) if self.rejection == "bimodal" else rej1_M(self.gamma1)

    @property
    def M2(self) -> float:
        return bimodal_M(self.gamma2) if self.rejection == "bimodal" else rej1_M(self.gamma2)

    @property
    def M3(self) -> float:
        return rej1_M(self.gamma3)

    @property
    def expected_repetitions(self) -> float:
        """Expected signing attempts: the three rejection steps are independent."""
        return self.M1 * self.M2 * self.M3

    # --- verifier bounds (Fig. 6/7: ||z_i|| <= s_i sqrt(2 m_i d)) ------------------------------

    @property
    def B1(self) -> float:
        return self.std1 * math.sqrt(2 * self.m1 * self.d)

    @property
    def B2(self) -> float:
        return self.std2 * math.sqrt(2 * self.m2 * self.d)

    @property
    def B3(self) -> float:
        return self.std3 * math.sqrt(2 * JL_ROWS)

    def no_wraparound_margin(self) -> float:
        """How far below q the binary and norm sums stay for anything that passes the JL check.

        If ||z3|| <= B3 then (JL soundness bound) ||s3|| <= B3 / sqrt(30) =: b. The binary proof shows
        sum x_i (x_i - 1) = 0 mod q and the norm proof ||u||^2 = 1 mod q; both only imply the
        statement over the integers if the integer sums cannot reach q, i.e. b^2 + sqrt(n) b < q.
        Returns q / (b^2 + sqrt(n) b); must be > 1.
        """
        b = self.B3 / math.sqrt(JL_LOWER_SQ)
        n = self.n3 * self.d
        return self.q / (b * b + math.sqrt(n) * b)

    def summary(self) -> dict:
        return {
            "label": self.label, "N": self.ps.ring_size, "k": self.k, "m1": self.m1, "m2": self.m2,
            "n_com": self.n_com, "lam": self.lam, "rejection": self.rejection,
            "std1": round(self.std1), "std2": round(self.std2), "std3": round(self.std3),
            "expected_repetitions": round(self.expected_repetitions, 2),
            "zq_relations": self.num_zq_relations,
        }


def lam_for(q1: int, bits: int = 128) -> int:
    """Smallest lam with q1^-lam <= 2^-bits."""
    return math.ceil(bits / math.log2(q1))


def paper_scale(k: int = 2, rejection: str = "rej1") -> ProofParams:
    """Fig. 10 ring-signature parameters plus our derived proof parameters."""
    ps = ring_signature_params(k)
    pp = ProofParams(ps=ps, n_com=16, m2_extra=16, lam=lam_for(ps.q1), rejection=rejection,
                     label=f"paper-k{k}-{rejection}")
    if rejection == "bimodal":
        # Same repetition rate per step as Rej1 with gamma = 40 (M ~ 1.42), but bimodal gets
        # there with gamma = 1.2: a ~33x narrower Gaussian, which is the point of Section 3.
        pp = replace(pp, gamma1=1.2, gamma2=1.2)
    return pp


def toy(k: int = 2, rejection: str = "rej1") -> ProofParams:
    """Small, insecure parameters for fast tests (m = 2, small commitments)."""
    ps = replace(ring_signature_params(k), name=f"toy-k{k}", m=2)
    pp = ProofParams(ps=ps, n_com=2, m2_extra=2, lam=lam_for(ps.q1), rejection=rejection,
                     label=f"toy-k{k}-{rejection}")
    if rejection == "bimodal":
        pp = replace(pp, gamma1=1.2, gamma2=1.2)
    return pp
