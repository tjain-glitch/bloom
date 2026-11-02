"""Base case Pi_k: a zero-knowledge proof that the committed values satisfy R_k (p. 19).

This is the part Jie's email is about. After k folds (Figs. 9 and 8) the prover has committed
to s1 = (s, u_1, ..., u_k, v_{k+1}) in the Ajtai part and to w_1, ..., w_k in the BDLOP part.
Pi_k must convince the verifier that these satisfy every condition of R_k WITHOUT revealing
them. The paper (p. 19 and App. C.1) says each condition is one of the LNP22 statement types
of Section 2.7; this file writes them down and implements the LNP22 machinery that proves them.

    R_k condition (App. C version)                           how it is proved (LNP22 type)
    ----------------------------------------------------------------------------------------
    1  s in {-1,1}^{md}       i.e. x = (s+1)/2 binary        Type 3: JL bound on x + <x, x-1> = 0
    2  u_1..u_k, v_{k+1} signed monomials, i.e. ||u||^2 = 1  Type 4: JL bound on u + const(sigma(u)u) = 1
    3  u_j uses only the first `base` coefficients           Type 2: const(X^{-t} u_j) = 0, t >= base
    4  T_k v_{k+1} = w_k                                     Type 2: d linear equations
    5  const(X^{i base} u_1 sigma(w_1) - sigma(a_i)^T s) = 0  Type 2: quadratic (Lemma 4.1, Fig. 9)
    6  const(X^{i base} u_{j+1} sigma(w_{j+1}) - sigma(phi_{j,i}) w_j) = 0   Type 2 (Fig. 8)

All six are "the constant coefficient of a quadratic expression is zero" (Type 2), except that
1 and 2 also need the approximate shortness proof (Type 5) so the integer sums cannot wrap
around mod q (see oom_params.no_wraparound_margin).

The protocol, in the order the transcript sees it (made non-interactive with Fiat-Shamir):

    a. JL approximate shortness (LNP22): commit to masks y3, get R <- H, send z3 = y3 + R s3
       with rejection sampling. Adds 256 more Type-2 rows: z3_r = y3_r + <R_r, s3>.
    b. Z_q -> R_q lift (LNP22): commit to garbage g_1..g_lam with zero constant coefficient,
       get gamma <- H, send h_i = g_i + sum_j gamma_ij F_j. The verifier checks const(h_i) = 0:
       if any F_j had a nonzero constant coefficient this fails except w.p. q1^-lam.
    c. Collapse the lam R_q relations "g_i + sum_j gamma_ij F_j - h_i = 0" into ONE relation
       with random polynomial weights mu <- H.
    d. Prove that one quadratic relation with the commit-and-prove protocol of Fig. 7
       (BLOOM Sec. 3.2 / LNP22 Sec. 4): masks y1, y2, garbage t = b^T s2 + g1, value v,
       challenge c <- H, responses z1 = y1 + c s1, z2 = y2 + c s2 with rejection sampling.

Reference implementation used for cross-checking: the TU Graz "lattice-zk" notebook
(Lugstein, 2023), functions abdlop_quadratic_poly / abdlop_multiple_quadratic /
abdlop_single_quadratic / abdlop_toolbox. That code is interactive, SageMath-only, uses a
prime modulus and carries no licence, so nothing was copied; this file is our own
implementation of the same LNP22 steps, non-interactive and for q = q1 q2.

Deliberate simplifications (also in the README):
  * The JL step uses standard Rej1 (no sign bit). LNP22 hides R s3 with a committed sign and
    bimodal sampling; Rej1 needs a wider Gaussian for z3 (~100 bytes) but no sign proof.
  * No proof-size optimisations from LNP22 Sec. 6.1 (dropping low bits of t_A, etc.).

The OpeningChecker at the end opens every commitment and checks R_k in the clear. It reveals
the whole witness, so it is a DEBUGGING ORACLE ONLY (tests, bench/naive_verifier.py); the ring
signature never uses it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import abdlop
from .fiat_shamir import OomTranscript
from .oom_params import JL_ROWS, ProofParams
from .recursion import link_first_value, link_next_value
from .ring import (centered, conj_array, matmul_mod, matvec, polymul, polyvec_dot, reduce, rot_batch,
                   scalar_mulmod)
from .sampling import gaussian_array, rej1, rej_bimodal, uniform_array


# =====================================================================================
# Result type shared by every verifier in the package
# =====================================================================================


@dataclass
class VerifyResult:
    ok: bool
    reason: str = "accepted"
    failed: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.ok

    @classmethod
    def reject(cls, reason: str, failed: list[str] | None = None) -> "VerifyResult":
        return cls(False, reason, failed or [reason])


# =====================================================================================
# Variable layout: x = s1 || m, the vector every relation is written in
# =====================================================================================


class Layout:
    """Index of every committed polynomial inside x = s1 || m.

    Ajtai part s1:  S_0..S_{m-1} (secret key s), U_1..U_k, V (= v_{k+1})
    BDLOP part m:   W_1..W_k, Y_0..Y_{n_y3-1} (JL masks), G_0..G_{lam-1} (garbage)
    The BDLOP variables are in the same order as the rows of B in abdlop.py.
    """

    def __init__(self, pp: ProofParams) -> None:
        self.pp = pp
        m, k = pp.ps.m, pp.k
        self.m1 = pp.m1
        self.n = pp.n_vars

        self.S = list(range(0, m))
        self.U = [None] + list(range(m, m + k))           # 1-indexed like the paper: U[1]..U[k]
        self.V = m + k
        self.W = [None] + list(range(self.m1, self.m1 + k))
        self.Y = list(range(self.m1 + k, self.m1 + k + pp.n_y3))
        self.G = list(range(self.m1 + k + pp.n_y3, self.n))

    @property
    def bdlop_vars(self) -> range:
        return range(self.m1, self.n)


# =====================================================================================
# Relation algebra: batches of quadratic expressions with automorphisms (eq. 16 / 17)
# =====================================================================================


class Relations:
    """A batch F_0, ..., F_{count-1} of quadratic expressions in x with sigma_{-1}.

    Each F_r = sum a * sig^ci(x_i) sig^cj(x_j)  +  sum b * sig^ci(x_i)  +  r0, where
    sig^0 = identity, sig^1 = sigma_{-1}, and a, b, r0 are polynomials. Stored sparsely as
    {key: coefficient array of shape (count, d)} with keys
        ("q", i, ci, j, cj)   quadratic term
        ("l", i, ci)          linear term
        ("c",)                constant
    Storing by key (rather than one object per relation) lets us combine hundreds of
    relations with a random matrix in one matrix product per key.
    """

    def __init__(self, count: int, d: int, q: int) -> None:
        self.count, self.d, self.q = count, d, q
        self.terms: dict[tuple, np.ndarray] = {}

    # --- building ----------------------------------------------------------------------

    def _add(self, rows, key: tuple, coeffs) -> None:
        if key not in self.terms:
            self.terms[key] = np.zeros((self.count, self.d), dtype=np.int64)
        arr = self.terms[key]
        arr[rows] = (arr[rows] + reduce(np.asarray(coeffs), self.q)) % self.q

    def quad(self, rows, i: int, ci: int, j: int, cj: int, coeffs) -> None:
        self._add(rows, ("q", i, ci, j, cj), coeffs)

    def lin(self, rows, i: int, ci: int, coeffs) -> None:
        self._add(rows, ("l", i, ci), coeffs)

    def const(self, rows, coeffs) -> None:
        self._add(rows, ("c",), coeffs)

    @staticmethod
    def stack(parts: list["Relations"]) -> "Relations":
        d, q = parts[0].d, parts[0].q
        out = Relations(sum(p.count for p in parts), d, q)
        start = 0
        for p in parts:
            for key, arr in p.terms.items():
                out._add(slice(start, start + p.count), key, arr)
            start += p.count
        return out

    # --- combining -----------------------------------------------------------------------

    def combine(self, W: np.ndarray) -> "Relations":
        """New batch G_o = sum_r W[o, r] F_r with integer weights mod q (the gamma step)."""
        out = Relations(W.shape[0], self.d, self.q)
        for key, arr in self.terms.items():
            out.terms[key] = matmul_mod(W, arr, self.q)
        return out

    def combine_poly(self, mu: np.ndarray) -> "Relations":
        """Single relation sum_r mu_r * F_r with polynomial weights mu (shape (count, d))."""
        out = Relations(1, self.d, self.q)
        rmu = rot_batch(mu, self.q)                                # (count, d, d)
        for key, arr in self.terms.items():
            prod = _rot_apply_batch(rmu, arr, self.q)              # mu_r * coeff_r
            out.terms[key] = (prod.sum(axis=0, keepdims=True) % self.q)
        return out

    # --- evaluating ----------------------------------------------------------------------

    def _times(self, coeffs: np.ndarray, p: np.ndarray) -> np.ndarray:
        """Every row of `coeffs` (count, d) multiplied by the single polynomial p (d,)."""
        return matmul_mod(coeffs, rot_batch(p, self.q).T, self.q)

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """F_r(x) for every r, shape (count, d). x has shape (n_vars, d), reduced mod q."""
        q = self.q
        xs = (x, conj_array(x, q))
        out = np.zeros((self.count, self.d), dtype=np.int64)
        for key, arr in self.terms.items():
            if key[0] == "q":
                _, i, ci, j, cj = key
                out += self._times(arr, polymul(xs[ci][i], xs[cj][j], q))
            elif key[0] == "l":
                _, i, ci = key
                out += self._times(arr, xs[ci][i])
            else:
                out += arr
            out %= q
        return out

    def homogeneous(self, z: np.ndarray, c: np.ndarray) -> np.ndarray:
        """sum a sig(z_i) sig(z_j) + c * sum b sig(z_i) + c^2 r0   (the verifier's side of Fig. 7).

        If z = y + c x then this equals c^2 F(x) + c g1 + g0 (see garbage()).
        """
        q = self.q
        zs = (z, conj_array(z, q))
        quad = np.zeros((self.count, self.d), dtype=np.int64)
        lin = np.zeros_like(quad)
        const = np.zeros_like(quad)
        for key, arr in self.terms.items():
            if key[0] == "q":
                _, i, ci, j, cj = key
                quad = (quad + self._times(arr, polymul(zs[ci][i], zs[cj][j], q))) % q
            elif key[0] == "l":
                _, i, ci = key
                lin = (lin + self._times(arr, zs[ci][i])) % q
            else:
                const = (const + arr) % q
        c2 = polymul(c, c, q)
        return (quad + self._times(lin, c) + self._times(const, c2)) % q

    def garbage(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(g1, g0) with  F_hom(y + c x) = c^2 F(x) + c g1 + g0  (eq. 25 generalised).

        g1 = sum a (sig(y_i) sig(x_j) + sig(x_i) sig(y_j)) + sum b sig(y_i)
        g0 = sum a sig(y_i) sig(y_j)
        """
        q = self.q
        xs = (x, conj_array(x, q))
        ys = (y, conj_array(y, q))
        g1 = np.zeros((self.count, self.d), dtype=np.int64)
        g0 = np.zeros_like(g1)
        for key, arr in self.terms.items():
            if key[0] == "q":
                _, i, ci, j, cj = key
                cross = (polymul(ys[ci][i], xs[cj][j], q) + polymul(xs[ci][i], ys[cj][j], q)) % q
                g1 = (g1 + self._times(arr, cross)) % q
                g0 = (g0 + self._times(arr, polymul(ys[ci][i], ys[cj][j], q))) % q
            elif key[0] == "l":
                _, i, ci = key
                g1 = (g1 + self._times(arr, ys[ci][i])) % q
        return g1, g0


def _rot_apply_batch(rmu: np.ndarray, arr: np.ndarray, q: int) -> np.ndarray:
    """Row-wise products mu_r * arr_r for rmu = rot(mu) of shape (count, d, d)."""
    lo, hi = arr & 0xFFFF, arr >> 16
    r_lo = np.einsum("rij,rj->ri", rmu, lo) % q
    r_hi = np.einsum("rij,rj->ri", rmu, hi) % q
    return (r_lo + (r_hi << 16) % q) % q


# =====================================================================================
# Small polynomial constants
# =====================================================================================


def _scalar(value: int, d: int, q: int) -> np.ndarray:
    out = np.zeros(d, dtype=np.int64)
    out[0] = value % q
    return out


def _monomial(e: int, d: int, q: int) -> np.ndarray:
    """X^e for any integer e (X^d = -1)."""
    e %= 2 * d
    out = np.zeros(d, dtype=np.int64)
    if e < d:
        out[e] = 1
    else:
        out[e - d] = q - 1
    return out


# =====================================================================================
# R_k written as relations (rows 1-6 of the table in the module docstring)
# =====================================================================================


@dataclass
class BaseStatement:
    """Public input of Pi_k: everything the verifier recomputed during the recursion."""

    T_k: np.ndarray             # (d, d): last folded matrix (Fig. 8 at j = k-1)
    a: np.ndarray               # (ell, m, d): a_i = A^T phi_i from Fig. 9
    phis: list[np.ndarray]      # phis[j] has shape (ell, rows); phis[1..k-1] are used in row 6
    tA: np.ndarray              # (n_com, d)
    t_w: np.ndarray             # (k, d): commitments t_j = b_j^T s2 + w_j


@dataclass
class BaseWitness:
    s1: np.ndarray              # (m1, d) small integers: (s, u_1..u_k, v_{k+1}), possibly sign-flipped
    s2: np.ndarray              # (m2, d) ternary commitment randomness
    w: np.ndarray               # (k, d) packed w_1..w_k, reduced mod q


def rk_relations(pp: ProofParams, lay: Layout, st: BaseStatement) -> Relations:
    """Rows 1-6 of R_k as 'constant coefficient is zero' relations (before JL rows)."""
    ps, d, q, k = pp.ps, pp.d, pp.q, pp.k
    base, ell = ps.base, ps.ell
    count = pp.num_zq_relations - JL_ROWS
    F = Relations(count, d, q)
    inv2 = (q + 1) // 2
    inv4 = (inv2 * inv2) % q
    ones = np.ones(d, dtype=np.int64)
    sig_ones = conj_array(ones, q)
    row = 0

    # Row 1 (binary). x_p = (S_p + 1)/2;  sum_p const( sigma(x_p) (x_p - 1) ) = sum_i x_i (x_i - 1) = 0.
    # Expanding with sigma(x_p) = (sigma(S_p) + sigma(1))/2 and x_p - 1 = (S_p - 1)/2:
    for p in lay.S:
        F.quad(row, p, 1, p, 0, _scalar(inv4, d, q))
        F.lin(row, p, 1, scalar_mulmod(ones, -inv4, q))
        F.lin(row, p, 0, scalar_mulmod(sig_ones, inv4, q))
        F.const(row, scalar_mulmod(polymul(sig_ones, ones, q), -inv4, q))
    row += 1

    # Row 2 (exact norm). const( sigma(u) u ) = ||u||^2 = 1 for u_1..u_k and v_{k+1}.
    for var in lay.U[1:] + [lay.V]:
        F.quad(row, var, 1, var, 0, _scalar(1, d, q))
        F.const(row, _scalar(-1, d, q))
        row += 1

    # Row 3 (support). Coefficient t of u_j is const( X^{-t} u_j ) = 0 for base <= t < d.
    for j in range(1, k + 1):
        for t in range(base, d):
            F.lin(row, lay.U[j], 0, _monomial(-t, d, q))
            row += 1

    # Row 4. (T_k v_{k+1})_r - (w_k)_r = const( sigma(T_k[r]) v - sigma(X^r) w_k ) = 0.
    T_rows = conj_array(reduce(st.T_k, q), q)                     # sigma of each row as a poly
    F.lin(slice(row, row + d), lay.V, 0, T_rows)
    F.lin(slice(row, row + d), lay.W[k], 0,
          np.stack([(q - _monomial(-r, d, q)) % q for r in range(d)]))
    row += d

    # Row 5 (Fig. 9 linking, Lemma 4.1). const( X^{i base} u_1 sigma(w_1) - sigma(a_i)^T s ) = 0.
    for i in range(ell):
        F.quad(row, lay.U[1], 0, lay.W[1], 1, _monomial(i * base, d, q))
        for p in lay.S:
            F.lin(row, p, 0, (q - conj_array(st.a[i, p], q)) % q)
        row += 1

    # Row 6 (Fig. 8 linking). const( X^{i base} u_{j+1} sigma(w_{j+1}) - sigma(phi_{j,i}) w_j ) = 0.
    for j in range(1, k):
        for i in range(ell):
            F.quad(row, lay.U[j + 1], 0, lay.W[j + 1], 1, _monomial(i * base, d, q))
            F.lin(row, lay.W[j], 0, (q - conj_array(reduce(st.phis[j][i], q), q)) % q)
            row += 1

    assert row == count, (row, count)
    return F


def jl_relations(pp: ProofParams, lay: Layout, R: np.ndarray, z3: np.ndarray) -> Relations:
    """The 256 JL rows: z3_r = y3_r + <R_r, s3>, written as const(...) = 0.

    s3 = (x_0..x_{m-1}, u_1..u_k, v_{k+1}) with x_p = (S_p + 1)/2, so it is an affine function
    of s1, and y3_r is coefficient r mod d of the mask polynomial Y_{r // d}.
    """
    d, q, m = pp.d, pp.q, pp.ps.m
    inv2 = (q + 1) // 2
    F = Relations(JL_ROWS, d, q)
    Rb = reduce(R, q).reshape(JL_ROWS, pp.n3, d)
    ones = np.ones(d, dtype=np.int64)
    all_rows = slice(0, JL_ROWS)
    const = np.zeros((JL_ROWS, d), dtype=np.int64)
    for p in lay.S:
        sig_r = conj_array(Rb[:, p], q)                             # sigma(R_{r,p}) for every r
        half_sig_r = scalar_mulmod(sig_r, inv2, q)
        F.lin(all_rows, p, 0, half_sig_r)
        const = (const + matmul_mod(half_sig_r, rot_batch(ones, q).T, q)) % q
    for offset, var in enumerate(lay.U[1:] + [lay.V]):
        F.lin(all_rows, var, 0, conj_array(Rb[:, m + offset], q))
    for r in range(JL_ROWS):
        F.lin(r, lay.Y[r // d], 0, _monomial(-(r % d), d, q))
    const[:, 0] = (const[:, 0] - reduce(z3, q)) % q
    F.const(all_rows, const)
    return F


def jl_vector(pp: ProofParams, s1: np.ndarray) -> np.ndarray:
    """s3 = ((s+1)/2, u_1..u_k, v_{k+1}) as a flat centred integer vector."""
    m = pp.ps.m
    x = (s1[:m] + 1) // 2
    return np.concatenate([x.ravel(), s1[m:].ravel()]).astype(np.int64)


# =====================================================================================
# The proof object
# =====================================================================================


@dataclass
class BaseCaseProof:
    t_y3: np.ndarray     # (n_y3, d)  commitments to the JL masks
    t_g: np.ndarray      # (lam, d)   commitments to the garbage polynomials
    z3: np.ndarray       # (256,)     masked JL projection, centred integers
    h: np.ndarray        # (lam, d)   h_i = g_i + sum_j gamma_ij F_j, constant coefficient 0
    t_quad: np.ndarray   # (d,)       t = b^T s2 + g1
    c: np.ndarray        # (d,)       the challenge (verifier recomputes and compares)
    z1: np.ndarray       # (m1, d)    centred integers
    z2: np.ndarray       # (m2, d)    centred integers


@dataclass
class AttemptStats:
    """Why an attempt was thrown away (reported by bench/timing.py)."""
    jl_bound: int = 0
    rej3: int = 0
    rej1: int = 0
    rej2: int = 0


def _values(pp: ProofParams, lay: Layout, wit: BaseWitness, y3: np.ndarray, g: np.ndarray) -> np.ndarray:
    """x = s1 || (w, y3, g), reduced mod q."""
    q = pp.q
    return np.concatenate([reduce(wit.s1, q), reduce(wit.w, q), reduce(y3, q), reduce(g, q)])


def _lifted(G: Relations, lay: Layout, h: np.ndarray) -> Relations:
    """R_q relations g_i + G_i(x) - h_i = 0 (one per garbage polynomial)."""
    H = Relations(G.count, G.d, G.q)
    H.terms = {key: arr.copy() for key, arr in G.terms.items()}
    one = _scalar(1, G.d, G.q)
    for i, var in enumerate(lay.G):
        H.lin(i, var, 0, one)
    H.const(slice(0, G.count), (G.q - h) % G.q)
    return H


def prove(pp: ProofParams, ck: abdlop.CommitmentKey, tr: OomTranscript, st: BaseStatement,
          wit: BaseWitness, rng=None, stats: AttemptStats | None = None,
          check_witness: bool = True) -> BaseCaseProof | None:
    """One attempt at Pi_k. Returns None if a rejection step fires (caller restarts everything).

    check_witness=False lets tests run a CHEATING prover on a bad witness, to show that the
    verifier catches it; an honest caller leaves it on so bugs surface immediately.
    """
    d, q = pp.d, pp.q
    lay = Layout(pp)
    stats = stats or AttemptStats()

    # (a) JL approximate shortness. Masks y3 and garbage g are committed BEFORE R and gamma.
    y3 = gaussian_array((pp.n_y3, d), pp.std3, rng)
    g = uniform_array((pp.lam, d), q, rng)
    g[:, 0] = 0
    t_y3 = abdlop.commit_rows(ck, ck.rows_y3(pp), wit.s2, reduce(y3, q))
    t_g = abdlop.commit_rows(ck, ck.rows_g(pp), wit.s2, g)
    tr.absorb_array("t_y3", t_y3)
    tr.absorb_array("t_g", t_g)

    R = tr.jl_matrix(pp.n3 * d)
    Rs3 = R @ jl_vector(pp, wit.s1)                          # exact: |R| <= 1, |s3| <= 1
    if float(np.dot(Rs3.astype(float), Rs3.astype(float))) > pp.T3 ** 2:
        stats.jl_bound += 1
        return None
    z3 = y3.ravel() + Rs3
    if rej1(z3, Rs3, pp.std3, pp.M3, rng):
        stats.rej3 += 1
        return None
    tr.absorb_array("z3", reduce(z3, q))

    # (b) Z_q -> R_q lift: hide the combined relations behind garbage g with const coeff 0.
    F = Relations.stack([rk_relations(pp, lay, st), jl_relations(pp, lay, R, z3)])
    gamma = tr.uniform("gamma", (pp.lam, F.count))
    G = F.combine(gamma)
    x = _values(pp, lay, wit, y3, g)
    Gx = G.evaluate(x)
    if check_witness and np.any(Gx[:, 0] != 0):
        bad = np.nonzero(F.evaluate(x)[:, 0])[0]
        raise ValueError(f"witness does not satisfy R_k (relations {bad[:10].tolist()} ...)")
    h = (g + Gx) % q
    tr.absorb_array("h", h)

    # (c) collapse to a single R_q relation with random polynomial weights mu.
    mu = tr.uniform("mu", (pp.lam, d))
    Q = _lifted(G, lay, h).combine_poly(mu)

    # (d) Fig. 7: prove the quadratic relation Q(x) = 0.
    y1 = gaussian_array((pp.m1, d), pp.std1, rng)
    y2 = gaussian_array((pp.m2, d), pp.std2, rng)
    w = (matvec(ck.A1, reduce(y1, q), q) + matvec(ck.A2, reduce(y2, q), q)) % q
    B_vars = ck.B[: pp.n_vars_bdlop]
    y_tilde = np.concatenate([reduce(y1, q), (q - matvec(B_vars, reduce(y2, q), q)) % q])
    g1, g0 = Q.garbage(x, y_tilde)
    b_quad = ck.B[ck.row_quad(pp)]
    t_quad = (polyvec_dot(b_quad, reduce(wit.s2, q), q) + g1[0]) % q
    v = (g0[0] + polyvec_dot(b_quad, reduce(y2, q), q)) % q
    tr.absorb_array("t", t_quad)
    tr.absorb_array("w", w)
    tr.absorb_array("v", v)
    c = tr.challenge_c().c

    cs1 = centered(np.stack([polymul(c, reduce(p, q), q) for p in wit.s1]), q)
    cs2 = centered(np.stack([polymul(c, reduce(p, q), q) for p in wit.s2]), q)
    z1 = y1 + cs1
    z2 = y2 + cs2
    reject = rej_bimodal if pp.rejection == "bimodal" else rej1
    if reject(z1, cs1, pp.std1, pp.M1, rng):
        stats.rej1 += 1
        return None
    if reject(z2, cs2, pp.std2, pp.M2, rng):
        stats.rej2 += 1
        return None
    return BaseCaseProof(t_y3=t_y3, t_g=t_g, z3=z3, h=h, t_quad=t_quad, c=c, z1=z1, z2=z2)


def _check_shapes(pp: ProofParams, pf: BaseCaseProof) -> str | None:
    d = pp.d
    expected = {"t_y3": (pp.n_y3, d), "t_g": (pp.lam, d), "z3": (JL_ROWS,), "h": (pp.lam, d),
                "t_quad": (d,), "c": (d,), "z1": (pp.m1, d), "z2": (pp.m2, d)}
    for name, shape in expected.items():
        arr = getattr(pf, name)
        if not isinstance(arr, np.ndarray) or arr.shape != shape:
            return f"malformed proof: {name} should have shape {shape}"
    for name in ("t_y3", "t_g", "h", "t_quad", "c"):
        arr = getattr(pf, name)
        if arr.min() < 0 or arr.max() >= pp.q:
            return f"malformed proof: {name} not reduced mod q"
    return None


def _norm(arr: np.ndarray) -> float:
    f = arr.astype(np.float64).ravel()
    return math.sqrt(float(np.dot(f, f)))


def verify(pp: ProofParams, ck: abdlop.CommitmentKey, tr: OomTranscript, st: BaseStatement,
           pf: BaseCaseProof) -> VerifyResult:
    """Verifier of Pi_k. Never sees the witness; returns the first failing check."""
    d, q = pp.d, pp.q
    lay = Layout(pp)
    problem = _check_shapes(pp, pf)
    if problem:
        return VerifyResult.reject(problem)

    tr.absorb_array("t_y3", pf.t_y3)
    tr.absorb_array("t_g", pf.t_g)
    R = tr.jl_matrix(pp.n3 * d)
    if _norm(pf.z3) > pp.B3:
        return VerifyResult.reject(f"JL check: ||z3|| = {_norm(pf.z3):.0f} exceeds {pp.B3:.0f}")
    tr.absorb_array("z3", reduce(pf.z3, q))

    F = Relations.stack([rk_relations(pp, lay, st), jl_relations(pp, lay, R, pf.z3)])
    gamma = tr.uniform("gamma", (pp.lam, F.count))
    G = F.combine(gamma)
    if np.any(pf.h[:, 0] != 0):
        return VerifyResult.reject("Z_q lift: some h_i has a nonzero constant coefficient, "
                                   "so a committed value violates R_k")
    tr.absorb_array("h", pf.h)
    mu = tr.uniform("mu", (pp.lam, d))
    Q = _lifted(G, lay, pf.h).combine_poly(mu)

    if _norm(pf.z1) > pp.B1:
        return VerifyResult.reject(f"||z1|| = {_norm(pf.z1):.0f} exceeds {pp.B1:.0f}")
    if _norm(pf.z2) > pp.B2:
        return VerifyResult.reject(f"||z2|| = {_norm(pf.z2):.0f} exceeds {pp.B2:.0f}")

    # Recompute the prover's first message of Fig. 7 from the responses (standard
    # Fiat-Shamir compression): w = A1 z1 + A2 z2 - c tA, then v from the relation.
    c = pf.c
    z1, z2 = reduce(pf.z1, q), reduce(pf.z2, q)
    w = (matvec(ck.A1, z1, q) + matvec(ck.A2, z2, q)
         - np.stack([polymul(c, t, q) for t in st.tA])) % q
    tB = np.concatenate([st.t_w, pf.t_y3, pf.t_g])
    B_vars = ck.B[: pp.n_vars_bdlop]
    c_tB = np.stack([polymul(c, t, q) for t in tB])
    z_tilde = np.concatenate([z1, (c_tB - matvec(B_vars, z2, q)) % q])
    b_quad = ck.B[ck.row_quad(pp)]
    f = (polymul(c, pf.t_quad, q) - polyvec_dot(b_quad, z2, q)) % q
    v = (Q.homogeneous(z_tilde, c)[0] - f) % q

    tr.absorb_array("t", pf.t_quad)
    tr.absorb_array("w", w)
    tr.absorb_array("v", v)
    c_expected = tr.challenge_c().c
    if not np.array_equal(c_expected, c):
        return VerifyResult.reject("Fiat-Shamir check: the challenge c does not match this "
                                   "statement and transcript (proof made for something else, "
                                   "or a commitment/response was altered)")
    return VerifyResult(True)


# =====================================================================================
# OpeningChecker: DEBUGGING ORACLE ONLY (reveals the witness; not zero-knowledge)
# =====================================================================================


class OpeningChecker:
    """Opens the commitments and checks every condition of R_k in the clear.

    Jie's email: "an opening-based checker can be retained as an intermediate debugging tool,
    but it should not replace the actual Pi_k protocol". We use it to (1) cross-check that the
    Relations encoding above says exactly what R_k says, and (2) as the baseline in
    bench/naive_verifier.py. It is independent of the Relations code on purpose: it computes
    each condition directly from the paper's definitions.
    """

    def __init__(self, pp: ProofParams, ck: abdlop.CommitmentKey) -> None:
        self.pp, self.ck = pp, ck

    def check_openings(self, st: BaseStatement, wit: BaseWitness) -> list[str]:
        pp, ck, q = self.pp, self.ck, self.pp.q
        failed = []
        if not np.array_equal(abdlop.commit_ajtai(ck, wit.s1, wit.s2), st.tA):
            failed.append("opening: t_A != A1 s1 + A2 s2")
        if not np.array_equal(abdlop.commit_rows(ck, ck.rows_w(pp), wit.s2, reduce(wit.w, q)), st.t_w):
            failed.append("opening: t_j != b_j^T s2 + w_j")
        return failed

    def check_relation(self, st: BaseStatement, wit: BaseWitness) -> list[str]:
        """Every row of R_k, computed directly. Returns the names of the failing rows."""
        pp, q = self.pp, self.pp.q
        ps, k, base, ell = pp.ps, pp.k, pp.ps.base, pp.ps.ell
        s1 = centered(reduce(wit.s1, q), q)
        s, us, v = s1[: ps.m], [None] + list(s1[ps.m: ps.m + k]), s1[ps.m + k]
        w = [None] + list(reduce(wit.w, q))
        failed = []
        if not np.all(np.abs(s) == 1):
            failed.append("row 1: s is not in {-1,1}^{md}")
        for j in range(1, k + 1):
            u = us[j]
            if np.count_nonzero(u) != 1 or np.abs(u).max() != 1:
                failed.append(f"row 2: u_{j} is not a signed monomial")
            if np.any(u[base:] != 0):
                failed.append(f"row 3: u_{j} uses coefficients beyond the first {base}")
        if np.count_nonzero(v) != 1 or np.abs(v).max() != 1:
            failed.append(f"row 2: v_{k + 1} is not a signed monomial")
        Tv = matmul_mod(reduce(st.T_k, q), reduce(v, q)[:, None], q)[:, 0]
        if not np.array_equal(Tv, w[k]):
            failed.append(f"row 4: T_k v_{k + 1} != w_{k}")
        for i in range(ell):
            if link_first_value(reduce(us[1], q), w[1], st.a[i], reduce(s, q), i, base, q):
                failed.append(f"row 5: Fig. 9 linking check fails for i = {i}")
                break
        for j in range(1, k):
            for i in range(ell):
                if link_next_value(reduce(us[j + 1], q), w[j + 1], st.phis[j][i], w[j], i, base, q):
                    failed.append(f"row 6: Fig. 8 linking check fails at level {j}, i = {i}")
                    break
        return failed

    def check(self, st: BaseStatement, wit: BaseWitness, openings_only: bool = False) -> VerifyResult:
        failed = self.check_openings(st, wit)
        if not openings_only:
            failed += self.check_relation(st, wit)
        if failed:
            return VerifyResult(False, failed[0], failed)
        return VerifyResult(True, "openings and every R_k condition hold (witness revealed!)")
