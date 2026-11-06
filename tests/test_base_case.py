"""Base case Pi_k (the LNP22-style zero-knowledge proof of R_k).

Three kinds of test, kept apart on purpose (Jie's email, point 3):

  * functional: an honest prover's proof is accepted;
  * negative:   a prover whose witness breaks one row of R_k is caught, and the debug
                OpeningChecker names that row. These show the checks are wired up; they are
                not a soundness proof (that is LNP22's theorem plus MSIS);
  * sanity checks on zero-knowledge: e.g. after rejection sampling z1 is not correlated
    with c s1. Again evidence, not a proof (ZK rests on MLWE / Ext-MLWE).
"""

from __future__ import annotations

import random
from dataclasses import replace

import numpy as np
import pytest

from bloom import abdlop, base_case, oom
from bloom.base_case import BaseWitness, Layout, OpeningChecker, rk_relations
from bloom.oom_params import JL_ROWS
from bloom.ring import reduce
from bloom.sampling import sign_array


# --- helpers ---


def reduced(inst, sign=1, seed=0):
    """Run the k folds as the honest prover. Returns (transcript after the folds, st, wit)."""
    tr = inst.transcript()
    _, st, bw = oom.reduce_prove(inst.pp, inst.ck, inst.stmt, inst.wit, tr, sign, random.Random(seed))
    return tr, st, bw


def recommit(inst, st, bw):
    """A cheater can commit to anything: recompute t_A and t_1..t_k for a modified witness."""
    pp, ck = inst.pp, inst.ck
    return replace(st, tA=abdlop.commit_ajtai(ck, bw.s1, bw.s2),
                   t_w=abdlop.commit_rows(ck, ck.rows_w(pp), bw.s2, reduce(bw.w, pp.q)))


def run_base_case(inst, tr, st, bw, check_witness=True, attempts=60, seed=0):
    """Prove then verify Pi_k from the same transcript state. Returns (proof, VerifyResult)."""
    rng = random.Random(seed)
    for _ in range(attempts):
        ptr, vtr = tr.fork(), tr.fork()
        pf = base_case.prove(inst.pp, inst.ck, ptr, st, bw, rng, check_witness=check_witness)
        if pf is not None:
            return pf, base_case.verify(inst.pp, inst.ck, vtr, st, pf)
    return None, None


def row_groups(pp):
    """Row index ranges of rk_relations, by row of the R_k table."""
    ps, k = pp.ps, pp.k
    sizes = [("row 1", 1), ("row 2", k + 1), ("row 3", (ps.d - ps.base) * k), ("row 4", ps.d),
             ("row 5", ps.ell), ("row 6", ps.ell * (k - 1))]
    out, start = {}, 0
    for name, size in sizes:
        out[name] = range(start, start + size)
        start += size
    return out


def failing_rows_by_relations(inst, st, bw):
    """Which R_k rows the Relations encoding (what the ZK proof proves) says are violated."""
    pp = inst.pp
    lay = Layout(pp)
    F = rk_relations(pp, lay, st)
    filler = np.zeros((pp.n_y3 + pp.lam, pp.d), dtype=np.int64)
    x = np.concatenate([reduce(bw.s1, pp.q), reduce(bw.w, pp.q), filler])
    bad = set(np.nonzero(F.evaluate(x)[:, 0])[0].tolist())
    return {name for name, rows in row_groups(pp).items() if bad & set(rows)}


def failing_rows_by_checker(inst, st, bw):
    failed = OpeningChecker(inst.pp, inst.ck).check(st, bw).failed
    return {f[:5] for f in failed if f.startswith("row")}


# --- mutations: each breaks one condition of R_k (others may break as a consequence) ---


def _s1(bw):
    return bw.s1.copy()


def bad_s_not_sign(pp, bw):
    s1 = _s1(bw)
    s1[0, 3] = 3                                   # s must be in {-1, 1}
    return replace(bw, s1=s1)


def bad_u_norm(pp, bw):
    s1 = _s1(bw)
    s1[pp.ps.m] *= 2                               # u_1 = 2 X^a: norm 2, not a signed monomial
    return replace(bw, s1=s1)


def bad_u_support(pp, bw):
    s1 = _s1(bw)
    s1[pp.ps.m] = 0
    s1[pp.ps.m, pp.ps.base + 2] = 1                # u_1 = X^{base+2}: outside the first `base` coefficients
    return replace(bw, s1=s1)


def bad_v_two_terms(pp, bw):
    s1 = _s1(bw)
    v = s1[pp.ps.m + pp.k]
    j = int(np.nonzero(v)[0][0])
    v[(j + 1) % pp.d] = 1                          # v_{k+1} = X^a + X^b
    return replace(bw, s1=s1)


def bad_w_last(pp, bw):
    w = bw.w.copy()
    w[pp.k - 1, 5] = (w[pp.k - 1, 5] + 1) % pp.q   # w_k != T_k v_{k+1}
    return replace(bw, w=w)


def bad_w_first(pp, bw):
    w = bw.w.copy()
    w[0, 9] = (w[0, 9] + 1) % pp.q
    return replace(bw, w=w)


def bad_other_key(pp, bw):
    """A forger: a perfectly formed key s' that isn't the key at the claimed ring position."""
    s1 = _s1(bw)
    s1[: pp.ps.m] = sign_array((pp.ps.m, pp.d), random.Random(404))
    return replace(bw, s1=s1)


MUTATIONS = [
    (bad_s_not_sign, "row 1"),
    (bad_u_norm, "row 2"),
    (bad_u_support, "row 3"),
    (bad_v_two_terms, "row 2"),
    (bad_w_last, "row 4"),
    (bad_w_first, "row 6"),
    (bad_other_key, "row 5"),
]


# --- functional ---


@pytest.mark.parametrize("fixture", ["toy_rej1", "toy_bimodal"])
def test_honest_base_case_is_accepted(fixture, request):
    inst = request.getfixturevalue(fixture)
    tr, st, bw = reduced(inst)
    assert OpeningChecker(inst.pp, inst.ck).check(st, bw).ok
    pf, res = run_base_case(inst, tr, st, bw)
    assert pf is not None and res.ok, res


def test_relation_count(toy_rej1):
    pp = toy_rej1.pp
    _, st, _ = reduced(toy_rej1)
    assert rk_relations(pp, Layout(pp), st).count == pp.num_zq_relations - JL_ROWS


# --- negative: one broken row at a time ---


@pytest.mark.parametrize("mutate, row", MUTATIONS, ids=[m.__name__ for m, _ in MUTATIONS])
def test_relations_encoding_agrees_with_opening_checker(toy_rej1, mutate, row):
    """The constraint system the ZK proof proves says exactly what the paper's R_k says.

    OpeningChecker computes each R_k condition directly from the definitions; rk_relations
    is the 'constant coefficient = 0' encoding. They must flag the same rows.
    """
    _, st, bw = reduced(toy_rej1)
    bad = mutate(toy_rej1.pp, bw)
    st = recommit(toy_rej1, st, bad)
    by_checker = failing_rows_by_checker(toy_rej1, st, bad)
    assert row in by_checker
    assert failing_rows_by_relations(toy_rej1, st, bad) == by_checker


@pytest.mark.parametrize("mutate, row", MUTATIONS, ids=[m.__name__ for m, _ in MUTATIONS])
def test_cheating_prover_is_rejected(toy_rej1, mutate, row):
    """A prover that runs the protocol on a bad witness gets caught by the Z_q lift check."""
    tr, st, bw = reduced(toy_rej1)
    bad = mutate(toy_rej1.pp, bw)
    st = recommit(toy_rej1, st, bad)
    with pytest.raises(ValueError, match="does not satisfy R_k"):
        base_case.prove(toy_rej1.pp, toy_rej1.ck, tr.fork(), st, bad, random.Random(0))
    pf, res = run_base_case(toy_rej1, tr, st, bad, check_witness=False)
    assert pf is not None, "cheating prover never got past rejection sampling"
    assert not res.ok
    assert "Z_q lift" in res.reason


def test_lying_about_h_is_caught_by_the_challenge(toy_rej1):
    """Zeroing the constant coefficients of h passes the lift check but breaks the final relation."""
    tr, st, bw = reduced(toy_rej1)
    bad = bad_other_key(toy_rej1.pp, bw)
    st = recommit(toy_rej1, st, bad)
    pf, res = run_base_case(toy_rej1, tr, st, bad, check_witness=False)
    assert not res.ok
    h = pf.h.copy()
    h[:, 0] = 0
    res2 = base_case.verify(toy_rej1.pp, toy_rej1.ck, tr.fork(), st, replace(pf, h=h))
    assert not res2.ok and "Fiat-Shamir" in res2.reason


@pytest.mark.parametrize("field, change, reason", [
    ("z1", lambda a: _bump(a), "Fiat-Shamir"),
    ("z2", lambda a: _bump(a), "Fiat-Shamir"),
    ("z3", lambda a: _bump(a), "Fiat-Shamir"),
    ("t_quad", lambda a: _bump(a), "Fiat-Shamir"),
    ("t_y3", lambda a: _bump(a), "Fiat-Shamir"),
    ("t_g", lambda a: _bump(a), "Fiat-Shamir"),
    ("c", lambda a: np.roll(a, 1), "Fiat-Shamir"),
    ("h", lambda a: _bump(a, 0), "Z_q lift"),
    ("z1", lambda a: a * 1000, "||z1||"),
    ("z3", lambda a: a * 1000, "JL check"),
    ("z2", lambda a: a[:-1], "malformed"),
])
def test_tampered_proof_is_rejected(toy_rej1, field, change, reason):
    tr, st, bw = reduced(toy_rej1)
    pf, res = run_base_case(toy_rej1, tr, st, bw)
    assert res.ok
    bad = replace(pf, **{field: change(getattr(pf, field))})
    res = base_case.verify(toy_rej1.pp, toy_rej1.ck, tr.fork(), st, bad)
    assert not res.ok and reason in res.reason, res.reason


def _bump(a, where=1):
    a = a.copy()
    flat = a.reshape(-1)
    flat[where] = flat[where] + 1
    if a.dtype != object and flat[where] >= 2**31 and a.min() >= 0:
        flat[where] %= 65437 * 65629
    return a


# --- sign invariance (Lemma C.1): why the bimodal trick on s1 is free for even k ---


def test_flipped_witness_satisfies_rk_for_even_k(toy_rej1):
    _, st, bw = reduced(toy_rej1, sign=-1)
    assert OpeningChecker(toy_rej1.pp, toy_rej1.ck).check(st, bw).ok


# (For odd k the flipped witness doesn't satisfy R_k; params.py refuses odd k, so that case
#  is shown with the bare recursion functions in test_recursion.py.)


# --- zero-knowledge sanity checks (evidence only) ---


@pytest.mark.parametrize("fixture", ["toy_rej1", "toy_bimodal"])
def test_responses_look_like_the_simulated_gaussians(fixture, request):
    """A simulator outputs z1, z2, z3 as fresh Gaussians of widths std1, std2, std3.

    After rejection sampling the real responses should have the same scale, whatever the
    witness. (The direction test for the bimodal step is in test_sampling.py, where the signal
    is strong enough to measure; here ||c s1|| is far below std1 * sqrt(n), so only the scale
    is checked.)
    """
    inst = request.getfixturevalue(fixture)
    pp = inst.pp
    tr, st, bw = reduced(inst, seed=3)
    pf, res = run_base_case(inst, tr, st, bw, seed=3)
    assert res.ok
    for z, std in ((pf.z1, pp.std1), (pf.z2, pp.std2), (pf.z3, pp.std3)):
        ratio = float(np.linalg.norm(z.astype(float))) / (std * np.sqrt(z.size))
        assert 0.85 < ratio < 1.15, ratio


def test_proofs_are_randomised(toy_rej1):
    tr, st, bw = reduced(toy_rej1)
    pf1, _ = run_base_case(toy_rej1, tr, st, bw, seed=1)
    pf2, _ = run_base_case(toy_rej1, tr, st, bw, seed=2)
    assert not np.array_equal(pf1.z1, pf2.z1)


def test_opening_checker_is_debug_only(toy_rej1):
    """It needs the witness: a verifier with only the proof cannot call it."""
    import inspect
    params = inspect.signature(OpeningChecker.check).parameters
    assert "wit" in params
    assert "OpeningChecker" not in inspect.getsource(base_case.verify)
    assert isinstance(reduced(toy_rej1)[2], BaseWitness)
