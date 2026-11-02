"""Recursion levels and the Lemma 4.1 linking checks (rows 5 and 6 of R_k)."""

import random

import numpy as np
import pytest

from bloom.fold import first_round_a
from bloom.recursion import (level_step, link_first_value, link_next_value, pack_w,
                             shifted_selector, unpack_w)
from bloom.ring import Ring, reduce
from bloom.sampling import sign_array, uniform_array
from bloom.tensor import kron_all, selector_poly, selectors

Q = 65437 * 65629
D, BASE, ELL, K = 64, 8, 8, 2
RING = Ring(D, Q)


def test_pack_unpack_roundtrip():
    w = uniform_array((ELL * BASE,), Q, random.Random(1))
    packed = pack_w(w, D, Q)
    assert packed.shape == (D,)
    assert np.array_equal(unpack_w(packed, ELL, BASE).ravel(), w)
    with pytest.raises(ValueError):
        pack_w(np.zeros(D + 1, dtype=np.int64), D, Q)


def test_shifted_selector_moves_coefficients_without_wraparound():
    u = np.zeros(D, dtype=np.int64)
    u[3] = 1
    for i in range(ELL):
        out = shifted_selector(u, i, BASE, Q)
        assert out[i * BASE + 3] == 1 and np.count_nonzero(out) == 1


def _honest_levels(seed, sign=1):
    """Run both levels for k = 2 with real selectors; return everything rows 5/6 need."""
    rng = random.Random(seed)
    m = 3
    s = sign_array((m, D), rng)
    A_rot = uniform_array((64, m * D), Q, rng)
    index = rng.randrange(4096)
    vs = selectors(index, D, BASE, K, sign)
    T = uniform_array((64, 4096), Q, rng)
    # Make the statement true: column `index` of T equals A s. With signed selectors the
    # prover uses (b s, b v_1, ..., b v_3); since k + 1 = 3 is odd, T (b^3 e_i) = A (b s) still holds.
    col = (A_rot.astype(object) @ reduce(s.reshape(-1), Q).astype(object)) % Q
    T[:, index] = np.asarray(col, dtype=np.int64)
    phis0 = uniform_array((ELL, 64), Q, rng)
    a = first_round_a(A_rot, phis0, m, D, Q)
    T1, w1 = level_step(T, phis0, vs[1:], BASE, D, Q)
    phis1 = uniform_array((ELL, T1.shape[0]), Q, rng)
    T2, w2 = level_step(T1, phis1, vs[2:], BASE, D, Q)
    us = [selector_poly(v, RING).c for v in vs]
    return dict(s=reduce(sign * s, Q), a=a, w1=w1, w2=w2, T2=T2, phis1=phis1, us=us, vs=vs)


@pytest.mark.parametrize("sign", [1, -1])
def test_linking_checks_hold_for_honest_prover(sign):
    """For even k the signed witness (b s, b u_1, ..., b v_{k+1}) also satisfies them (Lemma C.1)."""
    h = _honest_levels(5, sign)
    for i in range(ELL):
        assert link_first_value(h["us"][0], h["w1"], h["a"][i], h["s"], i, BASE, Q) == 0
        assert link_next_value(h["us"][1], h["w2"], h["phis1"][i], h["w1"], i, BASE, Q) == 0
    # Row 4: T_k v_{k+1} = w_k
    assert np.array_equal((h["T2"].astype(object) @ reduce(h["vs"][2], Q)) % Q, h["w2"] % Q)


def test_sign_flip_breaks_linking_for_odd_k():
    """Lemma C.1 needs k even: with k = 1, flipping every sign breaks the Fig. 9 check.

    T e_i = A s, prover uses (b s, b v_1, b v_2): w_1 = T_1 (b v_2) = b w, so the left side
    const(X^{i base} (b u_1) sigma(b w)) keeps its sign while <a_i, b s> flips.
    """
    rng = random.Random(12)
    m = 3
    s = sign_array((m, D), rng)
    A_rot = uniform_array((64, m * D), Q, rng)
    index = rng.randrange(512)
    T = uniform_array((64, 512), Q, rng)
    T[:, index] = np.asarray((A_rot.astype(object) @ reduce(s.reshape(-1), Q).astype(object)) % Q, dtype=np.int64)
    phis = uniform_array((ELL, 64), Q, rng)
    a = first_round_a(A_rot, phis, m, D, Q)
    for sign, should_hold in ((1, True), (-1, False)):
        vs = selectors(index, D, BASE, 1, sign)
        _, w1 = level_step(T, phis, vs[1:], BASE, D, Q)
        u1 = selector_poly(vs[0], RING).c
        values = [link_first_value(u1, w1, a[i], reduce(sign * s, Q), i, BASE, Q) for i in range(ELL)]
        assert (not any(values)) == should_hold


def test_linking_check_fails_for_wrong_selector():
    h = _honest_levels(6)
    wrong = np.roll(h["us"][0], 1)          # still a monomial with support < base (unless it wrapped)
    values = [link_first_value(wrong, h["w1"], h["a"][i], h["s"], i, BASE, Q) for i in range(ELL)]
    assert any(values)


def test_linking_check_fails_for_wrong_secret():
    h = _honest_levels(7)
    other = reduce(sign_array(h["s"].shape, random.Random(99)), Q)
    values = [link_first_value(h["us"][0], h["w1"], h["a"][i], other, i, BASE, Q) for i in range(ELL)]
    assert any(values)


def test_verifier_level_step_has_no_witness():
    rng = random.Random(8)
    T = uniform_array((64, 512), Q, rng)
    T1, w = level_step(T, uniform_array((ELL, 64), Q, rng), None, BASE, D, Q)
    assert w is None and T1.shape == (ELL * BASE, 64)


def test_w_is_folded_matrix_times_rest():
    rng = random.Random(9)
    T = uniform_array((64, 4096), Q, rng)
    phis = uniform_array((ELL, 64), Q, rng)
    vs = selectors(777, D, BASE, K)
    T1, w1 = level_step(T, phis, vs[1:], BASE, D, Q)
    expected = (T1.astype(object) @ kron_all(vs[1:])) % Q
    assert np.array_equal(w1, np.asarray(expected, dtype=np.int64))
