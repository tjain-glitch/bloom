import math
import random

import numpy as np
import pytest

from bloom.params import RINGSIG_K0 as P
from bloom.ring import Ring
from bloom.sampling import (
    XofRandom,
    challenge_norm_ok,
    challenge_poly,
    expand_mat,
    gaussian_int,
    gaussian_vec,
    random_sign,
    sign_vec,
)

R = Ring(P.d, P.q)


def test_xof_is_deterministic():
    a, b = XofRandom(b"x" * 32), XofRandom(b"x" * 32)
    assert [a.randrange(1000) for _ in range(50)] == [b.randrange(1000) for _ in range(50)]
    c = XofRandom(b"y" * 32)
    assert [XofRandom(b"x" * 32).getrandbits(64) for _ in range(1)] != [c.getrandbits(64)]


def test_xof_rejects_short_seed():
    with pytest.raises(ValueError):
        XofRandom(b"short")


def test_xof_randrange_in_range():
    r = XofRandom(b"z" * 32)
    vals = [r.randrange(P.q) for _ in range(2000)]
    assert min(vals) >= 0 and max(vals) < P.q
    # crude uniformity check: top half and bottom half roughly equal
    assert 850 < sum(v < P.q // 2 for v in vals) < 1150


def test_signs():
    v = sign_vec(R, 3, random.Random(1))
    assert set(int(x) for x in v.centered()) == {-1, 1}
    assert {random_sign(random.Random(i)) for i in range(20)} == {-1, 1}


@pytest.mark.slow
@pytest.mark.parametrize("s", [3.0, 50.0])
def test_gaussian_moments(s):
    r = random.Random(7)
    xs = np.array([gaussian_int(s, r) for _ in range(20000)], dtype=float)
    assert abs(xs.mean()) < 0.05 * s
    assert abs(xs.std() - s) < 0.03 * s


def test_gaussian_bad_s():
    with pytest.raises(ValueError):
        gaussian_int(0)
    with pytest.raises(ValueError):
        gaussian_int(-1.0)


def test_gaussian_vec_shape():
    v = gaussian_vec(R, 4, 10.0, random.Random(3))
    assert len(v) == 4
    assert v.norm_inf() <= math.ceil(14 * 10.0)


def test_challenge_in_challenge_space():
    r = random.Random(11)
    for _ in range(20):
        c = challenge_poly(R, P.kappa, P.eta, P.chal_pow, r)
        assert c.conj() == c                  # sigma_{-1}(c) = c
        assert c.norm_inf() <= P.kappa
        assert c.c[P.d // 2] == 0             # forced by sigma_{-1}-invariance
        assert challenge_norm_ok(c.centered(), P.eta, P.chal_pow)


@pytest.mark.slow
def test_challenge_acceptance_rate_matches_paper():
    # Section 2.5: eta is chosen so a random c in S passes the norm check >= 99% of the time.
    r = random.Random(5)
    d, kappa = P.d, P.kappa
    passed = 0
    trials = 300
    for _ in range(trials):
        c = np.zeros(d, dtype=object)
        for i in range(d // 2):
            c[i] = r.randrange(-kappa, kappa + 1)
        for i in range(1, d // 2):
            c[d - i] = -c[i]
        passed += challenge_norm_ok(c, P.eta, P.chal_pow)
    assert passed / trials >= 0.97


def test_challenge_operator_norm_bound():
    # The point of the eta check: ||c r|| <= eta ||r||. Check it on short vectors, where
    # nothing wraps mod q so the centred norms are the true integer norms.
    r = random.Random(13)
    for _ in range(20):
        c = challenge_poly(R, P.kappa, P.eta, P.chal_pow, r)
        x = sign_vec(R, 1, r)[0]
        assert (c * x).norm2_sq() <= P.eta**2 * x.norm2_sq()


def test_expand_mat_deterministic_and_label_separated():
    small = Ring(8, 97)
    a = expand_mat(small, b"seed" * 8, "A1", 2, 3)
    assert a == expand_mat(small, b"seed" * 8, "A1", 2, 3)
    assert a != expand_mat(small, b"seed" * 8, "A2", 2, 3)
    assert a != expand_mat(small, b"other" * 8, "A1", 2, 3)


# --- vectorised samplers and rejection sampling (used by the proof layer) ------------------

from bloom.sampling import (  # noqa: E402
    bimodal_M,
    binomial_array,
    expand_array,
    gaussian_array,
    rej1,
    rej1_M,
    rej_bimodal,
    sign_array,
    ternary_array,
    uniform_array,
)


def test_array_samplers_ranges():
    rng = random.Random(1)
    u = uniform_array((1000,), P.q, rng)
    assert u.min() >= 0 and u.max() < P.q
    assert set(np.unique(ternary_array((1000,), rng))) == {-1, 0, 1}
    assert set(np.unique(sign_array((1000,), rng))) == {-1, 1}
    assert uniform_array((3, 4), 7, rng).shape == (3, 4)


def test_uniform_array_is_unbiased():
    counts = np.bincount(uniform_array((60000,), 6, random.Random(2)), minlength=6)
    assert np.all(np.abs(counts / 60000 - 1 / 6) < 0.01)


def test_expand_array_deterministic_and_label_separated():
    a = expand_array(b"s" * 32, "x", (5, 64), P.q)
    assert np.array_equal(a, expand_array(b"s" * 32, "x", (5, 64), P.q))
    assert not np.array_equal(a, expand_array(b"s" * 32, "y", (5, 64), P.q))
    assert not np.array_equal(a, expand_array(b"t" * 32, "x", (5, 64), P.q))


def test_binomial_array_is_deterministic():
    assert np.array_equal(binomial_array((4, 9), b"k" * 32, "jl"), binomial_array((4, 9), b"k" * 32, "jl"))


@pytest.mark.parametrize("s", [3.0, 1000.0, 150000.0])
def test_gaussian_array_moments(s):
    x = gaussian_array((20000,), s, random.Random(int(s)))
    assert abs(x.mean()) < 4 * s / math.sqrt(20000)
    assert abs(x.std() / s - 1) < 0.03


@pytest.mark.slow
@pytest.mark.parametrize("mode", ["rej1", "bimodal"])
def test_rejection_acceptance_rate_is_about_one_over_M(mode):
    """Lemma 2.8: with ||v|| at the bound T and s = gamma T, acceptance is close to 1/M.

    For Rej1 the 1/M is a lower bound (M carries a tail-cut slack), so we only check
    1/M <= rate; for the bimodal test the rate is 1/M exactly in expectation.
    """
    rng = random.Random(5)
    n, trials = 256, 1500
    v = np.zeros(n, dtype=np.int64)
    v[:16] = 25                                  # ||v|| = 100
    T = float(np.linalg.norm(v))
    if mode == "rej1":
        gamma = 12.0
        s, M = gamma * T, rej1_M(gamma)
        accepted = sum(not rej1(gaussian_array((n,), s, rng) + v, v, s, M, rng) for _ in range(trials))
        assert accepted / trials >= 1 / M - 0.04
    else:
        gamma = 1.0
        s, M = gamma * T, bimodal_M(gamma)
        accepted = 0
        for _ in range(trials):
            b = random_sign(rng)
            accepted += not rej_bimodal(gaussian_array((n,), s, rng) + b * v, v, s, M, rng)
        assert abs(accepted / trials - 1 / M) < 0.04


def test_bimodal_output_does_not_depend_on_v():
    """After bimodal rejection z = y + b v is distributed as D_s, independent of v ([DDLL13]).

    We look at t = <z, v> / (s ||v||). For z ~ D_s, t is ~N(0, 1), variance 1. Without the
    rejection step, t = b ||v|| / s + N(0, 1) has variance 1 + ||v||^2 / s^2 = 2 here.
    """
    rng = random.Random(6)
    n = 256
    v = np.zeros(n, dtype=np.int64)
    v[:16] = 25
    s = float(np.linalg.norm(v))                 # gamma = 1
    M = bimodal_M(1.0)
    before, after = [], []
    for _ in range(1500):
        z = gaussian_array((n,), s, rng) + random_sign(rng) * v
        t = float(z @ v) / (s * s)
        before.append(t)
        if not rej_bimodal(z, v, s, M, rng):
            after.append(t)
    assert abs(np.var(before) - 2.0) < 0.3
    assert abs(np.var(after) - 1.0) < 0.2
