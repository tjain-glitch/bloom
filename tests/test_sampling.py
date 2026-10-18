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
