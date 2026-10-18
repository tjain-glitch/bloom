import random

import numpy as np
import pytest

from bloom.ring import Poly, PolyMat, PolyVec, Ring, negacyclic_mul
from bloom.sampling import uniform_mat, uniform_poly, uniform_vec

Q = 65437 * 65629
R = Ring(64, Q)
RP = Ring(64, 65437)


@pytest.fixture
def rng():
    return random.Random(2026)


def naive_mul(a, b, d, q):
    """Textbook reference: multiply term by term and reduce with X^d = -1."""
    out = [0] * d
    for i, ai in enumerate(a):
        for j, bj in enumerate(b):
            e = i + j
            if e < d:
                out[e] += ai * bj
            else:
                out[e - d] -= ai * bj
    return [x % q for x in out]


def test_mul_matches_naive(rng):
    for _ in range(20):
        a, b = uniform_poly(R, rng), uniform_poly(R, rng)
        assert list((a * b).c) == naive_mul(list(a.c), list(b.c), R.d, Q)


def test_x_to_the_d_is_minus_one():
    assert R.monomial(R.d) == -R.one()
    assert R.monomial(2 * R.d) == R.one()
    assert R.monomial(-1) * R.monomial(1) == R.one()
    assert R.monomial(R.d - 1) * R.monomial(1) == R(([-1] + [0] * (R.d - 1)))


def test_ring_axioms(rng):
    a, b, c = (uniform_poly(R, rng) for _ in range(3))
    assert a * b == b * a
    assert (a * b) * c == a * (b * c)
    assert a * (b + c) == a * b + a * c
    assert a - a == R.zero()
    assert a + 5 == a + R.monomial(0, 5)
    assert 3 * a == a + a + a


def test_sigma_is_ring_automorphism(rng):
    a, b = uniform_poly(R, rng), uniform_poly(R, rng)
    for i in (-1, 3, 5, 2 * R.d - 1):
        assert (a * b).sigma(i) == a.sigma(i) * b.sigma(i)
        assert (a + b).sigma(i) == a.sigma(i) + b.sigma(i)
    assert R.monomial(1).sigma(5) == R.monomial(5)
    assert a.conj().conj() == a
    with pytest.raises(ValueError):
        a.sigma(2)


def test_lemma_2_2_inner_product(rng):
    # The constant coefficient of sigma_{-1}(u)^T v equals <u, v> (coefficient vectors).
    u, v = uniform_vec(R, 5, rng), uniform_vec(R, 5, rng)
    lhs = u.conj().dot(v).const
    rhs = int(np.dot(u.coeffs(), v.coeffs())) % Q
    assert lhs == rhs


def test_centered_and_norms():
    p = R([1, -1, 5, Q // 2, -(Q // 2)] + [0] * 59)
    assert list(p.centered()[:5]) == [1, -1, 5, Q // 2, -(Q // 2)]
    assert p.norm_inf() == Q // 2
    assert R([3, -4] + [0] * 62).norm2_sq() == 25


def test_rot_matches_multiplication(rng):
    a, b = uniform_poly(R, rng), uniform_poly(R, rng)
    assert list(a.rot().dot(b.c) % Q) == list((a * b).c)

    A = uniform_mat(R, 2, 3, rng)
    s = uniform_vec(R, 3, rng)
    assert list(A.rot().dot(s.coeffs()) % Q) == list((A @ s).coeffs())


def test_bytes_roundtrip_and_validation(rng):
    a = uniform_poly(R, rng)
    data = a.to_bytes()
    assert len(data) == 64 * 4
    assert R.poly_from_bytes(data) == a
    with pytest.raises(ValueError):
        R.poly_from_bytes(data[:-1])
    with pytest.raises(ValueError):
        R.poly_from_bytes((Q).to_bytes(4, "little") * 64)  # coefficient == q


def test_ring_mismatch_raises(rng):
    with pytest.raises(ValueError):
        uniform_poly(R, rng) + uniform_poly(RP, rng)
    with pytest.raises(ValueError):
        Poly(R, [1, 2, 3])
    with pytest.raises(TypeError):
        uniform_poly(R, rng) * 1.5


def test_lift_keeps_small_values():
    small = RP([1, -1, 0, 7] + [0] * 60)
    lifted = small.lift(R)
    assert list(lifted.centered()) == list(small.centered())


def test_polyvec_helpers(rng):
    v = uniform_vec(R, 3, rng)
    assert PolyVec.from_coeffs(R, list(v.coeffs())) == v
    w = PolyVec.concat(v, v)
    assert len(w) == 6 and w[3] == v[0]
    c = uniform_poly(R, rng)
    assert v.scale(c)[1] == v[1] * c
    assert (v - v) == PolyVec.zeros(R, 3)


def test_matvec_shape_check(rng):
    A = uniform_mat(R, 2, 3, rng)
    with pytest.raises(ValueError):
        A @ uniform_vec(R, 2, rng)
    with pytest.raises(ValueError):
        PolyMat(R, [[R.zero()], [R.zero(), R.zero()]])


def test_negacyclic_mul_without_modulus():
    a = np.array([2, 0, 0, 1], dtype=object)    # 2 + X^3
    b = np.array([0, 1, 0, 0], dtype=object)    # X
    assert list(negacyclic_mul(a, b)) == [-1, 2, 0, 0]  # 2X + X^4 = 2X - 1
