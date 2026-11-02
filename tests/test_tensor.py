"""Tensor decomposition of the signer's index (Section 4, eq. 29)."""

import random

import numpy as np
import pytest

from bloom.ring import Ring
from bloom.tensor import (describe, digits_to_index, index_to_digits, kron_all, ring_size,
                          selector_poly, selectors)

D, BASE = 64, 8


@pytest.mark.parametrize("k, n", [(0, 64), (1, 512), (2, 4096), (4, 262144), (6, 2**24)])
def test_ring_sizes_match_appendix_c(k, n):
    assert ring_size(D, BASE, k) == n


@pytest.mark.parametrize("k", [1, 2, 3])
def test_digits_roundtrip(k):
    rng = random.Random(k)
    n = ring_size(D, BASE, k)
    for i in [0, 1, n - 1] + [rng.randrange(n) for _ in range(50)]:
        digits = index_to_digits(i, D, BASE, k)
        assert len(digits) == k + 1
        assert all(0 <= x < BASE for x in digits[:-1]) and 0 <= digits[-1] < D
        assert digits_to_index(digits, D, BASE) == i


def test_first_digit_is_most_significant():
    # index = ((i_1 * base) + i_2) * d + i_3 for k = 2
    assert index_to_digits((3 * BASE + 5) * D + 7, D, BASE, 2) == [3, 5, 7]


@pytest.mark.parametrize("sign", [1, -1])
@pytest.mark.parametrize("k", [1, 2, 3])
def test_kron_of_selectors_is_signed_unit_vector(k, sign):
    """v_1 (x) ... (x) v_{k+1} = sign^{k+1} e_i: for even k+1 the signs cancel (Lemma C.1)."""
    rng = random.Random(10 + k)
    n = ring_size(D, BASE, k)
    for i in [0, n - 1, rng.randrange(n)]:
        vs = selectors(i, D, BASE, k, sign)
        assert [len(v) for v in vs] == [BASE] * k + [D]
        e = np.zeros(n, dtype=np.int64)
        e[i] = sign ** (k + 1)
        assert np.array_equal(kron_all(vs), e)


def test_selector_poly_is_signed_monomial():
    ring = Ring(D, 65437 * 65629)
    v = selectors(4000, D, BASE, 2, -1)[0]
    u = selector_poly(v, ring).centered()
    assert np.count_nonzero(u) == 1 and np.abs(u).sum() == 1
    assert np.all(u[BASE:] == 0)


def test_bad_inputs():
    with pytest.raises(ValueError):
        index_to_digits(4096, D, BASE, 2)
    with pytest.raises(ValueError):
        index_to_digits(-1, D, BASE, 2)
    with pytest.raises(ValueError):
        selectors(0, D, BASE, 2, sign=2)
    with pytest.raises(ValueError):
        digits_to_index([8, 0, 0], D, BASE)


def test_describe_mentions_every_level():
    text = describe(1234, D, BASE, 2)
    assert "v_1" in text and "v_2" in text and "v_3" in text and "1234" in text
