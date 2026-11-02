"""ABDLOP commitment (Section 2.6): key expansion, commit/open, row layout."""

import random

import numpy as np
import pytest

from bloom import abdlop, oom_params
from bloom.ring import reduce
from bloom.sampling import sign_array, ternary_array, uniform_array

PP = oom_params.toy(2)


@pytest.fixture(scope="module")
def ck():
    return abdlop.keygen(PP, b"commitment-seed-0123456789")


def test_key_shapes(ck):
    d = PP.d
    assert ck.A1.shape == (PP.n_com, PP.m1, d)
    assert ck.A2.shape == (PP.n_com, PP.m2, d)
    assert ck.B.shape == (PP.n_bdlop_rows, PP.m2, d)
    for M in (ck.A1, ck.A2, ck.B):
        assert M.min() >= 0 and M.max() < PP.q


def test_keygen_is_deterministic_and_seed_dependent(ck):
    again = abdlop.keygen(PP, b"commitment-seed-0123456789")
    other = abdlop.keygen(PP, b"commitment-seed-9876543210")
    assert np.array_equal(ck.A1, again.A1) and np.array_equal(ck.B, again.B)
    assert not np.array_equal(ck.A1, other.A1)
    with pytest.raises(ValueError):
        abdlop.keygen(PP, b"short")


def test_row_slices_partition_bdlop_rows(ck):
    rows = list(range(PP.n_bdlop_rows))
    used = (rows[ck.rows_w(PP)] + rows[ck.rows_y3(PP)] + rows[ck.rows_g(PP)] + [ck.row_quad(PP)])
    assert used == rows


def test_commit_and_open(ck):
    rng = random.Random(1)
    s1 = sign_array((PP.m1, PP.d), rng)
    s2 = ternary_array((PP.m2, PP.d), rng)
    msgs = uniform_array((PP.k, PP.d), PP.q, rng)
    tA = abdlop.commit_ajtai(ck, s1, s2)
    tB = abdlop.commit_rows(ck, ck.rows_w(PP), s2, msgs)
    assert abdlop.opens(ck, tA, tB, ck.rows_w(PP), s1, s2, msgs)
    # commit_row agrees with commit_rows
    for j in range(PP.k):
        assert np.array_equal(abdlop.commit_row(ck, j, s2, msgs[j]), tB[j])
    # any change to the opening is detected
    s1_bad = s1.copy()
    s1_bad[0, 0] *= -1
    assert not abdlop.opens(ck, tA, tB, ck.rows_w(PP), s1_bad, s2, msgs)
    msgs_bad = msgs.copy()
    msgs_bad[1, 5] = (msgs_bad[1, 5] + 1) % PP.q
    assert not abdlop.opens(ck, tA, tB, ck.rows_w(PP), s1, s2, msgs_bad)


def test_commitment_is_linear(ck):
    """t(s1 + s1', s2 + s2') = t(s1, s2) + t(s1', s2'): what lets the verifier compute A1 z1 + A2 z2 - c tA."""
    rng = random.Random(2)
    a1, b1 = sign_array((PP.m1, PP.d), rng), sign_array((PP.m1, PP.d), rng)
    a2, b2 = ternary_array((PP.m2, PP.d), rng), ternary_array((PP.m2, PP.d), rng)
    lhs = abdlop.commit_ajtai(ck, a1 + b1, a2 + b2)
    rhs = (abdlop.commit_ajtai(ck, a1, a2) + abdlop.commit_ajtai(ck, b1, b2)) % PP.q
    assert np.array_equal(lhs, rhs)


def test_commit_rows_shape_check(ck):
    s2 = ternary_array((PP.m2, PP.d), random.Random(3))
    with pytest.raises(ValueError):
        abdlop.commit_rows(ck, ck.rows_w(PP), s2, np.zeros((PP.k + 1, PP.d), dtype=np.int64))


def test_hiding_sanity(ck):
    """Not a proof of hiding (that is MLWE): just that the same message commits differently each time."""
    rng = random.Random(4)
    msg = np.zeros((PP.k, PP.d), dtype=np.int64)
    t1 = abdlop.commit_rows(ck, ck.rows_w(PP), ternary_array((PP.m2, PP.d), rng), msg)
    t2 = abdlop.commit_rows(ck, ck.rows_w(PP), ternary_array((PP.m2, PP.d), rng), msg)
    assert not np.array_equal(t1, t2)
    assert reduce(t1, PP.q).std() > PP.q / 10      # spread over Z_q, not near the message
