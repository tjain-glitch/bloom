"""One fold of the recursion (Figs. 8 and 9) as plain linear algebra, no commitments."""

import random

import numpy as np
import pytest

from bloom.fold import apply_selector, first_round_a, fold, fold_identity_holds
from bloom.ring import conj_array, matmul_mod, polyvec_dot, reduce
from bloom.sampling import sign_array, uniform_array
from bloom.tensor import kron_all, selectors

Q = 65437 * 65629
D, BASE, ELL = 64, 8, 8


def test_fold_shape():
    rng = random.Random(1)
    T = uniform_array((64, 4096), Q, rng)
    phis = uniform_array((ELL, 64), Q, rng)
    assert fold(T, phis, BASE, Q).shape == (ELL * BASE, 4096 // BASE)


@pytest.mark.parametrize("seed", range(5))
def test_fold_identity(seed):
    """<T (v (x) rest), phi_i> = <v, w_i> with w_i = T_{j+1} rest restricted to block i."""
    rng = random.Random(seed)
    T = uniform_array((64, 4096), Q, rng)
    phis = uniform_array((ELL, 64), Q, rng)
    vs = selectors(rng.randrange(4096), D, BASE, 2, rng.choice([1, -1]))
    assert fold_identity_holds(T, vs[0], kron_all(vs[1:]), phis, BASE, Q)


def test_fold_identity_fails_for_wrong_first_selector():
    rng = random.Random(7)
    T = uniform_array((64, 4096), Q, rng)
    phis = uniform_array((ELL, 64), Q, rng)
    vs = selectors(100, D, BASE, 2)
    wrong = np.roll(vs[0], 1)
    # w is computed with the honest rest, but the identity is checked for another v_1.
    full = np.kron(vs[0], kron_all(vs[1:]))
    lhs = matmul_mod(phis, matmul_mod(T, full[:, None] % Q, Q), Q)[:, 0]
    w = apply_selector(fold(T, phis, BASE, Q), kron_all(vs[1:]), Q).reshape(ELL, BASE)
    assert not np.array_equal(lhs, matmul_mod(w, wrong[:, None], Q)[:, 0])


def test_first_round_a_gives_inner_product_via_lemma_2_2():
    """<s, A^T phi_i> equals const( sigma(a_i)^T s ) (Lemma 2.2), which row 5 of R_k uses."""
    rng = random.Random(3)
    m = 12
    A_rot = uniform_array((64, m * D), Q, rng)
    phis = uniform_array((ELL, 64), Q, rng)
    s = sign_array((m, D), rng)
    a = first_round_a(A_rot, phis, m, D, Q)
    direct = matmul_mod(matmul_mod(phis, A_rot, Q), reduce(s.reshape(-1), Q)[:, None], Q)[:, 0]
    for i in range(ELL):
        assert polyvec_dot(conj_array(a[i], Q), reduce(s, Q), Q)[0] == direct[i]


def test_fold_rejects_bad_shapes():
    T = np.zeros((64, 100), dtype=np.int64)
    with pytest.raises(ValueError):
        fold(T, np.zeros((ELL, 64), dtype=np.int64), BASE, Q)       # 100 not divisible by 8
    with pytest.raises(ValueError):
        fold(np.zeros((64, 64), dtype=np.int64), np.zeros((ELL, 32), dtype=np.int64), BASE, Q)
