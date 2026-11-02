"""Protocol transcript: what is absorbed and the challenges derived from it."""

import numpy as np
import pytest

from bloom import oom_params
from bloom.fiat_shamir import OomTranscript

PP = oom_params.toy(2)
A = np.arange(PP.ps.n * PP.ps.m * PP.d).reshape(PP.ps.n, PP.ps.m, PP.d) % PP.ps.p
T = (np.arange(64 * 16).reshape(64, 16) * 7) % PP.ps.p


def make(msg=b"m", ck_seed=b"seed" * 4, A_keys=A, ring=T, pp=PP):
    tr = OomTranscript(pp)
    tr.absorb_statement(ck_seed, A_keys, ring, msg)
    return tr


def phis(tr):
    return tr.phis(0, 64)


def test_deterministic():
    assert np.array_equal(phis(make()), phis(make()))
    assert np.array_equal(make().challenge_c().c, make().challenge_c().c)


@pytest.mark.parametrize("change", ["msg", "seed", "A", "ring", "ring order", "params"])
def test_every_statement_item_changes_the_challenges(change):
    kwargs = {}
    if change == "msg":
        kwargs["msg"] = b"M"
    elif change == "seed":
        kwargs["ck_seed"] = b"seeD" * 4
    elif change == "A":
        A2 = A.copy()
        A2[0, 0, 0] += 1
        kwargs["A_keys"] = A2
    elif change == "ring":
        T2 = T.copy()
        T2[3, 5] += 1
        kwargs["ring"] = T2
    elif change == "ring order":
        kwargs["ring"] = T[:, ::-1].copy()
    elif change == "params":
        kwargs["pp"] = oom_params.toy(2, "bimodal")
    assert not np.array_equal(phis(make()), phis(make(**kwargs)))


def test_absorbing_a_commitment_changes_later_challenges():
    a, b = make(), make()
    a.absorb_array("tA", np.zeros((2, 64), dtype=np.int64))
    b.absorb_array("tA", np.ones((2, 64), dtype=np.int64))
    assert not np.array_equal(phis(a), phis(b))


def test_fork_is_independent():
    base = make()
    f1, f2 = base.fork(), base.fork()
    f1.absorb_array("x", np.zeros(3, dtype=np.int64))
    f2.absorb_array("x", np.zeros(3, dtype=np.int64))
    assert np.array_equal(phis(f1), phis(f2))
    f3 = base.fork()
    f3.absorb_array("x", np.ones(3, dtype=np.int64))
    assert not np.array_equal(phis(f1), phis(f3))


def test_challenge_ranges():
    tr = make()
    ph = tr.phis(0, 64)
    assert ph.shape == (PP.ps.ell, 64) and ph.min() >= 0 and ph.max() < PP.q
    R = tr.jl_matrix(100)
    assert R.shape == (256, 100) and set(np.unique(R)) <= {-1, 0, 1}
    c = tr.challenge_c()
    cc = c.centered()
    assert np.abs(cc).max() <= PP.ps.kappa


def test_jl_matrix_is_bin1():
    R = make().jl_matrix(1000)
    frac = [(R == v).mean() for v in (-1, 0, 1)]
    assert abs(frac[0] - 0.25) < 0.01 and abs(frac[1] - 0.5) < 0.01 and abs(frac[2] - 0.25) < 0.01


def test_rejects_unreduced_arrays_and_non_bytes_message():
    tr = make()
    with pytest.raises(ValueError):
        tr.absorb_array("bad", np.array([-1]))
    with pytest.raises(TypeError):
        make(msg="not bytes")
