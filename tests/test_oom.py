"""The full one-out-of-many proof (recursion + base case + restarts)."""

import random
from dataclasses import replace

import numpy as np
import pytest

from bloom import oom, oom_params

from conftest import make_instance


@pytest.fixture(scope="module")
def proved(toy_rej1):
    trace = oom.ProveTrace()
    pf = oom.prove(toy_rej1.pp, toy_rej1.ck, toy_rej1.stmt, toy_rej1.wit, toy_rej1.transcript(),
                   random.Random(1), trace)
    return pf, trace


def verify(inst, pf, msg=None, stmt=None, T=None):
    tr = inst.transcript() if msg is None and T is None else inst.transcript(msg or b"test message", T)
    return oom.verify(inst.pp, inst.ck, stmt or inst.stmt, pf, tr)


def test_statement_holds(toy_rej1):
    assert oom.statement_holds(toy_rej1.pp, toy_rej1.stmt, toy_rej1.wit)
    other = oom.Witness(s=toy_rej1.wit.s, index=(toy_rej1.index + 1) % toy_rej1.ps.ring_size)
    assert not oom.statement_holds(toy_rej1.pp, toy_rej1.stmt, other)


def test_honest_proof_accepted(toy_rej1, proved):
    pf, trace = proved
    assert verify(toy_rej1, pf).ok
    assert trace.attempts >= 1


def test_bimodal_proof_accepted(toy_bimodal):
    pf = oom.prove(toy_bimodal.pp, toy_bimodal.ck, toy_bimodal.stmt, toy_bimodal.wit,
                   toy_bimodal.transcript(), random.Random(2))
    assert verify(toy_bimodal, pf).ok


@pytest.mark.parametrize("index", [0, 63, 64, 4095])
def test_edge_indices(index):
    inst = make_instance(oom_params.toy(2), index=index, seed=30 + index)
    pf = oom.prove(inst.pp, inst.ck, inst.stmt, inst.wit, inst.transcript(), random.Random(index))
    assert verify(inst, pf).ok


def test_trace_follows_figs_9_and_8(toy_rej1, proved):
    _, trace = proved
    ps = toy_rej1.ps
    assert [lv.figure for lv in trace.levels] == ["Fig. 9", "Fig. 8"]
    assert trace.levels[0].T_shape == (64, 4096)
    assert trace.levels[0].T_next_shape == (ps.ell * ps.base, 4096 // ps.base)
    assert trace.levels[1].T_next_shape == (ps.ell * ps.base, 64)
    for lv in trace.levels:
        assert lv.w_blocks.shape == (ps.ell, ps.base)
    assert str(toy_rej1.index) in trace.decomposition


def test_wrong_message_rejected(toy_rej1, proved):
    res = verify(toy_rej1, proved[0], msg=b"another message")
    assert not res.ok and "Fiat-Shamir" in res.reason


def test_honest_prover_refuses_false_statement(toy_rej1):
    bad = oom.Witness(s=toy_rej1.wit.s, index=(toy_rej1.index + 1) % toy_rej1.ps.ring_size)
    with pytest.raises(ValueError, match="does not satisfy R_k"):
        oom.prove(toy_rej1.pp, toy_rej1.ck, toy_rej1.stmt, bad, toy_rej1.transcript(), random.Random(3))


def test_bad_witness_shapes(toy_rej1):
    with pytest.raises(ValueError):
        oom.prove(toy_rej1.pp, toy_rej1.ck, toy_rej1.stmt,
                  oom.Witness(s=np.zeros_like(toy_rej1.wit.s), index=0), toy_rej1.transcript())
    with pytest.raises(ValueError):
        oom.prove(toy_rej1.pp, toy_rej1.ck, toy_rej1.stmt,
                  oom.Witness(s=toy_rej1.wit.s, index=-1), toy_rej1.transcript())


@pytest.mark.parametrize("field", ["tA", "t_w"])
def test_tampered_commitments_rejected(toy_rej1, proved, field):
    pf = proved[0]
    arr = getattr(pf, field).copy()
    arr.reshape(-1)[0] = (arr.reshape(-1)[0] + 1) % toy_rej1.pp.q
    res = verify(toy_rej1, replace(pf, **{field: arr}))
    assert not res.ok


def test_malformed_proof_rejected_not_crashing(toy_rej1, proved):
    pf = proved[0]
    assert not verify(toy_rej1, replace(pf, tA=pf.tA[:-1])).ok
    assert not verify(toy_rej1, replace(pf, t_w=pf.t_w[:1])).ok
    assert not verify(toy_rej1, replace(pf, tA=pf.tA - 1)).ok      # negative entries
    assert not verify(toy_rej1, "not a proof").ok


def test_modified_statement_rejected(toy_rej1, proved):
    """Change one coefficient of one ring member (not the signer) and verify the same proof."""
    other = (toy_rej1.index + 7) % toy_rej1.ps.ring_size
    T_p = toy_rej1.T_p.copy()
    T_p[0, other] = (T_p[0, other] + 1) % toy_rej1.ps.p
    stmt = oom.Statement(T=(T_p * toy_rej1.ps.q2) % toy_rej1.pp.q, A_rot=toy_rej1.stmt.A_rot)
    assert not verify(toy_rej1, proved[0], stmt=stmt, T=T_p).ok
    # Even if the transcript were (wrongly) left unchanged, the folds use the new T:
    assert not verify(toy_rej1, proved[0], stmt=stmt).ok
