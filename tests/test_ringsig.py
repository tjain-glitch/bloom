"""Ring signature (App. C.2): KeyGen / Sign / Verify."""

import random

import numpy as np
import pytest

from bloom import oom_params, ringsig
from bloom.oom import ProveTrace


@pytest.fixture(scope="module")
def setup():
    params = ringsig.setup(oom_params.toy(2), b"ringsig-test-public-seed")
    keys = ringsig.keygen_many(params, params.pp.ps.ring_size, random.Random(1))
    ring = [k.pk for k in keys]
    sig = ringsig.sign(params, keys[1000], ring, b"hello", random.Random(2))
    return params, keys, ring, sig


def test_keygen(setup):
    params, keys, _, _ = setup
    ps = params.pp.ps
    k = keys[0]
    assert k.sk.shape == (ps.m, ps.d) and set(np.unique(k.sk)) <= {-1, 1}
    assert k.pk.shape == (ps.n * ps.d,) and k.pk.min() >= 0 and k.pk.max() < ps.p
    assert np.array_equal(ringsig.public_key(params, k.sk), k.pk)
    single = ringsig.keygen(params, random.Random(3))
    assert np.array_equal(ringsig.public_key(params, single.sk), single.pk)


def test_public_key_is_A_times_s_mod_p(setup):
    """pk = A s in R_p, computed with ring multiplication instead of rot(A)."""
    from bloom.ring import polymul
    params, keys, _, _ = setup
    ps = params.pp.ps
    sk = keys[5].sk
    pk = np.zeros(ps.d, dtype=np.int64)
    for j in range(ps.m):
        pk = (pk + polymul(params.A[0, j], sk[j] % ps.p, ps.p)) % ps.p
    assert np.array_equal(pk, keys[5].pk)


def test_sign_verify(setup):
    params, _, ring, sig = setup
    assert ringsig.verify(params, ring, b"hello", sig).ok


def test_every_member_can_sign(setup):
    params, keys, ring, _ = setup
    for i in (0, 4095):
        sig = ringsig.sign(params, keys[i], ring, b"m", random.Random(i))
        assert ringsig.verify(params, ring, b"m", sig).ok


def test_signature_does_not_reveal_index_size_or_shape(setup):
    """Anonymity is a ZK property (not testable); this just checks no field depends on i."""
    from bloom import serialize
    params, keys, ring, sig = setup
    other = ringsig.sign(params, keys[7], ring, b"hello", random.Random(4))
    for a, b in ((sig.tA, other.tA), (sig.t_w, other.t_w), (sig.base.z1, other.base.z1)):
        assert a.shape == b.shape
    assert len(serialize.encode(sig, params.pp)) // 100 == len(serialize.encode(other, params.pp)) // 100


def test_wrong_message(setup):
    params, _, ring, sig = setup
    res = ringsig.verify(params, ring, b"hellO", sig)
    assert not res.ok and "Fiat-Shamir" in res.reason


def test_signer_must_be_in_ring(setup):
    params, keys, ring, _ = setup
    outsider = ringsig.keygen(params, random.Random(5))
    with pytest.raises(ValueError, match="not in the ring"):
        ringsig.sign(params, outsider, ring, b"x")


def test_secret_key_must_match(setup):
    params, keys, ring, _ = setup
    fake = ringsig.KeyPair(sk=keys[1].sk, pk=keys[2].pk)
    with pytest.raises(ValueError, match="does not match"):
        ringsig.sign(params, fake, ring, b"x")


def test_bad_inputs_are_rejected_not_crashing(setup):
    params, _, ring, sig = setup
    assert not ringsig.verify(params, ring[:-1], b"hello", sig).ok
    assert not ringsig.verify(params, ring, "hello", sig).ok
    bad = list(ring)
    bad[0] = bad[0] + params.pp.ps.p
    assert "invalid input" in ringsig.verify(params, bad, b"hello", sig).reason
    with pytest.raises(TypeError):
        ringsig.sign(params, None, ring, "text")


def test_setup_needs_a_real_seed():
    with pytest.raises(ValueError):
        ringsig.setup(oom_params.toy(2), b"short")


@pytest.mark.slow
@pytest.mark.parametrize("mode", ["rej1", "bimodal"])
def test_paper_scale_ring_of_4096(mode):
    """Fig. 10 parameters (m = 12) with N = 2^12 keys."""
    params = ringsig.setup(oom_params.paper_scale(2, mode), b"paper-scale-public-seed")
    keys = ringsig.keygen_many(params, 4096, random.Random(6))
    ring = [k.pk for k in keys]
    trace = ProveTrace()
    sig = ringsig.sign(params, keys[2024], ring, b"paper", random.Random(7), trace)
    assert ringsig.verify(params, ring, b"paper", sig).ok
    assert not ringsig.verify(params, ring, b"Paper", sig).ok
    assert trace.attempts < 40
