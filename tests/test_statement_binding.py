"""The same signature has to fail after any change to the public statement.

This is the property the demo shows live (part 2), and the bug the team found earlier
(a transcript that skipped the ring accepted modified statements). Each case keeps the
signature fixed and changes one public input.
"""

import random

import pytest

from bloom import oom_params, ringsig


@pytest.fixture(scope="module")
def world():
    params = ringsig.setup(oom_params.toy(2), b"binding-test-public-seed")
    keys = ringsig.keygen_many(params, params.pp.ps.ring_size, random.Random(1))
    ring = [k.pk for k in keys]
    signer = 2500
    sig = ringsig.sign(params, keys[signer], ring, b"vote: yes", random.Random(2))
    assert ringsig.verify(params, ring, b"vote: yes", sig).ok
    return params, keys, ring, sig, signer


def _modify_key(ring, i, p):
    ring = list(ring)
    pk = ring[i].copy()
    pk[0] = (pk[0] + 1) % p
    ring[i] = pk
    return ring


@pytest.mark.parametrize("which", ["first", "signer", "last", "neighbour"])
def test_modified_public_key(world, which):
    params, _, ring, sig, signer = world
    i = {"first": 0, "signer": signer, "last": len(ring) - 1, "neighbour": signer + 1}[which]
    res = ringsig.verify(params, _modify_key(ring, i, params.pp.ps.p), b"vote: yes", sig)
    assert not res.ok and res.reason


def test_replaced_public_key(world):
    params, _, ring, sig, signer = world
    ring = list(ring)
    ring[17] = ringsig.keygen(params, random.Random(9)).pk
    assert not ringsig.verify(params, ring, b"vote: yes", sig).ok


@pytest.mark.parametrize("i, j", [(0, 1), (100, 3000), (2500, 2501)])
def test_reordered_ring(world, i, j):
    params, _, ring, sig, _ = world
    ring = list(ring)
    ring[i], ring[j] = ring[j], ring[i]
    assert not ringsig.verify(params, ring, b"vote: yes", sig).ok


@pytest.mark.parametrize("msg", [b"vote: no", b"vote: yes ", b"", b"VOTE: YES"])
def test_modified_message(world, msg):
    params, _, ring, sig, _ = world
    assert not ringsig.verify(params, ring, msg, sig).ok


def test_different_public_parameters(world):
    params, _, ring, sig, _ = world
    other = ringsig.setup(params.pp, b"binding-test-public-seeD")
    assert not ringsig.verify(other, ring, b"vote: yes", sig).ok


def test_different_rejection_mode_params(world):
    params, _, ring, sig, _ = world
    other = ringsig.PublicParams(pp=oom_params.toy(2, "bimodal"), seed=params.seed)
    assert not ringsig.verify(other, ring, b"vote: yes", sig).ok


def test_unchanged_statement_still_verifies(world):
    """Control: verification is deterministic, so the failures above are due to the change."""
    params, _, ring, sig, _ = world
    assert ringsig.verify(params, list(ring), b"vote: yes", sig).ok
