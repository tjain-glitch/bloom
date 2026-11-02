"""Byte encoding of signatures (used to MEASURE sizes; Fig. 10 numbers are estimates)."""

import random

import numpy as np
import pytest

from bloom import oom_params, ringsig, serialize


@pytest.fixture(scope="module")
def signed():
    params = ringsig.setup(oom_params.toy(2), b"serialize-test-public-seed")
    keys = ringsig.keygen_many(params, params.pp.ps.ring_size, random.Random(1))
    ring = [k.pk for k in keys]
    sig = ringsig.sign(params, keys[9], ring, b"msg", random.Random(2))
    return params, ring, sig


def test_roundtrip(signed):
    params, ring, sig = signed
    blob = serialize.encode(sig, params.pp)
    back = serialize.decode(blob, params.pp)
    assert serialize.encode(back, params.pp) == blob
    for a, b in ((sig.tA, back.tA), (sig.t_w, back.t_w), (sig.base.z1, back.base.z1),
                 (sig.base.z2, back.base.z2), (sig.base.z3, back.base.z3), (sig.base.c, back.base.c),
                 (sig.base.h, back.base.h)):
        assert np.array_equal(a, b)
    assert ringsig.verify(params, ring, b"msg", back).ok


def test_breakdown_adds_up(signed):
    params, _, sig = signed
    br = serialize.size_breakdown(sig, params.pp)
    parts = sum(v for k, v in br.items() if not k.startswith(("total", "z entropy")))
    assert parts + len(serialize.MAGIC) == br["total (encoded)"] == len(serialize.encode(sig, params.pp))


def test_decode_rejects_garbage(signed):
    params, _, sig = signed
    blob = serialize.encode(sig, params.pp)
    with pytest.raises(ValueError, match="magic"):
        serialize.decode(b"XXXX" + blob[4:], params.pp)
    with pytest.raises(ValueError):
        serialize.decode(blob[:-1], params.pp)
    with pytest.raises(ValueError, match="trailing"):
        serialize.decode(blob + b"\0", params.pp)
    with pytest.raises(TypeError):
        serialize.decode("text", params.pp)
    unreduced = bytearray(blob)
    unreduced[4:8] = b"\xff\xff\xff\xff"                # first coefficient of t_A = 2^32 - 1 >= q
    with pytest.raises(ValueError, match="mod q"):
        serialize.decode(bytes(unreduced), params.pp)


def test_flipped_bit_does_not_verify(signed):
    params, ring, sig = signed
    blob = bytearray(serialize.encode(sig, params.pp))
    for pos in (10, len(blob) // 2, len(blob) - 3):
        b = bytearray(blob)
        b[pos] ^= 1
        try:
            res = ringsig.verify(params, ring, b"msg", serialize.decode(bytes(b), params.pp))
            assert not res.ok
        except ValueError:
            pass                                          # rejected while decoding: also fine
