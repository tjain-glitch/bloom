import pytest

from bloom.params import RINGSIG_K0 as P
from bloom.ring import Ring
from bloom.transcript import Transcript

R = Ring(P.d, P.q)


def make(msg=b"hello", extra=None):
    t = Transcript("test")
    t.absorb("msg", msg)
    t.absorb("x", R.monomial(3))
    if extra is not None:
        t.absorb("extra", extra)
    return t


def test_same_inputs_same_challenges():
    assert make().challenge_bytes("c") == make().challenge_bytes("c")


def test_different_inputs_different_challenges():
    assert make(b"hello").challenge_bytes("c") != make(b"hellp").challenge_bytes("c")
    assert make(extra=1).challenge_bytes("c") != make(extra=2).challenge_bytes("c")


def test_order_matters():
    a, b = Transcript("t"), Transcript("t")
    a.absorb("x", b"1"); a.absorb("y", b"2")
    b.absorb("y", b"2"); b.absorb("x", b"1")
    assert a.challenge_bytes("c") != b.challenge_bytes("c")


def test_framing_prevents_ambiguity():
    a, b = Transcript("t"), Transcript("t")
    a.absorb("a", b"bc")
    b.absorb("ab", b"c")
    assert a.challenge_bytes("c") != b.challenge_bytes("c")


def test_type_tag_prevents_ambiguity():
    a, b = Transcript("t"), Transcript("t")
    a.absorb("v", b"\x05")
    b.absorb("v", 5)
    assert a.challenge_bytes("c") != b.challenge_bytes("c")


def test_protocol_label_separates_domains():
    a, b = Transcript("ringsig"), Transcript("other")
    assert a.challenge_bytes("c") != b.challenge_bytes("c")


def test_repeated_label_gives_fresh_challenge():
    t = make()
    assert t.challenge_bytes("c") != t.challenge_bytes("c")


def test_challenge_poly_is_valid_and_reproducible():
    c1 = make().challenge_poly("c", R, P.kappa, P.eta, P.chal_pow)
    c2 = make().challenge_poly("c", R, P.kappa, P.eta, P.chal_pow)
    assert c1 == c2
    assert c1.conj() == c1 and c1.norm_inf() <= P.kappa


def test_challenge_ints_in_range():
    xs = make().challenge_ints("gamma", 500, P.q)
    assert len(xs) == 500 and all(0 <= x < P.q for x in xs)


def test_fork_is_independent():
    t = make()
    f = t.fork()
    assert f.challenge_bytes("c") == t.challenge_bytes("c")
    f.absorb("only-in-fork", b"x")
    assert f.challenge_bytes("d") != t.challenge_bytes("d")


def test_rejects_bad_values():
    t = Transcript("t")
    with pytest.raises(ValueError):
        t.absorb("neg", -1)
    with pytest.raises(TypeError):
        t.absorb("float", 1.5)
    with pytest.raises(ValueError):
        t.challenge_bytes("c", 0)
