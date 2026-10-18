from dataclasses import replace

import pytest

from bloom.params import ALL, RINGSIG_K0, RINGSIG_K2, is_prime, ring_signature_params


def test_ring_sizes_match_fig10():
    # Fig. 10: N = d * base^k for k = 0, 2, 4, 6
    assert [ALL[f"ringsig-k{k}"].ring_size for k in (0, 2, 4, 6)] == [2**6, 2**12, 2**18, 2**24]


def test_key_sizes_match_appendix_c2():
    # "the user public key (resp. secret key) has size 128B (resp. 96B)"
    assert RINGSIG_K0.pk_bytes == 128
    assert RINGSIG_K0.sk_bytes == 96


def test_modulus():
    assert RINGSIG_K2.q == 65437 * 65629
    assert RINGSIG_K2.q < 2**32
    assert RINGSIG_K2.q % RINGSIG_K2.p == 0


def test_is_prime():
    assert [n for n in range(30) if is_prime(n)] == [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    assert is_prime(65437) and is_prime(65629)
    assert not is_prime(65437 * 65629)


@pytest.mark.parametrize(
    "change, msg",
    [
        ({"k": 1}, "even"),
        ({"d": 48}, "power of two"),
        ({"q2": 65537}, "5 mod 8"),        # prime, but 1 mod 8
        ({"q2": 65629 + 8}, "not prime"),
        ({"p": 12289}, "divide"),
        ({"base": 0}, "base"),
    ],
)
def test_invalid_params_rejected(change, msg):
    with pytest.raises(ValueError, match=msg):
        replace(RINGSIG_K0, **change)


def test_papers_literal_ell_is_inconsistent():
    # The paper states (d, base, ell) = (64, 8, 16) but the packing needs ell * base <= d.
    # This test documents why we deviate to ell = 8.
    with pytest.raises(ValueError, match="ell"):
        replace(ring_signature_params(2), ell=16)
