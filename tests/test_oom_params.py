"""Derived proof parameters. These are FUNCTIONAL checks; no security level is claimed."""

import math

import pytest

from bloom import oom_params
from bloom.oom_params import JL_ROWS, lam_for


def test_paper_scale_k2_sizes():
    pp = oom_params.paper_scale(2)
    assert pp.ps.ring_size == 4096
    assert pp.m1 == 12 + 2 + 1                      # s, u_1, u_2, v_3
    assert pp.n_y3 == JL_ROWS // 64
    assert pp.lam == 9
    assert pp.num_zq_relations == 1 + 3 + 56 * 2 + 64 + 8 * 2 + 256


def test_lambda_gives_128_bit_lift():
    assert lam_for(65437) == 9
    assert 65437 ** -lam_for(65437) <= 2.0 ** -128


@pytest.mark.parametrize("k", [2, 4, 6])
@pytest.mark.parametrize("mode", ["rej1", "bimodal"])
def test_paper_presets_are_consistent(k, mode):
    pp = oom_params.paper_scale(k, mode)
    assert pp.no_wraparound_margin() > 1
    assert 1 < pp.expected_repetitions < 6
    assert pp.B1 > pp.std1 and pp.B2 > pp.std2


def test_bimodal_is_narrower_at_same_repetition_rate():
    """Section 3: same M per step, but sigma about 33x smaller."""
    r, b = oom_params.paper_scale(2, "rej1"), oom_params.paper_scale(2, "bimodal")
    assert abs(r.M1 - b.M1) < 0.01
    assert b.std1 * 30 < r.std1


def test_rejection_constants():
    pp = oom_params.paper_scale(2)
    assert math.isclose(pp.M1, math.exp(14 / 40 + 1 / (2 * 40 ** 2)))
    bp = oom_params.paper_scale(2, "bimodal")
    assert math.isclose(bp.M1, math.exp(1 / (2 * 1.2 ** 2)))


def test_invalid_parameters():
    with pytest.raises(ValueError):
        oom_params.paper_scale(1, "bimodal")         # Lemma C.1: odd k not sign-invariant
    with pytest.raises(ValueError):
        oom_params.toy(0)                            # k = 0 not implemented
    with pytest.raises(ValueError):
        oom_params.paper_scale(2, "rej3")
    from dataclasses import replace
    with pytest.raises(ValueError):
        replace(oom_params.paper_scale(2), gamma3=200.0)   # JL bound would allow wrap-around
