"""Shared helpers for the proof-layer tests.

make_instance() builds a one-out-of-many statement directly (no ring-signature layer): a random
key matrix A, one real key pair (s, A s), and a ring T of random columns with the real public
key at `index`. Everything is lifted from mod p to mod q exactly as ringsig.py does it.
"""

from __future__ import annotations

import random
from types import SimpleNamespace

import numpy as np
import pytest

from bloom import abdlop, oom, oom_params
from bloom.fiat_shamir import OomTranscript
from bloom.ring import matmul_mod, rot_batch
from bloom.sampling import sign_array, uniform_array


def make_instance(pp, index: int | None = None, seed: int = 1, message: bytes = b"test message"):
    rng = random.Random(seed)
    ps = pp.ps
    d, q, p, q2 = ps.d, ps.q, ps.p, ps.q2
    A = uniform_array((ps.n, ps.m, d), p, rng)
    A_rot = np.block([[rot_batch(A[i, j], p) for j in range(ps.m)] for i in range(ps.n)])
    s = sign_array((ps.m, d), rng)
    pk = matmul_mod(A_rot, (s.reshape(-1) % p)[:, None], p)[:, 0]
    T_p = uniform_array((ps.n * d, ps.ring_size), p, rng)
    index = rng.randrange(ps.ring_size) if index is None else index
    T_p[:, index] = pk
    stmt = oom.Statement(T=(T_p * q2) % q, A_rot=(A_rot * q2) % q)
    wit = oom.Witness(s=s, index=index)
    ck = abdlop.keygen(pp, b"test-commitment-key-seed")

    def transcript(msg: bytes = message, T=None) -> OomTranscript:
        tr = OomTranscript(pp)
        tr.absorb_statement(ck.seed, A, T_p if T is None else T, msg)
        return tr

    return SimpleNamespace(pp=pp, ps=ps, A=A, A_rot=A_rot, T_p=T_p, s=s, index=index,
                           stmt=stmt, wit=wit, ck=ck, transcript=transcript, rng=rng)


@pytest.fixture(scope="session")
def toy_rej1():
    return make_instance(oom_params.toy(2, "rej1"), seed=11)


@pytest.fixture(scope="session")
def toy_bimodal():
    return make_instance(oom_params.toy(2, "bimodal"), seed=12)
