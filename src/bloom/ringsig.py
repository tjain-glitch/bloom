"""Ring signature from the one-out-of-many proof (App. C.2).

    Setup   public seed -> key matrix A in R_p^{n x m} and the ABDLOP commitment key
    KeyGen  sk = s <- {-1,1}^{md},  pk = A s mod p                       (128 B / 96 B, App. C.2)
    Sign    non-interactive proof that you know s with A s = pk_i for some i in the ring,
            with the message mu absorbed into the Fiat-Shamir transcript
    Verify  re-run the verifier of the proof against the ring and the message

Keys live mod p but the proof runs mod q = p * q2 (the paper: "if p divides q"). We multiply
both sides of the key equation by q2:

    A s = pk  (mod p)   <=>   q2 A s = q2 pk  (mod q)

so the proof's statement is T' = q2 [pk_1 ... pk_N] and A' = q2 rot(A), reduced mod q. A
cheating index or key changes the mod-p equation and therefore the mod-q one.

Anonymity rests on the proof being zero-knowledge (simulatable under MLWE and the paper's
Extended-MLWE); unforgeability on MSIS (App. C.2). Neither is something a test can prove; the
tests check the functional side and the statement binding.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property

import numpy as np

from . import abdlop, oom
from .base_case import VerifyResult
from .fiat_shamir import OomTranscript
from .oom import OomProof, ProveTrace, Statement, Witness
from .oom_params import ProofParams
from .ring import matmul_mod, rot_batch
from .sampling import expand_array, random_bytes, sign_array

Signature = OomProof


@dataclass(frozen=True)
class PublicParams:
    """Everything a signer and verifier share: proof parameters and one public seed."""

    pp: ProofParams
    seed: bytes

    def __post_init__(self) -> None:
        if not isinstance(self.seed, bytes) or len(self.seed) < 16:
            raise ValueError("public seed must be at least 16 bytes")

    @cached_property
    def A(self) -> np.ndarray:
        """Key matrix A in R_p^{n x m}, shape (n, m, d)."""
        ps = self.pp.ps
        return expand_array(self.seed, "ringsig-A", (ps.n, ps.m, ps.d), ps.p)

    @cached_property
    def A_rot(self) -> np.ndarray:
        """rot(A) in Z_p^{nd x md} (eq. 28)."""
        ps = self.pp.ps
        return np.block([[rot_batch(self.A[i, j], ps.p) for j in range(ps.m)] for i in range(ps.n)])

    @cached_property
    def ck(self) -> abdlop.CommitmentKey:
        return abdlop.keygen(self.pp, self.seed + b"|abdlop")


def setup(pp: ProofParams, seed: bytes | None = None) -> PublicParams:
    """Fresh public parameters (or reproducible ones from a given seed)."""
    return PublicParams(pp=pp, seed=seed if seed is not None else random_bytes(32))


@dataclass
class KeyPair:
    sk: np.ndarray = field(repr=False)   # (m, d) entries in {-1, 1}
    pk: np.ndarray                       # (n*d,) coefficients mod p


def public_key(params: PublicParams, sk: np.ndarray) -> np.ndarray:
    p = params.pp.ps.p
    return matmul_mod(params.A_rot, (sk.reshape(-1) % p)[:, None], p)[:, 0]


def keygen(params: PublicParams, rng=None) -> KeyPair:
    ps = params.pp.ps
    sk = sign_array((ps.m, ps.d), rng)
    return KeyPair(sk=sk, pk=public_key(params, sk))


def keygen_many(params: PublicParams, count: int, rng=None) -> list[KeyPair]:
    """Many key pairs at once (one matrix product), for demos and benchmarks."""
    ps = params.pp.ps
    sks = sign_array((count, ps.m, ps.d), rng)
    pks = matmul_mod(params.A_rot, (sks.reshape(count, -1).T % ps.p), ps.p).T
    return [KeyPair(sk=sks[i], pk=pks[i].copy()) for i in range(count)]


def _ring_matrix(params: PublicParams, ring) -> np.ndarray:
    """Validate a ring (sequence of public keys) and return it as T, shape (n*d, N), mod p."""
    ps = params.pp.ps
    R = np.asarray([np.asarray(pk) for pk in ring])
    if R.ndim != 2 or R.shape[1] != ps.n * ps.d:
        raise ValueError(f"each public key must have {ps.n * ps.d} coefficients")
    if R.shape[0] != ps.ring_size:
        raise ValueError(f"this parameter set needs a ring of exactly N = {ps.ring_size} keys, "
                         f"got {R.shape[0]}")
    if not np.issubdtype(R.dtype, np.integer) or R.min() < 0 or R.max() >= ps.p:
        raise ValueError("public key coefficients must be integers in [0, p)")
    return R.T.astype(np.int64)


def _statement(params: PublicParams, T_p: np.ndarray) -> Statement:
    ps = params.pp.ps
    q, q2 = ps.q, ps.q2
    return Statement(T=(T_p * q2) % q, A_rot=(params.A_rot * q2) % q)


def _transcript(params: PublicParams, T_p: np.ndarray, message: bytes) -> OomTranscript:
    tr = OomTranscript(params.pp)
    tr.absorb_statement(params.seed, params.A, T_p, message)
    return tr


def sign(params: PublicParams, key: KeyPair, ring, message: bytes, rng=None,
         trace: ProveTrace | None = None) -> Signature:
    if not isinstance(message, (bytes, bytearray)):
        raise TypeError("message must be bytes")
    T_p = _ring_matrix(params, ring)
    matches = np.nonzero(np.all(T_p.T == key.pk, axis=1))[0]
    if matches.size == 0:
        raise ValueError("the signer's public key is not in the ring")
    if not np.array_equal(public_key(params, key.sk), key.pk):
        raise ValueError("secret key does not match public key")
    index = int(matches[0])
    stmt = _statement(params, T_p)
    wit = Witness(s=key.sk, index=index)
    if not oom.statement_holds(params.pp, stmt, wit):
        raise AssertionError("lifted statement does not hold; this is a bug")
    return oom.prove(params.pp, params.ck, stmt, wit, _transcript(params, T_p, bytes(message)), rng, trace)


def verify(params: PublicParams, ring, message: bytes, signature: Signature) -> VerifyResult:
    try:
        if not isinstance(message, (bytes, bytearray)):
            raise TypeError("message must be bytes")
        T_p = _ring_matrix(params, ring)
    except (TypeError, ValueError) as exc:
        return VerifyResult.reject(f"invalid input: {exc}")
    stmt = _statement(params, T_p)
    return oom.verify(params.pp, params.ck, stmt, signature, _transcript(params, T_p, bytes(message)))
