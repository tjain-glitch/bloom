"""Randomness: uniform, signs, discrete Gaussians, challenges, and seed expansion.

Every sampler takes an optional `rng`, which can be any random.Random-compatible object.
  - rng=None (the default) uses secrets.SystemRandom, i.e. the OS CSPRNG. Use this for
    anything secret: keys, masking vectors, commitment randomness.
  - XofRandom(seed) gives a deterministic stream from SHAKE-256. Use it for things that
    must be reproducible by the verifier: public matrices from a seed, Fiat-Shamir challenges.
  - random.Random(1234) is fine in tests, never in real code (it's not cryptographic).

None of this is constant-time. It's a research prototype, see the README limitations.
"""

from __future__ import annotations

import hashlib
import math
import random
import secrets

import numpy as np

from .ring import Poly, PolyMat, PolyVec, Ring, negacyclic_mul

_SYSTEM_RNG = secrets.SystemRandom()

# Discrete Gaussian tail cut, in standard deviations. Pr[|x| > 14 s] < 2^-140 (Lemma 2.5 style
# bound), so cutting the tail here changes the distribution negligibly.
GAUSSIAN_TAIL = 14


def _rng(rng: random.Random | None) -> random.Random:
    return _SYSTEM_RNG if rng is None else rng


class XofRandom(random.Random):
    """A deterministic random.Random driven by SHAKE-256 in counter mode.

    Block i of the stream is SHAKE-256("bloom-xof" || seed || i). We use counter mode rather
    than one long squeeze because hashlib's shake objects can't be read incrementally.
    """

    _BLOCK = 136  # SHAKE-256 rate in bytes

    def __init__(self, seed: bytes) -> None:
        if not isinstance(seed, (bytes, bytearray)) or len(seed) < 16:
            raise ValueError("XofRandom seed must be at least 16 bytes")
        self._seed = bytes(seed)
        self._ctr = 0
        self._buf = b""
        super().__init__()

    def seed(self, a=None, version=2) -> None:  # noqa: D401 - random.Random API
        # random.Random.__init__ calls seed(); our state comes from the constructor instead.
        pass

    def _read(self, n: int) -> bytes:
        while len(self._buf) < n:
            h = hashlib.shake_256(b"bloom-xof" + self._seed + self._ctr.to_bytes(8, "little"))
            self._buf += h.digest(self._BLOCK)
            self._ctr += 1
        out, self._buf = self._buf[:n], self._buf[n:]
        return out

    def getrandbits(self, k: int) -> int:
        if k < 0:
            raise ValueError("number of bits must be non-negative")
        if k == 0:
            return 0
        nbytes = (k + 7) // 8
        return int.from_bytes(self._read(nbytes), "little") >> (8 * nbytes - k)

    def random(self) -> float:
        return self.getrandbits(53) / (1 << 53)

    def getstate(self):
        raise NotImplementedError("XofRandom state is not exportable")

    def setstate(self, state) -> None:
        raise NotImplementedError("XofRandom state is not importable")


# --- uniform --------------------------------------------------------------------------


def uniform_poly(ring: Ring, rng: random.Random | None = None) -> Poly:
    r = _rng(rng)
    return Poly(ring, [r.randrange(ring.q) for _ in range(ring.d)])


def uniform_vec(ring: Ring, k: int, rng: random.Random | None = None) -> PolyVec:
    return PolyVec(ring, [uniform_poly(ring, rng) for _ in range(k)])


def uniform_mat(ring: Ring, rows: int, cols: int, rng: random.Random | None = None) -> PolyMat:
    return PolyMat(ring, [[uniform_poly(ring, rng) for _ in range(cols)] for _ in range(rows)])


def expand_mat(ring: Ring, seed: bytes, label: str, rows: int, cols: int) -> PolyMat:
    """Deterministically derive a uniform public matrix (e.g. A, A1, A2, B) from a seed.

    Different labels give independent matrices from the same seed.
    """
    rng = XofRandom(hashlib.sha3_256(b"bloom-expand|" + label.encode() + b"|" + seed).digest())
    return uniform_mat(ring, rows, cols, rng)


# --- small secrets -----------------------------------------------------------------------


def sign_poly(ring: Ring, rng: random.Random | None = None) -> Poly:
    """Coefficients uniform in {-1, 1}. This is the secret key distribution in App. C.2."""
    r = _rng(rng)
    return Poly(ring, [1 if r.getrandbits(1) else -1 for _ in range(ring.d)])


def sign_vec(ring: Ring, k: int, rng: random.Random | None = None) -> PolyVec:
    return PolyVec(ring, [sign_poly(ring, rng) for _ in range(k)])


def random_sign(rng: random.Random | None = None) -> int:
    """A single b in {-1, 1}, as used by bimodal rejection sampling."""
    return 1 if _rng(rng).getrandbits(1) else -1


# --- discrete Gaussian -------------------------------------------------------------------


def gaussian_int(s: float, rng: random.Random | None = None) -> int:
    """Sample from the discrete Gaussian over Z with standard deviation s (Definition 2.4).

    Plain rejection sampling: pick x uniformly in [-B, B] with B = ceil(14 s) and accept
    with probability exp(-x^2 / (2 s^2)). Expected about 11 tries per sample. Simple and
    easy to check, but slow and not constant-time.
    """
    if not s > 0:
        raise ValueError(f"standard deviation must be positive, got {s}")
    r = _rng(rng)
    bound = math.ceil(GAUSSIAN_TAIL * s)
    two_s2 = 2.0 * s * s
    while True:
        x = r.randrange(-bound, bound + 1)
        if r.random() < math.exp(-(x * x) / two_s2):
            return x


def gaussian_poly(ring: Ring, s: float, rng: random.Random | None = None) -> Poly:
    return Poly(ring, [gaussian_int(s, rng) for _ in range(ring.d)])


def gaussian_vec(ring: Ring, k: int, s: float, rng: random.Random | None = None) -> PolyVec:
    return PolyVec(ring, [gaussian_poly(ring, s, rng) for _ in range(k)])


# --- challenge space (Section 2.5) -------------------------------------------------------


def _int_poly_pow2(c: np.ndarray, log_e: int) -> np.ndarray:
    """c^(2^log_e) in Z[X]/(X^d + 1), exactly (no modulus, so Python big ints)."""
    for _ in range(log_e):
        c = negacyclic_mul(c, c)
    return c


def challenge_norm_ok(coeffs: np.ndarray, eta: int, chal_pow: int) -> bool:
    """The second condition in eq. (15): (||c^(2k)||_1)^(1/2k) <= eta, with k = chal_pow.

    This bounds the operator norm of multiplication by c, so ||c r|| <= eta ||r||.
    We compare ||c^(2k)||_1 <= eta^(2k) exactly with integers rather than taking roots.
    """
    e = 2 * chal_pow
    power = _int_poly_pow2(np.array([int(x) for x in coeffs], dtype=object), e.bit_length() - 1)
    return sum(abs(int(x)) for x in power) <= eta ** e


def challenge_poly(
    ring: Ring, kappa: int, eta: int, chal_pow: int, rng: random.Random | None = None
) -> Poly:
    """Sample c from C = {c : sigma_{-1}(c) = c, ||c||_inf <= kappa, eta check} (eq. 15).

    sigma_{-1}(X^i) = -X^(d-i), so sigma_{-1}(c) = c forces c_(d-i) = -c_i and c_(d/2) = 0.
    We pick c_0, ..., c_(d/2 - 1) uniformly from [-kappa, kappa], fill in the rest, and
    resample if the eta check fails (the paper picks eta so that happens < 1% of the time).
    """
    r = _rng(rng)
    d = ring.d
    while True:
        c = np.zeros(d, dtype=object)
        for i in range(d // 2):
            c[i] = r.randrange(-kappa, kappa + 1)
        for i in range(1, d // 2):
            c[d - i] = -c[i]
        if challenge_norm_ok(c, eta, chal_pow):
            return Poly(ring, c)


# --- vectorised helpers (added for the proof layer) ----------------------------------------
#
# The functions above draw one integer at a time, which is easy to read but slow for the
# thousands of coefficients a signature needs. The array versions below produce the same
# distributions from bulk random bytes. Secret values (masks, commitment randomness) still
# come from the OS CSPRNG; public values come from SHAKE-256 so the verifier can rebuild them.


def random_bytes(n: int, rng: random.Random | None = None) -> bytes:
    """n random bytes: OS CSPRNG by default, or drawn from `rng` (deterministic tests/XOF)."""
    if n <= 0:
        return b""
    if rng is None:
        return secrets.token_bytes(n)
    return rng.getrandbits(8 * n).to_bytes(n, "little")


def shake_bytes(seed: bytes, label: str, n: int) -> bytes:
    """n bytes of SHAKE-256(label || seed), used to expand public values from a seed."""
    return hashlib.shake_256(b"bloom-array|" + label.encode() + b"|" + seed).digest(n)


def _uniform_from_bytes(next_bytes, count: int, modulus: int) -> np.ndarray:
    """Uniform ints in [0, modulus) by rejection on 32-bit words (no modulo bias)."""
    if not 2 <= modulus <= 1 << 32:
        raise ValueError("modulus must be in [2, 2^32]")
    limit = ((1 << 32) // modulus) * modulus
    out = np.empty(0, dtype=np.int64)
    while out.size < count:
        need = count - out.size
        words = np.frombuffer(next_bytes(4 * (need + need // 8 + 8)), dtype="<u4").astype(np.int64)
        out = np.concatenate([out, words[words < limit] % modulus])
    return out[:count]


def uniform_array(shape: tuple[int, ...], modulus: int, rng: random.Random | None = None) -> np.ndarray:
    """Uniform int64 array in [0, modulus) with the given shape (secret randomness)."""
    count = int(np.prod(shape))
    return _uniform_from_bytes(lambda n: random_bytes(n, rng), count, modulus).reshape(shape)


def expand_array(seed: bytes, label: str, shape: tuple[int, ...], modulus: int) -> np.ndarray:
    """Deterministic uniform array from a public seed (fast version of expand_mat).

    Each call with a fresh counter block keeps the stream independent of earlier draws.
    """
    counter = [0]

    def next_bytes(n: int) -> bytes:
        counter[0] += 1
        return shake_bytes(seed + counter[0].to_bytes(4, "little"), label, n)

    count = int(np.prod(shape))
    return _uniform_from_bytes(next_bytes, count, modulus).reshape(shape)


def ternary_array(shape: tuple[int, ...], rng: random.Random | None = None) -> np.ndarray:
    """Coefficients uniform in {-1, 0, 1}: the commitment randomness s2 (Section 2.6)."""
    return uniform_array(shape, 3, rng) - 1


def sign_array(shape: tuple[int, ...], rng: random.Random | None = None) -> np.ndarray:
    """Coefficients uniform in {-1, 1}: secret keys in App. C.2."""
    count = int(np.prod(shape))
    bits = np.unpackbits(np.frombuffer(random_bytes((count + 7) // 8, rng), dtype=np.uint8))
    return (2 * bits[:count].astype(np.int64) - 1).reshape(shape)


def binomial_array(shape: tuple[int, ...], seed: bytes, label: str) -> np.ndarray:
    """Public matrix with entries in {-1, 0, 1} w.p. 1/4, 1/2, 1/4 (difference of two bits).

    This is the Bin_1 distribution LNP22 uses for its Johnson-Lindenstrauss projection.
    """
    count = int(np.prod(shape))
    bits = np.unpackbits(np.frombuffer(shake_bytes(seed, label, (2 * count + 7) // 8), dtype=np.uint8))
    bits = bits[: 2 * count].astype(np.int64).reshape(count, 2)
    return (bits[:, 0] - bits[:, 1]).reshape(shape)


def gaussian_array(shape: tuple[int, ...], s: float, rng: random.Random | None = None) -> np.ndarray:
    """Discrete Gaussian D_s over Z (Definition 2.4), vectorised version of gaussian_int.

    Same method: candidate x uniform in [-B, B] with B = ceil(14 s), accept with probability
    exp(-x^2 / (2 s^2)). Not constant-time (see README limitations).
    """
    if not s > 0:
        raise ValueError(f"standard deviation must be positive, got {s}")
    count = int(np.prod(shape))
    bound = math.ceil(GAUSSIAN_TAIL * s)
    width = 2 * bound + 1
    if width > 1 << 32:
        raise ValueError("standard deviation too large for the 32-bit sampler")
    out = np.empty(0, dtype=np.int64)
    two_s2 = 2.0 * s * s
    while out.size < count:
        n = 2 * (count - out.size) * GAUSSIAN_TAIL // 2 + 64
        x = uniform_array((n,), width, rng) - bound
        u = np.frombuffer(random_bytes(8 * n, rng), dtype="<u8") >> np.uint64(11)
        u = u.astype(np.float64) / float(1 << 53)
        out = np.concatenate([out, x[u < np.exp(-(x.astype(np.float64) ** 2) / two_s2)]])
    return out[:count].reshape(shape)


# --- rejection sampling (Lemma 2.8, Fig. 3) ------------------------------------------------


def rej1_M(gamma: float) -> float:
    """Repetition rate of Rej1 for s = gamma * T (Lemma 2.8, part 1)."""
    return math.exp(14.0 / gamma + 1.0 / (2.0 * gamma * gamma))


def bimodal_M(gamma: float) -> float:
    """Repetition rate of the bimodal test Rej2 for s = gamma * T (Lemma 2.8, part 2)."""
    return math.exp(1.0 / (2.0 * gamma * gamma))


def _inner_and_norm(z: np.ndarray, v: np.ndarray) -> tuple[float, float]:
    zf = np.asarray(z, dtype=np.float64).ravel()
    vf = np.asarray(v, dtype=np.float64).ravel()
    return float(np.dot(zf, vf)), float(np.dot(vf, vf))


def _log_uniform(rng: random.Random | None) -> float:
    u = _rng(rng).random()
    return math.log(u) if u > 0 else -math.inf


def rej1(z: np.ndarray, v: np.ndarray, s: float, M: float, rng: random.Random | None = None) -> bool:
    """Rej1 from Fig. 3 (left). Returns True if we should reject.

    Reject if u > (1/M) * exp((-2<z,v> + ||v||^2) / (2 s^2)), computed in log space so the
    exponential never overflows. z and v are centred integer vectors.
    """
    zv, vv = _inner_and_norm(z, v)
    log_threshold = (-2.0 * zv + vv) / (2.0 * s * s) - math.log(M)
    return _log_uniform(rng) > log_threshold


def _log_cosh(x: float) -> float:
    ax = abs(x)
    return ax + math.log1p(math.exp(-2.0 * ax)) - math.log(2.0)


def rej_bimodal(z: np.ndarray, v: np.ndarray, s: float, M: float, rng: random.Random | None = None) -> bool:
    """Rej2, the bimodal test from Fig. 3 (right). Returns True if we should reject.

    Reject if u > 1 / (M * exp(-||v||^2 / (2 s^2)) * cosh(<z,v> / s^2)). (The figure prints
    sigma^2 inside the cosh; it is the same s^2 as elsewhere, as in DDLL13.)
    Used with z = y + b*v for a random sign b (Sec. 3.2), or z = y + v for the commitment
    randomness, where Lemma 3.1 shows the sign is not needed (Sec. 3.1).
    """
    zv, vv = _inner_and_norm(z, v)
    log_threshold = -math.log(M) + vv / (2.0 * s * s) - _log_cosh(zv / (s * s))
    return _log_uniform(rng) > log_threshold
