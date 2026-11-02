# BLOOM: lattice one-out-of-many proofs (FIT5163 group project)

This is our group's repo for the FIT5163 programming assignment (Monash, Semester 2 2026). We were assigned the paper

> **BLOOM: Bimodal Lattice One-Out-of-Many Proofs and Applications**
> Vadim Lyubashevsky (IBM Research Europe) and Ngoc Khanh Nguyen (EPFL)

We implement the paper's lattice ring signature (App. C) end to end: the recursive one-out-of-many proof of Section 4 (Figs. 9 and 8), a zero-knowledge base case built from the LNP22 toolbox, and both standard and bimodal rejection sampling. We then compare measured sizes with the paper's estimates.

## What the paper is about (short version)

A one-out-of-many proof shows you know a secret for *one* entry of a public list without revealing which. With a list of public keys `t_1 ... t_N`, you prove you know a short `s` with `A·s = t_i mod p` for some `i`, leaking neither `i` nor `s`. Put everyone's key in the list and hash the message into Fiat-Shamir, and you get a **ring signature**.

The main ideas:

- **Tensor decomposition.** The index `i` becomes a one-hot vector `v = v_1 ⊗ v_2 ⊗ ... ⊗ v_{k+1}`, and the relation is proved recursively, so the proof grows with `log N`.
- **ABDLOP commitments and LNP22** (Lyubashevsky, Nguyen, Plançon, Crypto 2022) prove the base case in zero knowledge.
- **Bimodal Gaussian rejection sampling** (from BLISS / DDLL13) gives narrower masks at the same repetition rate, so proofs are smaller.
- **Exact norm proofs** allow a small modulus, so public keys are tiny (128 bytes).

## Scope

**Implemented:**
- Ring arithmetic over `R_q = Z_q[X]/(X^64 + 1)` with `q = 65437 · 65629`
- ABDLOP commitment, Fiat-Shamir transcript, Rej1 and bimodal rejection sampling (Fig. 3)
- Tensor decomposition and the recursive folds (Figs. 9 and 8)
- The base case `Π_k` as a genuine zero-knowledge proof (LNP22: JL approximate shortness, `Z_q` to `R_q` lift, Fig. 7 quadratic proof)
- Ring signature KeyGen / Sign / Verify (App. C.2), signature encoding, demo and benchmarks

**Not implemented:**
- The group signature (App. E) and integer proofs (App. D)
- `k = 0` (a direct proof without recursion). `k = 6` (N = 2^24) is supported but not run, since the ring alone needs about 8.6 GB.
- Constant-time or side-channel hardened code. This is a research prototype.

## Repo layout

```
src/bloom/
  params.py       Fig. 10 / Fig. 4 parameter sets
  ring.py         R_q arithmetic: Poly, PolyVec, PolyMat, sigma, rot(), fast int64 array helpers
  sampling.py     CSPRNG / SHAKE randomness, Gaussians, challenge space, Rej1 and bimodal
  transcript.py   generic Fiat-Shamir transcript (SHAKE-256)
  oom_params.py   derived proof parameters (commitment sizes, Gaussian widths, bounds)
  tensor.py       index <-> (v_1, ..., v_{k+1}) decomposition
  fold.py         T_j -> T_{j+1} and a_i = A^T phi_i
  recursion.py    one recursion level, packing w_j, Lemma 4.1 linking checks
  abdlop.py       ABDLOP commitment
  fiat_shamir.py  what the protocol absorbs, in which order
  base_case.py    Pi_k: R_k as relations + the LNP22 prover/verifier; OpeningChecker (debug only)
  oom.py          full one-out-of-many proof with restarts
  ringsig.py      KeyGen / Sign / Verify
  serialize.py    signature <-> bytes, size breakdown
tests/            pytest suite, one file per module plus statement-binding and demo tests
bench/            size.py, timing.py, naive_verifier.py; results/ holds their CSV output
demo.py           presentation demo
docs/             notes and plans
```

## Setup

Python 3.11 or newer. Nothing platform-specific.

```bash
python -m venv .venv
```

Activate it (`source .venv/bin/activate` on macOS/Linux, `.venv\Scripts\activate` on Windows), then:

```bash
pip install -r requirements.txt
pip install -e .
```

The foundation layer was developed on Windows 11 (Python 3.12.5); the proof layer was tested on Linux (Python 3.13, numpy 2.5).

## Running it

```bash
pytest                    # 204 tests, about 20 s
pytest -m "not slow"      # skips the statistical and full-size tests
python demo.py            # Fig. 10 parameters, 4096-key ring, about 2 s
```

`demo.py` shows (1) a valid signature accepted, (2) the same signature rejected after one public key in the ring is changed, with the verifier's reason, and (3) the signer's index traced through the tensor decomposition and each recursion level. Options: `--mode bimodal`, `--signer <i>`, `--seed <n>`, `--toy` (small insecure keys, faster).

### What the tests check

- **Foundation:** ring arithmetic against a naive reference, `σ` and Lemma 2.2, samplers (moments, ranges, acceptance rate ≈ 1/M), transcript separation, parameter validation
- **Recursion:** the tensor decomposition, the fold identity, and the Lemma 4.1 linking checks (including that a sign flip is harmless only for even `k`, Lemma C.1)
- **Base case:** honest proofs accepted; for each condition of `R_k`, a prover with a witness that breaks it is rejected; the relation encoding agrees with an independent opening checker; any tampered proof field is rejected
- **Ring signature:** sign/verify for any ring member, bad inputs rejected without crashing, encoding round-trips; the same signature fails after changing a key, the ring order, the message or the public parameters (`test_statement_binding.py`)

These are functional tests. They show the code follows the paper's algorithms and catches the cheating cases we wrote; they are not a security proof. Security rests on MSIS, MLWE and the paper's Extended-MLWE assumption.

## Results

Measured on a 4-core Linux machine. "Measured" is `len(serialize.encode(signature))`; Fig. 10 numbers are the paper's estimates.

| N | k | Fig. 10 estimate | Measured (Rej1) | Measured (bimodal) |
|---|---|---|---|---|
| 2^6  | 0 | 13.1 KB | not implemented | not implemented |
| 2^12 | 2 | 14.1 KB | 22.0 KB | 19.5 KB |
| 2^18 | 4 | 15.1 KB | 23.2 KB | 20.4 KB |
| 2^24 | 6 | 16.2 KB | not run | not run |

- Keys: 128 B public, 96 B secret (App. C.2 says the same).
- N = 4096, 20 signatures per mode: sign 0.37 s (Rej1) / 0.40 s (bimodal), verify 0.15 s, 3.2 / 3.6 attempts per signature on average.
- `bench/naive_verifier.py`, 100 cheating witnesses: a check that only opens the commitments accepts all 100; `Π_k` accepts 0.
- Challenge space (Sec. 2.5): 99.15% of 2000 random candidates pass the η = 140 check (the paper says at least 99%).

## Reproducing the results

```bash
python bench/size.py           # sizes vs Fig. 10, k = 2 and 4 (about 40 s, about 650 MB RAM)
python bench/timing.py         # sign/verify times and attempts, Rej1 vs bimodal (about 25 s)
python bench/naive_verifier.py # opening-only checkers vs Pi_k on cheating witnesses (about 20 s)
```

Each writes CSV files to `bench/results/` (overwriting the committed ones). `--help` lists the options.

## Deviations from the paper

- **ℓ = 8 instead of 16.** The paper states `(d, 𝔡, ℓ) = (64, 8, 16)` but also needs `ℓ·𝔡 ≤ d` to pack `w_j` into one polynomial, and 16·8 = 128 > 64. With ℓ = 8, soundness per fold is about `q1^-8 ≈ 2^-128`. See `tests/test_params.py::test_papers_literal_ell_is_inconsistent`.
- **Keys mod p, proof mod q.** The paper says "if p divides q" without detail. We multiply the ring `T` and `A` by `q2`, which turns `A·s = T·v mod p` into an equivalent equation mod `q = p·q2`.
- **Fig. 9's `A1` has `m+k` columns,** but `s1 = (s, u_1, ..., u_k, v_{k+1})` has `m+k+1`. We use `m+k+1`.
- **Proof parameters the paper does not give** (commitment rows, randomness length, Gaussian widths, `λ = 9` garbage polynomials) are chosen in `oom_params.py` so the protocol works. They are not checked with a lattice estimator, so we claim no concrete security level for them.
- **JL step uses standard rejection sampling,** not LNP22's committed-sign bimodal variant. The JL constants (√337 honest bound, √30 extraction bound) are taken from LNP22 and the TU Graz code as read, not re-derived.
- **No LNP22 Sec. 6.1 size optimisations.** Every mod-q value is sent at 32 bits.
- **Rings must have exactly N = 64·8^k keys** (no padding).
- **Discrete Gaussian sampler.** Rejection from a bounded uniform (tail cut at 14σ). Simple to verify, not constant-time.
- **Fiat-Shamir.** SHAKE-256 with length-prefixed, type-tagged framing. The statement (parameters, commitment-key seed, `A`, the full ordered ring, the message) is absorbed first. Any rejection restarts the whole signature with fresh randomness.
- **Public matrices** are expanded from a seed with SHAKE-256. **Secret randomness** comes from the OS CSPRNG (`secrets`); seeded generators are only used in tests and benchmarks.

## Third-party code, libraries and AI use

Libraries: **numpy** (arrays, convolution, int64 arithmetic with 16-bit limb splitting so products stay exact for q < 2^32), **pytest**, and the Python standard library (`hashlib` for SHAKE-256, `secrets`).

No code is copied from other repositories. The TU Graz "lattice-zk" notebook (F. Lugstein, 2023, https://extgit.iaik.tugraz.at/public-shared/lattice-zk) was read as a reference for how LNP22's sub-protocols fit together. It has no licence, so nothing was copied; `base_case.py` is our own non-interactive implementation for `q = q1·q2`. The paper's authors released no code.

AI assistance:

- The foundation layer (`params.py`, `ring.py`, `sampling.py`, `transcript.py` and their tests) was drafted with AI assistance, then reviewed and tested by the team.
- Everything added after it was generated with Claude (Anthropic) from an implementation design the team agreed on: the remaining modules in `src/bloom/`, the new tests, `demo.py` and `bench/`. This includes the internal rewrite of `ring.py` (same API, int64 storage) and the array samplers added to `sampling.py`. The team reviews this code and is responsible for it.

## Team

| Name | Student ID | Email |
|---|---|---|
| Tushar Jain | 35896493 | tjai0005@student.monash.edu |
| Shanshan Li | 36404179 | slii0373@student.monash.edu |
| Shaopei Fu | 36858579 | sfuu0033@student.monash.edu |
| Mitchell Padula | 33114579 | mpad0004@student.monash.edu |

Branch and commit rules are in [CONTRIBUTING.md](CONTRIBUTING.md). Everyone is expected to understand the whole codebase, since the Q&A can ask any of us about any part.

## Key dates

- **Presentation + live demo:** Thursday 5 Nov 2026
- **Report and code due:** Friday 6 Nov 2026, 23:55

## Reference

V. Lyubashevsky and N. K. Nguyen, *BLOOM: Bimodal Lattice One-Out-of-Many Proofs and Applications*.
