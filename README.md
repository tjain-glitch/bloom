# BLOOM: lattice one-out-of-many proofs (FIT5163 group project)

This is our group's code for the FIT5163 programming assignment (Monash, Semester 2 2026). We were assigned the paper

> **BLOOM: Bimodal Lattice One-Out-of-Many Proofs and Applications**
> Vadim Lyubashevsky (IBM Research Europe) and Ngoc Khanh Nguyen (EPFL)

We implement the paper's lattice ring signature (App. C) end to end: the recursive one-out-of-many proof of Section 4 (Figs. 9 and 8), a zero-knowledge base case built from the LNP22 toolbox, and both standard and bimodal rejection sampling. We then compare measured signature sizes with the paper's estimates.

**Contents:** [the paper](#what-the-paper-is-about-short-version) · [scope](#scope) · [requirements](#requirements) · [installation](#installation) · [configuration](#configuration) · [running the demo](#running-the-demo) · [tests](#running-the-tests) · [reproducing the results](#reproducing-the-results) · [results](#results) · [repo layout](#repo-layout) · [deviations](#deviations-from-the-paper) · [security notes](#security-notes-and-limitations) · [provenance](#third-party-code-libraries-and-ai-use) · [team](#team)

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

**Not implemented, and why:**
- **The group signature (App. E) and integer proofs (App. D).** They are further applications built on the same one-out-of-many proof. The ring signature is the paper's headline application and the one Fig. 10 gives sizes for, so it is what lets us check the paper's main claims (logarithmic size, small keys, the bimodal saving). Adding App. E would mean a second commitment layer and verifiable encryption, which did not fit in the teaching period.
- **`k = 0`** (a direct proof without recursion, N = 64). It needs a separate code path without folds; the recursion is the paper's contribution, so we spent the time there instead. Fig. 10's N = 2^6 row is therefore not reproduced.
- **`k = 6`** (N = 2^24) is supported by the code but not run, since the ring alone needs about 8.6 GB of RAM.
- **Constant-time / side-channel hardened code.** This is a research prototype (see [security notes](#security-notes-and-limitations)).

Effect on the evaluation: we can compare sizes for N = 2^12 and 2^18 only, and the security of the derived proof parameters is not estimated (see [deviations](#deviations-from-the-paper)).

## Requirements

**Software**

| | Version used | Notes |
|---|---|---|
| Python | 3.12.5 | 3.11 or newer is required (`pyproject.toml`) |
| numpy | 2.5.4 | pinned in `requirements.txt`; any numpy ≥ 1.26 should work |
| pytest | 9.1.1 | only needed for the tests |
| OS | Windows 11 Home (build 26200) | nothing platform-specific; the proof layer was also tested on Linux with Python 3.13 |

Everything else (`hashlib` for SHAKE-256, `secrets`, `argparse`) is in the Python standard library. No GPU, network access, external service or dataset is needed.

**Hardware.** The results below were measured on an Intel Core i7-11800H laptop (8 cores / 16 threads, 2.3 GHz base) with 64 GB RAM. The code is single-threaded. The tests and the demo need well under 1 GB of RAM; `bench/size.py` peaks at about 650 MB (the k = 4 ring has 262,144 keys).

## Installation

From the repository root:

```bash
python -m venv .venv
```

Activate it (`source .venv/bin/activate` on macOS/Linux, `.venv\Scripts\activate` on Windows), then:

```bash
pip install -r requirements.txt
pip install -e .
```

`pip install -e .` installs the `bloom` package from `src/` so that the tests, `demo.py` and the `bench/` scripts can `import bloom`. To check the install:

```bash
python -c "import bloom, numpy; print('ok', numpy.__version__)"
```

## Configuration

There are no configuration files, secrets or environment variables. Everything is set in code or on the command line:

- **Paper parameters** (Fig. 10 and Fig. 4: `d`, `p`, `q`, `n`, `m`, base, `ℓ`, `k`, the challenge space) are in `src/bloom/params.py`.
- **Derived proof parameters** the paper does not give (commitment sizes, Gaussian widths, bounds) are in `src/bloom/oom_params.py`. It has two presets: `paper_scale(k, mode)` (Fig. 10 keys, `m = 12`) and `toy(k, mode)` (tiny, **insecure** keys with `m = 2`, used by the test suite so it runs fast).
- **Rejection sampling mode** is `rej1` (standard) or `bimodal`, chosen per run.
- **Randomness.** Real signing uses the OS CSPRNG (`secrets`). The demo, tests and benchmarks pass a seeded `random.Random` so their output is reproducible; use `--seed` to change it.

## Running the demo

```bash
python demo.py                  # Fig. 10 parameters, 4096-key ring (k = 2), about 5 s
python demo.py --mode bimodal   # bimodal rejection sampling instead of Rej1
python demo.py --toy            # insecure toy keys, faster
```

Other options: `--signer <i>` (signer's position in the ring), `--k <even k ≥ 2>`, `--seed <n>`; `python demo.py --help` lists them. Bad values (an odd `k`, a signer outside the ring) are rejected with a usage message.

The demo shows (1) a valid signature accepted, (2) the same signature rejected after one public key in the ring is changed, and after the message is changed, with the verifier's reason, and (3) the signer's index traced through the tensor decomposition and each recursion level. Expected output (abridged, `python demo.py` with the default seed; timings vary by machine):

```
parameters : Fig. 10 ring signature, k = 2, rejection = rej1
generated  : 4096 key pairs in 0.54 s; signer is ring member 1459
...
1. Sign and verify
sign       : 0.27 s, 1 attempt(s) (expected about 4.1; restarts: JL 0, z1 0, z2 0)
size       : 21935 bytes as encoded by serialize.py (measured, not the Fig. 10 estimate)
verify     : 0.17 s  ->  ACCEPT
from bytes : decode(encode(sig)) verifies too  ->  ACCEPT
...
2. Same signature, modified public statement
change     : coefficient 0 of public key #2824 (not the signer's), 55711 -> 55712
verify     : REJECT
    reason: Fiat-Shamir check: the challenge c does not match this statement and transcript ...
also       : original ring, message with one extra '!'  ->  REJECT
...
3. Tensor decomposition through the recursion (prover's view)
index 1459 = digits (2 6 51) in radix (8, 8, 64)
level 1 (Fig. 9): T_0 is 64 x 4096  ->  T_1 is 64 x 512
  check: <v_1, w_1,i> == <phi_i, A s>  (= <phi_i, T e_i>) for all 8 i: True
level 2 (Fig. 8): T_1 is 64 x 512  ->  T_2 is 64 x 64
  check: <v_2, w_2,i> == <phi_1,i, w_1> for all 8 i: True
...
Result: all three demo checks behaved as expected
```

The demo exits with status 0 when all three checks behave as expected.

## Running the tests

```bash
pytest                    # 208 tests, about 1 minute on the machine above
pytest -m "not slow"      # skips the statistical and full-size tests
pytest tests/test_statement_binding.py -v   # one file, verbose
```

Expected result: `208 passed`.

### What the tests check

- **Foundation:** ring arithmetic against a naive reference, `σ` and Lemma 2.2, samplers (moments, ranges, acceptance rate ≈ 1/M), transcript separation, parameter validation
- **Recursion:** the tensor decomposition, the fold identity, and the Lemma 4.1 linking checks (including that a sign flip is harmless only for even `k`, Lemma C.1)
- **Base case:** honest proofs accepted; for each condition of `R_k`, a prover with a witness that breaks it is rejected; the relation encoding agrees with an independent opening checker; any tampered proof field is rejected
- **Ring signature:** sign/verify for any ring member, bad inputs rejected without crashing, encoding round-trips; the same signature fails after changing a key, the ring order, the message or the public parameters (`test_statement_binding.py`)
- **Demo:** runs end to end and reports success; bad command-line arguments give a usage error, not a traceback

These are functional tests. They show the code follows the paper's algorithms and catches the cheating cases we wrote; they are not a security proof. Security rests on MSIS, MLWE and the paper's Extended-MLWE assumption.

## Reproducing the results

```bash
python bench/size.py           # sizes vs Fig. 10, k = 2 and 4 (about 50 s, about 650 MB RAM)
python bench/timing.py         # sign/verify times and attempts, Rej1 vs bimodal (about 45 s)
python bench/naive_verifier.py # opening-only checkers vs Pi_k on cheating witnesses (about 45 s)
```

Each script prints a summary and writes CSV files to `bench/results/` (overwriting the committed ones):

| Script | Output files |
|---|---|
| `size.py` | `size.csv` (measured vs estimated size per k and mode), `size_breakdown.csv` (bytes per signature component) |
| `timing.py` | `timing.csv` (one row per signature), `timing_summary.csv` (means, attempts, restarts) |
| `naive_verifier.py` | `naive_verifier.csv` (accept counts and time per check for each verifier and cheating case) |

All three use fixed seeds, so the sizes and accept counts come out the same on every run; timings depend on the machine. `--help` on each script lists its options. The committed CSVs are the ones the numbers below come from.

## Results

Measured on the machine in [requirements](#requirements). "Measured" is `len(serialize.encode(signature))`, the mean over 3 signatures; Fig. 10 numbers are the paper's estimates.

| N | k | Fig. 10 estimate | Measured (Rej1) | Measured (bimodal) |
|---|---|---|---|---|
| 2^6  | 0 | 13.1 KB | not implemented | not implemented |
| 2^12 | 2 | 14.1 KB | 22.0 KB | 19.5 KB |
| 2^18 | 4 | 15.1 KB | 23.2 KB | 20.4 KB |
| 2^24 | 6 | 16.2 KB | not run | not run |

- Keys: 128 B public, 96 B secret (App. C.2 says the same).
- Size growth from N = 2^12 to 2^18 (64× more keys): +1.2 KB (Rej1) / +0.9 KB (bimodal), against +1.0 KB in Fig. 10, so the logarithmic growth matches. The fixed gap of about 5–8 KB comes mostly from not applying LNP22 Sec. 6.1 compression (every mod-q value, including `t_A` and the garbage commitments, is sent at 32 bits) and from the simplified JL step (see [deviations](#deviations-from-the-paper)); `size_breakdown.csv` shows where the bytes go.
- Bimodal sampling makes signatures about 2.5–2.8 KB (11–12%) smaller than Rej1 at the same ring size.
- N = 4096, 20 signatures per mode (`timing_summary.csv`): sign 1.23 s mean / 0.95 s median (Rej1) and 0.98 s / 0.72 s (bimodal); verify 0.46 s / 0.33 s; 3.15 / 3.55 attempts per signature on average, below the expected bound of about 4.1. The machine was not idle (other applications at roughly 50% CPU), and repeated runs varied by up to 2×: an earlier run gave sign 1.00 / 0.51 s and verify 0.33 / 0.18 s, and the single demo run above signed in 0.27 s. Attempt counts and sizes are seeded and do not change between runs.
- Gaussian width of the masks: `std1` = 155,495 (Rej1) vs 4,665 (bimodal), about 33× narrower, which is where the size saving comes from.
- `bench/naive_verifier.py`, 5 kinds of cheating witness × 20 trials: a checker that only opens the commitments accepts all 100; `Π_k` accepts 0, and still accepts all 20 honest proofs. A check with `Π_k` takes about 40–110 ms.
- Challenge space (Sec. 2.5): 99.15% of 2000 random candidates pass the η = 140 check (the paper says at least 99%).

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
docs/             the original implementation plan (historical)
```

Data flow for one signature: `ringsig.sign` builds the statement (parameters, `A`, the ring, the message) → `oom.prove` commits to the witness with `abdlop`, then for each level squeezes challenges from `fiat_shamir` and folds the ring with `fold` / `recursion` → `base_case.prove` proves the remaining relation `R_k` → `serialize.encode` turns the proof into bytes. `ringsig.verify` replays the same transcript and checks each step.

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

## Security notes and limitations

- **Primitives.** Hashing and matrix expansion use SHAKE-256 from Python's `hashlib`; secret randomness comes from `secrets`. We did not write our own hash or RNG. The lattice arithmetic, samplers and proof system are our own code, since no maintained Python library implements BLOOM or LNP22.
- **Input validation.** `ringsig.verify` returns a reject with a reason (it does not raise) on malformed signatures, wrong-length rings, out-of-range keys or tampered fields; the tests cover these cases. `demo.py` rejects bad command-line arguments with a usage message.
- **Not constant-time.** Sampling, rejection and arithmetic branch on secret values, so the code is not safe against timing or other side-channel attacks. It is a research prototype and should not protect real keys.
- **No concrete security level** for the derived proof parameters (see deviations). The paper's own parameters are used unchanged.
- **No secrets in the repo.** There are no passwords, API keys or personal data. Keys created by the demo, tests and benchmarks are generated at run time from fixed seeds and never written to disk.

## Third-party code, libraries and AI use

Libraries: **numpy** (arrays, convolution, int64 arithmetic with 16-bit limb splitting so products stay exact for q < 2^32), **pytest**, and the Python standard library (`hashlib` for SHAKE-256, `secrets`). We chose Python with numpy over C/C++ so every team member could read and explain all of the code, and because with `d = 64` it is fast enough to sign with N = 2^18. We did not use LaZer (IBM's LNP22 library): its proof system has standard rejection sampling built in, so testing bimodal sampling would have meant changing its C internals.

No code is copied from other repositories. The TU Graz "lattice-zk" notebook (F. Lugstein, 2023, https://extgit.iaik.tugraz.at/public-shared/lattice-zk) was read as a reference for how LNP22's sub-protocols fit together. It has no licence, so nothing was copied; `base_case.py` is our own non-interactive implementation for `q = q1·q2`. The paper's authors released no code.

AI assistance:

- The foundation layer (`params.py`, `ring.py`, `sampling.py`, `transcript.py` and their tests) was drafted with AI assistance, then reviewed and tested by the team.
- Everything added after it was generated with Claude (Anthropic) from an implementation design the team agreed on: the remaining modules in `src/bloom/`, the new tests, `demo.py` and `bench/`. This includes the internal rewrite of `ring.py` (same API, int64 storage) and the array samplers added to `sampling.py`. The team reviews this code and is responsible for it.

The team's own work: the implementation design (module structure, how each step of Figs. 8 and 9 and App. C maps to code), resolving the gaps and inconsistencies in the paper listed under [deviations](#deviations-from-the-paper), deciding what the tests must show (honest acceptance, one negative test per condition of `R_k`, statement binding), the evaluation design and benchmarks, and reviewing and testing all generated code. Bugs we found and fixed this way include a Fiat-Shamir transcript that skipped the ring and the message, so modified statements were accepted (now covered by `tests/test_statement_binding.py`), and, after teaching-team feedback, an opening-based check of the base case, which needs the witness (so it reveals the signer) and on its own accepts cheating witnesses. It was replaced by the zero-knowledge `Π_k` and is kept only as a debugging baseline (`bench/naive_verifier.py`).

## Team

| Name | Student ID | Email |
|---|---|---|
| Tushar Jain | 35896493 | tjai0005@student.monash.edu |
| Shanshan Li | 36404179 | slii0373@student.monash.edu |
| Shaopei Fu | 36858579 | sfuu0033@student.monash.edu |
| Mitchell Padula | 33114579 | mpad0004@student.monash.edu |

Branch and commit rules are in [CONTRIBUTING.md](CONTRIBUTING.md). Everyone is expected to understand the whole codebase, since the Q&A can ask any of us about any part.

## Packaging the code for submission

The code zip should hold only tracked files (no `.venv/`, caches, report or slides). From the repository root:

```bash
git archive --format=zip -o FIT5163-A2-35896493-36404179-36858579-33114579-Code.zip HEAD
```

## Reference

V. Lyubashevsky and N. K. Nguyen, *BLOOM: Bimodal Lattice One-Out-of-Many Proofs and Applications*.
