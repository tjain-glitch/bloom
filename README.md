# BLOOM: lattice one-out-of-many proofs (FIT5163 group project)

This is our group's repo for the FIT5163 programming assignment (Monash, Semester 2 2026). We were assigned the paper

> **BLOOM: Bimodal Lattice One-Out-of-Many Proofs and Applications**
> Vadim Lyubashevsky (IBM Research Europe) and Ngoc Khanh Nguyen (EPFL)

and the job is to implement it (or a big enough core chunk of it), test it properly, and then compare what we get against what the paper claims.

The repo is still early. Most of what's below is the plan, and we'll fill in the setup and run sections once there's actual code to run.

## What the paper is about (short version)

A one-out-of-many proof lets you prove you know a secret for *one* entry in a public list without revealing which entry. In lattice terms, you have a list of public keys `t_1 ... t_N` and you prove you know a short vector `s` with `A·s = t_i mod p` for some `i`, without leaking `i` or `s`.

If you put everyone's public key into the list and use the message as input to Fiat-Shamir, you get a **ring signature**. That's the main application the paper goes after (there's also a Falcon-based group signature in Appendix E).

The ideas we need to get our heads around, roughly in the order the paper uses them:

- **Writing the unit vector as a tensor product.** The index `i` becomes a one-hot vector `v` of length N. Committing to that directly would blow up the proof size, so they split `v` into a tensor product of smaller unit vectors (`v = v_1 ⊗ v_2 ⊗ ...`). Then they prove the relation recursively, so the proof stays logarithmic in N.
- **ABDLOP commitments and the LNP22 framework** are used to prove the base case. ABDLOP combines Ajtai and BDLOP commitments, and LNP22 is the proof system from Lyubashevsky, Nguyen and Plançon (Crypto 2022).
- **Bimodal Gaussian rejection sampling.** The paper shows the cheaper bimodal version (from BLISS / DDLL13) can be used throughout the protocol. In the one-out-of-many case this costs nothing extra and makes the proofs smaller.
- **Exact norm proofs.** Earlier schemes only proved a relaxed statement. Proving the exact norm lets them use a much smaller modulus, which is why the public keys come out tiny (around 128 bytes per user).

### Numbers from the paper we want to check

The ring signature parameters (Fig. 10) use `(p, n, m, d, δ) = (65437, 1, 12, 64, 8)`:

| Ring size N | k | Claimed signature size |
|---|---|---|
| 2^6  | 0 | 13.1 KB |
| 2^12 | 2 | 14.1 KB |
| 2^18 | 4 | 15.1 KB |
| 2^24 | 6 | 16.2 KB |

Fig. 1 also claims a 7x to 15x reduction in user public key size compared with earlier lattice ring signatures. Reproducing (or failing to reproduce) the signature sizes above is going to be a big part of our evaluation.

## Scope

The full paper covers ring signatures, group signatures and integer-relation proofs, which is a lot for one semester. Our current plan:

**Planned:**
- Polynomial ring arithmetic over `R_q = Z_q[X]/(X^d + 1)`, plus the NTT where it's worth the effort
- ABDLOP commitment
- Bimodal Gaussian rejection sampling
- The base-case proof (knowledge of a unit vector `v` and a short `s` with `T·v = A·s`)
- The recursive one-out-of-many proof
- A ring signature built on top of that (KeyGen / Sign / Verify) using Fiat-Shamir
- Benchmarks: proof size, sign and verify times as N grows, rejection rates

**Probably out of scope (we'll explain why in the report):**
- The Falcon-based group signature (Appendix E). It needs trapdoor sampling and verifiable encryption, which is a whole other project.
- Integer addition and multiplication proofs (Appendix D)
- Constant-time or side-channel hardened code. This is a research prototype.

This list will change as we go. If we cut something or simplify it, it goes in the "Deviations from the paper" section below.

## Contributing

Branch naming, commit messages and the full pull / push / PR workflow are in [CONTRIBUTING.md](CONTRIBUTING.md). Please read it before your first push.

## Repo layout

```
.
├── src/bloom/
│   ├── params.py      # parameter sets from Fig. 10 / Fig. 4, with validation
│   ├── ring.py        # R_q = Z_q[X]/(X^d+1): Poly, PolyVec, PolyMat, sigma automorphisms, rot()
│   ├── sampling.py    # CSPRNG / SHAKE-based randomness, discrete Gaussian, challenge space
│   └── transcript.py  # Fiat-Shamir transcript
├── tests/             # pytest suite, one file per module
├── bench/             # benchmark scripts and results (coming)
├── docs/              # notes and plans
├── pyproject.toml
└── requirements.txt
```

The commitment scheme, the proofs, the one-out-of-many recursion and the ring signature come next. They'll sit on top of these four modules.

## Setup

**Requirements:** Python 3.11 or newer (developed on 3.12.5), numpy, pytest. Nothing platform-specific, so Windows, macOS and Linux should all work. Developed and tested on Windows 11.

```bash
python -m venv .venv
```

Activate it (`source .venv/bin/activate` on macOS/Linux, `.venv\Scripts\activate` on Windows), then:

```bash
pip install -r requirements.txt
```

```bash
pip install -e .
```

The editable install (`-e`) is what lets `import bloom` work from the tests and scripts.

## Running it

Run the whole test suite:

```bash
pytest
```

Expected output is `47 passed` in a couple of seconds. To skip the statistical tests (Gaussian moments, challenge acceptance rate):

```bash
pytest -m "not slow"
```

The demo and benchmarks don't exist yet.

### What the current tests check

- **ring:** multiplication matches a naive textbook implementation; `X^d = -1`; ring axioms; `σ_i` is an automorphism; Lemma 2.2 (constant coeff of `σ_{-1}(u)ᵀv` equals `⟨u, v⟩`); `rot(A)·s = A·s`; byte encoding round-trips and rejects bad input
- **sampling:** XOF determinism; signs are ±1; Gaussian mean and std dev; challenges are `σ_{-1}`-invariant with `‖c‖∞ ≤ κ`; `‖c·r‖ ≤ η‖r‖`
- **transcript:** same inputs give same challenges; any change in data, order, label, type or framing changes them
- **params:** ring sizes match Fig. 10; key sizes match App. C.2 (128 B public, 96 B secret); invalid parameters are rejected

### Early results

- Challenge space (Sec. 2.5): **99.15%** of 2000 random candidates pass the η = 140 norm check. The paper says "at least 99%".
- Key sizes: 128 B public key and 96 B secret key, matching App. C.2.

## Reproducing the paper's results

TODO: the exact commands to regenerate our signature-size and timing tables, and where the raw output gets saved.

## Deviations from the paper

We'll keep this updated. Anything the paper leaves out where we had to make our own call (parameter choices, how a hash is instantiated, sampling details, etc.) goes here with a short reason.

- **𝔩 = 8 instead of 16.** Section 4 and App. C.2 state `(d, 𝔡, 𝔩) = (64, 8, 16)`, but also require `d = 𝔩·𝔡`, and 16·8 = 128 ≠ 64. The packing of the 𝔩 vectors `w_i ∈ Z_q^𝔡` into one degree-64 polynomial only works when `𝔩·𝔡 ≤ d`, so we use 𝔩 = 8. Soundness per recursion step is then about `q1^-8 ≈ 2^-128`, which still hits the paper's 128-bit target. `tests/test_params.py::test_papers_literal_ell_is_inconsistent` documents this.
- **Discrete Gaussian sampler.** The paper doesn't specify one. We use plain rejection sampling from a bounded uniform (tail cut at 14σ, so the cut-off mass is under 2^-140). It's simple to verify, but slow and not constant-time.
- **Fiat-Shamir.** The paper doesn't specify the hashing. We use SHAKE-256 with length-prefixed, type-tagged framing for every absorbed value. Each challenge is fed back into the state, so later challenges depend on earlier ones.
- **Public matrices** are expanded from a seed with SHAKE-256, one label per matrix (`A`, `A1`, `A2`, `B`, ...).
- **Randomness for secrets** comes from `secrets.SystemRandom` (the OS CSPRNG). Seeded `random.Random` is only used in tests.

## Third-party code, libraries and AI use

The assignment requires us to be upfront about anything we didn't write ourselves, so this section lists:

- external libraries and what we use them for
- any code adapted from elsewhere (with a link and licence)
- where AI tools were used and what we changed afterwards

Libraries:

- **numpy**: coefficient arrays and convolution for polynomial multiplication. We considered a pure Python implementation (slower, no gain) and an NTT library (none we found handles q ≈ 2^32 split as q1·q2 with these primes).
- **pytest**: test runner.
- **Python standard library**: `hashlib` (SHAKE-256 / SHA3-256 for Fiat-Shamir and seed expansion), `secrets` (CSPRNG).

No code from the paper's authors is used, since they didn't release any. No code is adapted from other repositories.

AI assistance:

- The initial version of the foundation layer (`params.py`, `ring.py`, `sampling.py`, `transcript.py` and their tests) was drafted with AI assistance. The team then reviewed, tested and is responsible for it. Any later changes and fixes will be noted here.

## Team

| Name | Student ID | Email |
|---|---|---|
| Tushar Jain | 35896493 | tjai0005@student.monash.edu |
| Shanshan Li | 36404179 | slii0373@student.monash.edu |
| Shaopei Fu | 36858579 | sfuu0033@student.monash.edu |
| Mitchell Padula | 33114579 | mpad0004@student.monash.edu |

Who's working on what is tracked through issues and commit history rather than written here. Everyone is expected to understand the whole codebase anyway, since the Q&A can ask any of us about any part of it.

## Key dates

- **Presentation + live demo:** Week 14 (Mon/Tue)
- **Report and code due:** Friday 6 Nov 2026, 23:55

## Reference

V. Lyubashevsky and N. K. Nguyen, *BLOOM: Bimodal Lattice One-Out-of-Many Proofs and Applications*.
