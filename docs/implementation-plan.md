# Implementation plan

Working doc. It covers how we turn the paper into code, who does what, and what we cut if we run out of time. Edit it as things change.

## Before anything else: what we're actually up against

A few things that came out of reading the paper closely:

1. **There is no reference implementation.** The authors never wrote code. The signature sizes in Fig. 1 and Fig. 10 are *calculated* using the size formula from LNP22 Section 6.1, not measured. So "reproducing the results" for us means building it, serialising real signatures, and checking whether the measured size matches their estimate. That's a genuinely useful thing to report, whichever way it goes.
2. **No timing claims either.** The paper only talks about sizes. Any sign/verify timings we produce are new data, not a comparison.
3. **BLOOM sits on top of LNP22.** It doesn't define its own base proof system. It uses the LNP22 framework (Lyubashevsky, Nguyen, Plançon, Crypto 2022) as the base case and adds two things: the recursion that keeps the proof logarithmic in N, and bimodal rejection sampling. We'll have to implement the parts of LNP22 we need ourselves.
4. **We only need part of LNP22.** Section 4 and Appendix C reduce the whole ring signature to:
   - "Type-2": proving the constant coefficient of some quadratic expression in the committed values is zero
   - "Type-3": proving some committed coefficients are binary (which is itself a quadratic relation, `x(x-1) = 0`)

   We don't need the Euclidean norm proofs (Type-4) or the approximate shortness / JL projection (Type-5) for the main relation, which is a big saving. (See the open question about `‖u‖ = 1` below though.)
5. **Time.** It's Oct 18. The presentation is Nov 2/3 and the code is due Nov 6. That's about two weeks to a working demo. The plan below is built around that and has explicit fallbacks.

## Language and libraries

**Recommendation: Python 3.11+ with numpy.**

- With `d = 64` the polynomials are tiny, so Python is fast enough for a demo and for benchmarks up to N = 2^12 or so.
- Everyone can read it, which matters because any of us can be asked about any part of the code in the Q&A.
- `hashlib.shake_128/256` from the stdlib gives us the Fiat-Shamir hashing and the expansion of public matrices.
- `pytest` for tests, `matplotlib` for plots.

Alternatives we considered:

- **C / C++:** faster, but we'd spend the two weeks fighting memory bugs instead of getting the maths right.
- **LaZer** (IBM's lattice ZK library, which implements LNP22): worth a look, but it targets Linux, and its proof system uses standard rejection sampling internally. Swapping in bimodal sampling (the thing we're supposed to evaluate) would mean modifying its C internals. We'd also have to clearly separate their code from ours. Probably not worth it on this timeline, but someone should spend an hour checking whether it can at least serve as a reference to test against.

Arithmetic note: `q = 65437 · 65629 ≈ 2^32`, so a product of two coefficients overflows int64. Start with numpy `dtype=object` (Python ints) so correctness is easy. If it's too slow, switch to doing everything mod `q1` and mod `q2` separately in int64 (CRT), since both primes are under 2^17.

## Parameters (from Fig. 10 / Appendix C)

| Symbol | Value | Meaning |
|---|---|---|
| `d` | 64 | ring degree, `R_q = Z_q[X]/(X^64 + 1)`, and the length of the last unit vector `v_{k+1}` |
| `p` | 65437 | ring signature modulus (keys live mod p) |
| `q` | 65437 · 65629 | proof system modulus |
| `n, m` | 1, 12 | A is 1×12 over `R_p` |
| `𝔡` (`base` in code) | 8 | length of the unit vectors `v_1 … v_k` |
| `𝔩` (`ell` in code) | 8 | challenges per recursion step. The paper says 16, but needs `𝔩·𝔡 ≤ d` (see README deviations) |
| `k` | 0, 2, 4, 6 | recursion depth, must be **even** for the free bimodal trick |
| `N` | `d · 𝔡^k` | ring size: 2^6, 2^12, 2^18, 2^24 |
| challenge space | `d=64, κ=8, η=140` | Fig. 4 |

Secret key: `s ∈ {-1, 1}^{m·d}` (96 bytes). Public key: `A·s mod p` (128 bytes).

## Architecture

Bottom-up, each layer only uses the ones below it:

```
src/bloom/
├── ring.py          # Poly / PolyVec / PolyMat over Z_q[X]/(X^d+1), σ_{-1} automorphism, centred reps, norms
├── sampling.py      # discrete Gaussian, uniform {-1,1}, challenge space C, SHAKE-based expansion
├── params.py        # parameter sets (toy set for tests + the Fig. 10 sets)
├── rejection.py     # Rej1 (standard) and Rej2 (bimodal), Fig. 3
├── abdlop.py        # ABDLOP commitment: commit / open / relaxed-open check (Sec 2.6)
├── transcript.py    # Fiat-Shamir transcript: absorb bytes, squeeze challenges
├── proofs/
│   ├── linear.py    # Fig. 6: linear relations, bimodal on z2
│   ├── quadratic.py # Fig. 7: sign-invariant quadratic relation, bimodal on z1 and z2
│   └── lnp.py       # "engine": many constant-coefficient quadratic relations + binary
│                    #   coefficients, aggregated into one proof (the bits of LNP22 we need)
├── oom.py           # one-out-of-many: tensor decomposition, T_j / w_j recursion (Sec 4, Fig. 8/9)
├── ringsig.py       # KeyGen / Sign / Verify (Appendix C.2)
└── serialize.py     # byte encoding, so we can measure real signature sizes
tests/               # one test file per module, plus negative/security tests
bench/               # size + timing scripts, output CSVs and plots
demo.py              # what we run live in the presentation
```

### How the pieces connect (the actual algorithm)

**Ring signature → one-out-of-many.** The ring is the matrix `T = [pk_1 | ... | pk_N]`. The signer for index `i` proves knowledge of `s ∈ {-1,1}^{md}` and a unit vector `v` with `T·v = A·s`. The message goes into the Fiat-Shamir hash.

**Unit vector → tensor product.** Write the index `i` in mixed radix `(𝔡, ..., 𝔡, d)`. That gives `v = v_1 ⊗ v_2 ⊗ ... ⊗ v_k ⊗ v_{k+1}`, where `v_1 … v_k ∈ {0,1}^𝔡` and `v_{k+1} ∈ {0,1}^d` are one-hot. The short ones get zero-padded to length d. Each one becomes a monomial `u_j = X^{a_j}`. In the signed version from App. C.1 they're `±X^{a_j}`.

**What gets committed (ABDLOP).**
- Ajtai part `s1 = (s, u_1, ..., u_k, v_{k+1})`: small coefficients, and its size doesn't affect the commitment size
- BDLOP part `m = (w_1, ..., w_k)`: the intermediate vectors from the recursion, one polynomial each

**Recursion step j** (Fig. 8). Squeeze `𝔩` challenge vectors `γ_{j,i}` from the transcript. Split `T_j` into `𝔡` column blocks and build `T_{j+1}` from `γ_{j,i}^T · T_{j,block}`. Compute `w_{j+1} = T_{j+1}(v_{j+2} ⊗ ... ⊗ v_{k+1})`, packing the `𝔩` vectors of length `𝔡` into one polynomial (since `𝔩·𝔡 ≤ d`). Commit to it with `t_{j+1} = b_{j+1}^T s2 + w_{j+1}`. This is all plain linear algebra on public matrices plus one new commitment per level.

**Base case.** Once the tensor product is gone, everything left is a list of statements of these kinds:
- "the constant coefficient of `X^{(i-1)𝔡} · u_j · σ(w_j) − (γ_{j,i})^T w_{j-1}` is zero" (Lemma 4.1 trick, using `σ = σ_{-1}`)
- "`s` is in {-1,1}": prove `(s+1)/2` is binary
- "`u_j` is a signed monomial" (`‖u_j‖ = 1`, see open questions)
- "`T_k v_{k+1} = w_k`"

All of these go to `proofs/lnp.py`, which aggregates them with random challenges into a single relation and proves it in one shot, using the Fig. 7 structure (one garbage commitment `t`, one masked value `v`, bimodal `z1, z2`).

## Splitting the work

Everyone owns a layer but reviews at least one other person's PRs (see CONTRIBUTING.md). Suggested split, so swap freely:

| Person | Owns | Then helps with |
|---|---|---|
| A | `ring.py`, `sampling.py`, `params.py`, `transcript.py`, `serialize.py` | benchmarks + plots |
| B | `rejection.py`, `abdlop.py`, `proofs/linear.py`, `proofs/quadratic.py` | bimodal vs standard experiment |
| C | `proofs/lnp.py` (the hardest bit) | base case of `oom.py` |
| D | `oom.py` recursion, `ringsig.py`, `demo.py` | security / negative tests |

A's layer blocks everyone, so A should get a minimal `ring.py` (add, mul, σ, centred mod) pushed in the first day or two, even if it's slow.

## Timeline

| Dates | Goal |
|---|---|
| **Oct 19–21** | `ring`, `sampling`, `params`, `transcript` merged with tests. ABDLOP commit/open working. Decide the open questions below. |
| **Oct 22–25** | Fig. 6 and Fig. 7 proofs working with both Rej1 and Rej2. `lnp.py` proves "constant coefficient = 0" and binary statements. Recursion maths for `oom.py` worked out on paper and tested with plain (unproven) linear algebra. |
| **Oct 26–29** | Full one-out-of-many proof for k = 0 and k = 2. Ring signature sign/verify. Serialisation. |
| **Oct 30–Nov 1** | Benchmarks, negative tests, demo script, backup recording of the demo. **Code freeze Nov 1.** |
| Nov 2–3 | Presentation. |
| Nov 3–6 | Report. Fill in README setup/run/reproduce sections. Final tidy. |

## Fallback tiers

If we're behind, we drop down a tier and write up why in the report (the spec explicitly allows a reduced scope if it's justified).

- **Tier 1 (must have):** ABDLOP + the Fig. 6 / Fig. 7 proofs with standard vs bimodal rejection sampling, plus an experiment measuring rejection rate, `‖z‖` and proof size for both. This directly tests the paper's second contribution.
- **Tier 2:** one-out-of-many for `k = 0` (no recursion, N = 8) as a working ring signature, plus Tier 1.
- **Tier 3 (full target):** recursion for `k = 2` (N = 2^12), with sizes measured against Fig. 10. `k = 4, 6` are stretch goals, mostly limited by runtime and memory for building `T`.

## Evaluation plan

Things we can measure and put in the report:

1. **Correctness:** honest signatures verify across many random keys, indices and messages, at every supported k.
2. **Soundness / unforgeability (negative tests):** each of these must be rejected:
   - signing with a key that isn't in the ring
   - a tampered message, a tampered `z1`/`z2`, or tampered commitments
   - a signature replayed against a different ring
   - a wrong index
   - a "secret key" that isn't in {-1,1}
3. **Anonymity (statistical):** sign many times from different indices and check the distribution of the `z` vectors doesn't depend on the index (compare means and variances, and run a KS test). This isn't a proof, but it's evidence the rejection sampling is doing its job.
4. **Bimodal vs standard:** same proofs with Rej1 vs Rej2. Measure the std dev of `z`, the acceptance rate (the paper says the repetition rate is about 3) and the proof size. The paper claims roughly a 17x smaller std dev.
5. **Sizes vs Fig. 10:** our serialised signature size at N = 2^6, 2^12 (and further if we get there) vs 13.1 / 14.1 / 15.1 / 16.2 KB. Explain the gap: we don't do every compression trick LNP22 does, and their numbers are estimates.
6. **Public key size:** should be exactly 128 bytes. This one is easy to confirm.
7. **Timings:** keygen, sign and verify vs N, averaged over repeated runs, with hardware stated.

## Open questions to settle early (and document in the README)

- **mod p vs mod q.** Keys satisfy `A·s = pk mod p`, but the proof works mod `q = p·q2`. The paper says this works "if p divides q" and doesn't spell out how. Options: (a) prove `T·v − A·s = p·e` and commit to `e`, or (b) for our implementation, set the ring-sig modulus to `q` and accept bigger keys. Pick one, then document it.
- **`‖u_j‖ = 1` soundness.** The constant coefficient of `σ(u)·u` equalling 1 mod q only implies `‖u‖ = 1` over the integers if we also know `u` is short (no wrap-around). LNP22 handles this with approximate shortness (JL projection), which we wanted to avoid. A simpler alternative is to prove `u_j` is binary and its coefficients sum to 1. That isn't sign-invariant, though, so we'd lose the "free" bimodal trick on `s1` and need either Rej1 on `z1` or the commit-to-`b` approach from Sec 3.2. Either way it's a documented deviation.
- **Fiat-Shamir details.** What exactly gets hashed and in what order: public params, the ring, the message, all commitments up to that point. Write it down once in `transcript.py` and stick to it.
- **Discrete Gaussian sampler.** A simple CDT or rejection-based sampler is fine for a prototype. It won't be constant time, so say so in the limitations section.

## Things to be honest about in the report

- not constant time, so there's no side-channel resistance
- parameter security is taken from the paper; we don't re-estimate it with the lattice estimator
- any part of LNP22 we simplified or skipped
- measured sizes vs their calculated sizes, and why they differ
- all external code, libraries and AI-tool assistance listed in the README provenance section
