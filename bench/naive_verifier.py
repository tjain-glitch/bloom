"""Opening-based checkers vs the zero-knowledge base case Pi_k.

    python bench/naive_verifier.py              toy parameters, 20 trials per case
    python bench/naive_verifier.py --paper --trials 5

Jie's email: an opening-based checker is acceptable only as a debugging tool; zero-knowledge
is what makes it BLOOM. This script puts three verifiers side by side on the same inputs:

    A  openings only        OpeningChecker(openings_only=True): do t_A, t_1..t_k open to the
                            claimed witness? Needs the witness (so it gives away the signer).
    B  openings + R_k       OpeningChecker: also checks every row of R_k in the clear.
                            Needs the witness.
    C  Pi_k (ours)          base_case.verify: sees only the proof. Zero-knowledge.

Inputs: the honest witness, and cheating witnesses that each break one row of R_k (the
cheater commits to the bad witness, so its openings are consistent). For C the cheater runs
the real prover code on the bad witness (with the honest-witness self-check switched off).

Reported: how many of `trials` each verifier accepts, plus mean time per check. These are
functional results: they show the checks are wired up, not a soundness proof.

Writes bench/results/naive_verifier.csv.
"""

from __future__ import annotations

import argparse
import csv
import random
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from bloom import abdlop, base_case, oom, oom_params
from bloom.base_case import OpeningChecker
from bloom.fiat_shamir import OomTranscript
from bloom.ring import matmul_mod, reduce, rot_batch
from bloom.sampling import sign_array, uniform_array

RESULTS = Path(__file__).resolve().parent / "results"


def instance(pp, rng):
    ps = pp.ps
    d, p, q, q2 = ps.d, ps.p, ps.q, ps.q2
    A = uniform_array((ps.n, ps.m, d), p, rng)
    A_rot = np.block([[rot_batch(A[i, j], p) for j in range(ps.m)] for i in range(ps.n)])
    s = sign_array((ps.m, d), rng)
    T = uniform_array((ps.n * d, ps.ring_size), p, rng)
    index = rng.randrange(ps.ring_size)
    T[:, index] = matmul_mod(A_rot, (s.reshape(-1) % p)[:, None], p)[:, 0]
    stmt = oom.Statement(T=(T * q2) % q, A_rot=(A_rot * q2) % q)
    ck = abdlop.keygen(pp, rng.randbytes(32))
    tr = OomTranscript(pp)
    tr.absorb_statement(ck.seed, A, T, b"naive verifier bench")
    return stmt, oom.Witness(s=s, index=index), ck, tr


# Each returns a modified BaseWitness (see tests/test_base_case.py for the same list).
def honest(pp, bw, rng):
    return bw


def forged_key(pp, bw, rng):
    s1 = bw.s1.copy()
    s1[: pp.ps.m] = sign_array((pp.ps.m, pp.d), rng)       # a valid-looking key not in the ring
    return replace(bw, s1=s1)


def s_not_sign(pp, bw, rng):
    s1 = bw.s1.copy()
    s1[0, rng.randrange(pp.d)] = 3
    return replace(bw, s1=s1)


def u_outside_support(pp, bw, rng):
    s1 = bw.s1.copy()
    s1[pp.ps.m] = 0
    s1[pp.ps.m, pp.ps.base + rng.randrange(pp.d - pp.ps.base)] = 1
    return replace(bw, s1=s1)


def u_not_monomial(pp, bw, rng):
    s1 = bw.s1.copy()
    s1[pp.ps.m] *= 2
    return replace(bw, s1=s1)


def w_wrong(pp, bw, rng):
    w = bw.w.copy()
    j, r = rng.randrange(pp.k), rng.randrange(pp.d)
    w[j, r] = (w[j, r] + 1) % pp.q
    return replace(bw, w=w)


CASES = [honest, forged_key, s_not_sign, u_outside_support, u_not_monomial, w_wrong]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--paper", action="store_true", help="Fig. 10 parameters instead of toy")
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args(argv)
    rng = random.Random(args.seed)
    pp = oom_params.paper_scale(2) if args.paper else oom_params.toy(2)
    stmt, wit, ck, tr0 = instance(pp, rng)
    checker = OpeningChecker(pp, ck)
    RESULTS.mkdir(exist_ok=True)

    rows = []
    for case in CASES:
        acc = {"A": 0, "B": 0, "C": 0}
        secs = {"A": 0.0, "B": 0.0, "C": 0.0}
        no_proof = 0
        for _ in range(args.trials):
            tr = tr0.fork()
            _, st, bw = oom.reduce_prove(pp, ck, stmt, wit, tr, 1, rng)
            bw = case(pp, bw, rng)
            st = replace(st, tA=abdlop.commit_ajtai(ck, bw.s1, bw.s2),
                         t_w=abdlop.commit_rows(ck, ck.rows_w(pp), bw.s2, reduce(bw.w, pp.q)))
            t0 = time.perf_counter()
            acc["A"] += checker.check(st, bw, openings_only=True).ok
            secs["A"] += time.perf_counter() - t0
            t0 = time.perf_counter()
            acc["B"] += checker.check(st, bw).ok
            secs["B"] += time.perf_counter() - t0
            pf = None
            for _ in range(100):
                pf = base_case.prove(pp, ck, tr.fork(), st, bw, rng, check_witness=False)
                if pf is not None:
                    break
            if pf is None:
                no_proof += 1
                continue
            t0 = time.perf_counter()
            acc["C"] += base_case.verify(pp, ck, tr.fork(), st, pf).ok
            secs["C"] += time.perf_counter() - t0
        n = args.trials
        rows.append({
            "case": case.__name__, "trials": n,
            "A_openings_only_accepts": acc["A"], "B_openings_plus_Rk_accepts": acc["B"],
            "C_zk_pik_accepts": acc["C"], "C_no_proof_produced": no_proof,
            "A_ms": round(1000 * secs["A"] / n, 2), "B_ms": round(1000 * secs["B"] / n, 2),
            "C_ms": round(1000 * secs["C"] / max(1, n - no_proof), 2),
        })
        r = rows[-1]
        print(f"{case.__name__:18s} accepted out of {n}:  A {r['A_openings_only_accepts']:>3}   "
              f"B {r['B_openings_plus_Rk_accepts']:>3}   C {r['C_zk_pik_accepts']:>3}"
              f"   (time per check: A {r['A_ms']} ms, B {r['B_ms']} ms, C {r['C_ms']} ms)")

    print("\nA and B need the secret witness (and reveal the signer); C sees only the proof.")
    with open(RESULTS / "naive_verifier.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {RESULTS / 'naive_verifier.csv'}")


if __name__ == "__main__":
    main()
