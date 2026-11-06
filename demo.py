"""Presentation demo for the BLOOM ring signature.

    python demo.py                  Fig. 10 parameters, ring of N = 4096 keys (about 5 s)
    python demo.py --toy            insecure toy keys (m = 2), same ring size, faster
    python demo.py --mode bimodal   bimodal rejection sampling (Section 3) instead of Rej1

It shows the three things the teaching team asked for (consultation, 29 Oct):

    1. a valid signature is accepted;
    2. the same signature is rejected after one public key in the ring is changed, with the
       verifier's reason;
    3. the signer's index traced through the tensor decomposition and the recursion
       (Figs. 9 and 8), level by level.

The secret-witness parts of step 3 (the selectors v_j and the w blocks) are printed from the
prover's side for teaching purposes; a verifier never sees them.
"""

from __future__ import annotations

import argparse
import random
import sys
import time

import numpy as np

from bloom import oom, oom_params, ringsig, serialize
from bloom.tensor import index_to_digits, selectors

LINE = "=" * 78


def banner(title: str) -> None:
    print(f"\n{LINE}\n{title}\n{LINE}")


def verdict(res) -> str:
    return "ACCEPT" if res.ok else f"REJECT\n    reason: {res.reason}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--toy", action="store_true", help="toy keys (m = 2): faster, NOT secure")
    ap.add_argument("--mode", choices=["rej1", "bimodal"], default="rej1")
    ap.add_argument("--k", type=int, default=2, help="recursion depth (even); N = 64 * 8^k")
    ap.add_argument("--signer", type=int, default=None, help="signer's position in the ring")
    ap.add_argument("--seed", type=int, default=2026, help="seed for a reproducible run")
    args = ap.parse_args(argv)

    # Check the command line before doing any work, so bad input gives a usage message
    # instead of a traceback. k = 0 (no recursion) is not implemented, see the README.
    if args.k < 2 or args.k % 2:
        ap.error(f"--k must be an even number >= 2, got {args.k}")

    rng = random.Random(args.seed)
    pp = (oom_params.toy if args.toy else oom_params.paper_scale)(args.k, args.mode)
    ps = pp.ps
    N = ps.ring_size
    if args.signer is not None and not 0 <= args.signer < N:
        ap.error(f"--signer must be in [0, {N}) for k = {args.k}, got {args.signer}")

    banner("Setup")
    print(f"parameters : {'TOY (insecure, m = 2)' if args.toy else 'Fig. 10 ring signature'}, "
          f"k = {pp.k}, rejection = {pp.rejection}")
    print(f"ring       : R_q = Z_q[X]/(X^{ps.d}+1), q = {ps.q1} * {ps.q2} = {ps.q}, keys mod p = {ps.p}")
    print(f"keys       : sk in {{-1,1}}^(m*d) with m = {ps.m};  pk = A s mod p ({ps.n * ps.d} coefficients)")
    t0 = time.perf_counter()
    params = ringsig.setup(pp, rng.randbytes(32))
    keys = ringsig.keygen_many(params, N, rng)
    ring = [kp.pk for kp in keys]
    signer = rng.randrange(N) if args.signer is None else args.signer
    message = b"FIT5163 BLOOM demo: approve the budget"
    print(f"generated  : {N} key pairs in {time.perf_counter() - t0:.2f} s; signer is ring member {signer}")
    print(f"message    : {message.decode()!r}")

    # ------------------------------------------------------------------------------------
    banner("1. Sign and verify")
    trace = oom.ProveTrace()
    t0 = time.perf_counter()
    sig = ringsig.sign(params, keys[signer], ring, message, rng, trace)
    t_sign = time.perf_counter() - t0
    t0 = time.perf_counter()
    res = ringsig.verify(params, ring, message, sig)
    t_verify = time.perf_counter() - t0
    blob = serialize.encode(sig, pp)
    print(f"sign       : {t_sign:.2f} s, {trace.attempts} attempt(s) "
          f"(expected about {pp.expected_repetitions:.1f}; restarts: JL {trace.stats.jl_bound + trace.stats.rej3}, "
          f"z1 {trace.stats.rej1}, z2 {trace.stats.rej2})")
    print(f"size       : {len(blob)} bytes as encoded by serialize.py (measured, not the Fig. 10 estimate)")
    print(f"verify     : {t_verify:.2f} s  ->  {verdict(res)}")
    again = ringsig.verify(params, ring, message, serialize.decode(blob, pp))
    print(f"from bytes : decode(encode(sig)) verifies too  ->  {verdict(again)}")

    # ------------------------------------------------------------------------------------
    banner("2. Same signature, modified public statement")
    victim = (signer + N // 3) % N
    tampered = list(ring)
    tampered[victim] = tampered[victim].copy()
    tampered[victim][0] = (tampered[victim][0] + 1) % ps.p
    print(f"change     : coefficient 0 of public key #{victim} (not the signer's), "
          f"{ring[victim][0]} -> {tampered[victim][0]}")
    res2 = ringsig.verify(params, tampered, message, sig)
    print(f"verify     : {verdict(res2)}")
    # The verifier rebuilds everything from the statement, so one changed key affects every level.
    T_orig = ringsig._ring_matrix(params, ring)
    T_new = ringsig._ring_matrix(params, tampered)
    st_orig = oom.reduce_verify(pp, ringsig._statement(params, T_orig),
                                ringsig._transcript(params, T_orig, message), sig)
    st_new = oom.reduce_verify(pp, ringsig._statement(params, T_new),
                               ringsig._transcript(params, T_new, message), sig)
    print("why        : the ring is hashed into every challenge, so the verifier derives different")
    print(f"             phi's ({np.count_nonzero(st_orig.phis[0] != st_new.phis[0])} of {st_orig.phis[0].size} "
          f"entries of phi_1 change) and a different folded matrix T_{pp.k} "
          f"({np.count_nonzero(st_orig.T_k != st_new.T_k)} of {st_orig.T_k.size} entries change);")
    print("             the proof's responses no longer reproduce the challenge c.")
    res3 = ringsig.verify(params, ring, message + b"!", sig)
    print(f"also       : original ring, message with one extra '!'  ->  {verdict(res3)}")

    # ------------------------------------------------------------------------------------
    banner("3. Tensor decomposition through the recursion (prover's view)")
    print(trace.decomposition)
    digits = index_to_digits(signer, ps.d, ps.base, pp.k)
    print(f"check      : e_{signer} = v_1 (x) ... (x) v_{pp.k + 1}  "
          f"(positions {' , '.join(str(x) for x in digits)}); "
          f"selectors are {[len(v) for v in selectors(signer, ps.d, ps.base, pp.k)]} long")
    if pp.rejection == "bimodal":
        print("note       : bimodal mode signs the whole witness with a random b in {-1,1} (Lemma C.1)")
    q = pp.q
    T_lifted = ringsig._statement(params, ringsig._ring_matrix(params, ring)).T
    b = trace.sign
    prev_w = None
    for lv in trace.levels:
        j = lv.level
        digit = digits[j]
        print(f"\nlevel {j + 1} ({lv.figure}): T_{j} is {lv.T_shape[0]} x {lv.T_shape[1]}  ->  "
              f"T_{j + 1} is {lv.T_next_shape[0]} x {lv.T_next_shape[1]}")
        print(f"  {lv.selector}")
        print(f"  committed w_{j + 1} = {ps.ell} blocks of {ps.base} (one polynomial); v_{j + 1} = e_{digit} "
              f"picks entry [{digit}] of each block:")
        for i, block in enumerate(lv.w_blocks[:3]):
            cells = [f"[{x}]" if c == digit else f"{x}" for c, x in enumerate(block)]
            print(f"    w_{j + 1},{i}: " + " ".join(f"{c:>12s}" for c in cells))
        if len(lv.w_blocks) > 3:
            print(f"    ... ({len(lv.w_blocks) - 3} more blocks)")
        # Lemma 4.1: <v_{j+1}, w_{j+1,i}> must equal what the previous level committed to.
        picked = [(b * int(x)) % q for x in lv.w_blocks[:, digit]]
        if j == 0:
            target = [(b * int(np.dot(phi.astype(object), T_lifted[:, signer].astype(object)))) % q
                      for phi in lv.phis]
            what = "<phi_i, A s>  (= <phi_i, T e_i>)"
        else:
            target = [int(np.dot(phi.astype(object), prev_w.astype(object))) % q for phi in lv.phis]
            what = f"<phi_{j},i, w_{j}>"
        ok_link = picked == target
        print(f"  check: <v_{j + 1}, w_{j + 1},i> == {what} for all {ps.ell} i: {ok_link}")
        prev_w = lv.w_blocks.reshape(-1)
    print(f"\nbase case  : T_{pp.k} is {ps.d} x {ps.d}; Pi_k proves R_k in zero knowledge "
          f"({pp.num_zq_relations} constant-coefficient relations incl. 256 JL rows, collapsed into one "
          f"quadratic relation and proved with Fig. 7)")
    print(f"tried      : {trace.attempts} attempt(s), {trace.seconds:.2f} s")

    ok = res.ok and again.ok and not res2.ok and not res3.ok
    banner("Result: " + ("all three demo checks behaved as expected" if ok else "UNEXPECTED RESULT"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
