"""Running times, and the Rej1 vs bimodal rejection-sampling experiment (Section 3).

    python bench/timing.py                  k = 2 (N = 4096), 20 signatures per mode
    python bench/timing.py --runs 50 --k 4

For each rejection mode it records, per signature: signing time, number of attempts and why
each failed attempt restarted, verification time, and the encoded sizes of z1 and z2.

What to compare (the paper reports sizes only, so the timings have no paper baseline):
  * attempts per signature vs the expected value M1 * M2 * M3 from oom_params (the M's are
    upper bounds on the repetition rate of each rejection step, Lemma 2.8);
  * std1 and the size of z1/z2: bimodal reaches the same M with a ~33x narrower Gaussian.

The paper's own repetition rate is "about 3" (App. C.2); ours is set by our gamma choices.

Writes bench/results/timing.csv (one row per signature) and bench/results/timing_summary.csv.
"""

from __future__ import annotations

import argparse
import csv
import random
import statistics
import time
from pathlib import Path

from bloom import oom_params, ringsig, serialize
from bloom.oom import ProveTrace

RESULTS = Path(__file__).resolve().parent / "results"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--modes", nargs="+", default=["rej1", "bimodal"])
    ap.add_argument("--toy", action="store_true", help="toy keys (m = 2), insecure")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args(argv)
    rng = random.Random(args.seed)
    RESULTS.mkdir(exist_ok=True)

    per_sig, summary = [], []
    for mode in args.modes:
        pp = (oom_params.toy if args.toy else oom_params.paper_scale)(args.k, mode)
        params = ringsig.setup(pp, rng.randbytes(32))
        _ = params.A_rot, params.ck                     # expand public matrices before timing
        t0 = time.perf_counter()
        keys = ringsig.keygen_many(params, pp.ps.ring_size, rng)
        t_keygen = time.perf_counter() - t0
        ring = [kp.pk for kp in keys]
        for run in range(args.runs):
            msg = b"timing %d" % run
            signer = rng.randrange(len(keys))
            trace = ProveTrace()
            t0 = time.perf_counter()
            sig = ringsig.sign(params, keys[signer], ring, msg, rng, trace)
            t_sign = time.perf_counter() - t0
            t0 = time.perf_counter()
            ok = ringsig.verify(params, ring, msg, sig).ok
            t_verify = time.perf_counter() - t0
            assert ok
            br = serialize.size_breakdown(sig, pp)
            st = trace.stats
            per_sig.append({
                "mode": mode, "k": args.k, "N": pp.ps.ring_size, "run": run,
                "sign_s": round(t_sign, 4), "verify_s": round(t_verify, 4), "attempts": trace.attempts,
                "restart_jl_bound": st.jl_bound, "restart_z3": st.rej3, "restart_z1": st.rej1,
                "restart_z2": st.rej2, "z1_bytes": br["z1"], "z2_bytes": br["z2"],
                "total_bytes": br["total (encoded)"],
            })
        mine = [r for r in per_sig if r["mode"] == mode]
        att = [r["attempts"] for r in mine]
        summary.append({
            "mode": mode, "k": args.k, "N": pp.ps.ring_size, "runs": args.runs,
            "keygen_all_s": round(t_keygen, 3),
            "sign_mean_s": round(statistics.mean(r["sign_s"] for r in mine), 3),
            "sign_median_s": round(statistics.median(r["sign_s"] for r in mine), 3),
            "verify_mean_s": round(statistics.mean(r["verify_s"] for r in mine), 3),
            "attempts_mean": round(statistics.mean(att), 2),
            "attempts_expected_bound": round(pp.expected_repetitions, 2),
            "restarts_z1_total": sum(r["restart_z1"] for r in mine),
            "restarts_z2_total": sum(r["restart_z2"] for r in mine),
            "restarts_jl_total": sum(r["restart_z3"] + r["restart_jl_bound"] for r in mine),
            "std1": round(pp.std1), "std2": round(pp.std2),
            "z1_bytes_mean": round(statistics.mean(r["z1_bytes"] for r in mine)),
            "z2_bytes_mean": round(statistics.mean(r["z2_bytes"] for r in mine)),
            "total_bytes_mean": round(statistics.mean(r["total_bytes"] for r in mine)),
        })
        s = summary[-1]
        print(f"{mode:8s} N={s['N']}: keygen {s['keygen_all_s']} s for all keys | sign {s['sign_mean_s']} s "
              f"(median {s['sign_median_s']}) | verify {s['verify_mean_s']} s | attempts "
              f"{s['attempts_mean']} (bound {s['attempts_expected_bound']}) | std1 {s['std1']} | "
              f"z1 {s['z1_bytes_mean']} B, z2 {s['z2_bytes_mean']} B, total {s['total_bytes_mean']} B")

    for name, rows in (("timing.csv", per_sig), ("timing_summary.csv", summary)):
        with open(RESULTS / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    print(f"\nwrote {RESULTS / 'timing.csv'} and {RESULTS / 'timing_summary.csv'}")


if __name__ == "__main__":
    main()
