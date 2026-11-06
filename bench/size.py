"""Measured signature and key sizes, next to the paper's Fig. 10 estimates.

    python bench/size.py                     k = 2 and 4, Rej1 and bimodal, 3 signatures each
    python bench/size.py --ks 2 --runs 10

Two kinds of number, kept in separate columns on purpose (Jie's email, point 4):

  * measured  len(serialize.encode(signature)) for signatures this code actually produced;
  * estimate  the paper's Fig. 10 figure (LNP22 Sec. 6.1 counting). We also print our own
              "entropy" estimate, where the z's are counted at log2(sqrt(2 pi e) s) bits per
              coefficient as LNP22 does, instead of the fixed-width packing we transmit.
              It's an estimate of what entropy coding would give, not a measured size.

Only the signer's key is a real key pair; the other N - 1 ring members are uniform vectors
mod p. That is what a real public key looks like under MLWE, it does not change the proof or
its size, and it keeps memory manageable for k = 4 (N = 262144). k = 6 (N = 2^24) needs a
64 x 2^24 ring matrix (8.6 GB) and is not run; k = 0 is not implemented (see README).

Writes bench/results/size.csv and bench/results/size_breakdown.csv.
"""

from __future__ import annotations

import argparse
import csv
import random
import statistics
from pathlib import Path

from bloom import oom_params, ringsig, serialize
from bloom.sampling import uniform_array

FIG10_KB = {0: 13.1, 2: 14.1, 4: 15.1, 6: 16.2}         # paper, Fig. 10 (estimates)
RESULTS = Path(__file__).resolve().parent / "results"


def ring_with_signer(params, rng):
    ps = params.pp.ps
    key = ringsig.keygen(params, rng)
    T = uniform_array((ps.ring_size, ps.n * ps.d), ps.p, rng)
    index = rng.randrange(ps.ring_size)
    T[index] = key.pk
    return key, T, index


def key_sizes(params) -> tuple[int, int]:
    """Bytes for pk (coefficients mod p) and sk (one bit per +-1 coefficient)."""
    ps = params.pp.ps
    pk_bits = ps.n * ps.d * int(ps.p - 1).bit_length()
    sk_bits = ps.m * ps.d
    return (pk_bits + 7) // 8, (sk_bits + 7) // 8


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ks", type=int, nargs="+", default=[2, 4])
    ap.add_argument("--modes", nargs="+", default=["rej1", "bimodal"])
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args(argv)
    rng = random.Random(args.seed)
    RESULTS.mkdir(exist_ok=True)

    rows, breakdown = [], []
    for k in args.ks:
        for mode in args.modes:
            pp = oom_params.paper_scale(k, mode)
            params = ringsig.setup(pp, rng.randbytes(32))
            key, T, index = ring_with_signer(params, rng)
            sizes, est = [], []
            for run in range(args.runs):
                sig = ringsig.sign(params, key, T, b"size benchmark %d" % run, rng)
                assert ringsig.verify(params, T, b"size benchmark %d" % run, sig).ok
                br = serialize.size_breakdown(sig, pp)
                sizes.append(br["total (encoded)"])
                z_packed = br["z1"] + br["z2"] + br["z3"]
                est.append(br["total (encoded)"] - z_packed + br["z entropy estimate (z1+z2+z3)"])
                if run == 0:
                    breakdown += [{"k": k, "mode": mode, "component": name, "bytes": round(v)}
                                  for name, v in br.items()]
            pk, sk = key_sizes(params)
            rows.append({
                "k": k, "N": pp.ps.ring_size, "mode": mode, "runs": args.runs,
                "measured_bytes_mean": round(statistics.mean(sizes)),
                "measured_bytes_min": min(sizes), "measured_bytes_max": max(sizes),
                "measured_KB": round(statistics.mean(sizes) / 1000, 1),
                "our_entropy_estimate_KB": round(statistics.mean(est) / 1000, 1),
                "paper_fig10_estimate_KB": FIG10_KB.get(k, ""),
                "pk_bytes": pk, "sk_bytes": sk,
            })
            print(f"k={k} N={pp.ps.ring_size:>7} {mode:8s} measured {rows[-1]['measured_KB']:5.1f} KB "
                  f"(min {min(sizes)}, max {max(sizes)} B) | our entropy estimate "
                  f"{rows[-1]['our_entropy_estimate_KB']:5.1f} KB | Fig. 10 estimate {FIG10_KB.get(k, '-')} KB")

    with open(RESULTS / "size.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(RESULTS / "size_breakdown.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(breakdown[0]))
        w.writeheader()
        w.writerows(breakdown)

    print("\nkeys: pk = %d B, sk = %d B (paper: 128 B and 96 B)" % (rows[0]["pk_bytes"], rows[0]["sk_bytes"]))
    first = [b for b in breakdown if b["k"] == args.ks[0] and b["mode"] == args.modes[0]]
    print(f"\nbreakdown of one signature (k = {args.ks[0]}, {args.modes[0]}), measured bytes:")
    for b in first:
        print(f"  {b['component']:32s} {b['bytes']:>7}")
    print(f"\nwrote {RESULTS / 'size.csv'} and {RESULTS / 'size_breakdown.csv'}")


if __name__ == "__main__":
    main()
