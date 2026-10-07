#!/usr/bin/env python3
"""Proof of determinism: run the full pipeline twice in fresh processes with the same seed,
require byte-identical trial tables and submissions; run once more with another seed and
require a different search trace (so the check is not vacuous)."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys


def run(module: str, out: str, seed: int, extra: list[str]) -> dict:
    cmd = [sys.executable, "-m", module, "--seed", str(seed), "--out", out, *extra]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:], r.stderr[-4000:])
        raise SystemExit(f"{module} failed")
    return json.load(open(os.path.join(out, "results.json")))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trials", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--flaml", action="store_true", help="also check the FLAML comparator")
    p.add_argument("--out", default="out/determinism")
    args = p.parse_args()
    ok = True

    a = run("titanic.run", f"{args.out}/A", args.seed, ["--trials", str(args.trials)])
    b = run("titanic.run", f"{args.out}/B", args.seed, ["--trials", str(args.trials)])
    c = run("titanic.run", f"{args.out}/C", args.seed + 1, ["--trials", str(args.trials)])
    same_trials = a["trials_sha256"] == b["trials_sha256"]
    same_sub = a["submission_sha256"] == b["submission_sha256"]
    diff_seed = a["trials_sha256"] != c["trials_sha256"] and a["params_sha256"] != c["params_sha256"]
    print(f"optuna  seed {args.seed} run A vs B: trials identical={same_trials}, submission identical={same_sub}; "
          f"best acc {a['best_cv_accuracy']} vs {b['best_cv_accuracy']}")
    print(f"optuna  seed {args.seed + 1} differs from seed {args.seed} (sampled params and values): {diff_seed} (best acc {c['best_cv_accuracy']})")
    ok &= same_trials and same_sub and diff_seed

    if args.flaml:
        fa = run("titanic.flaml_baseline", f"{args.out}/FA", args.seed, [])
        fb = run("titanic.flaml_baseline", f"{args.out}/FB", args.seed, [])
        same_cfg = fa["best_config"] == fb["best_config"] and fa["best_estimator"] == fb["best_estimator"]
        same_sub = fa["submission_sha256"] == fb["submission_sha256"]
        print(f"flaml   seed {args.seed} run A vs B: config identical={same_cfg}, submission identical={same_sub}; "
              f"best acc {fa['best_cv_accuracy']} vs {fb['best_cv_accuracy']}")
        ok &= same_cfg and same_sub

    print("DETERMINISTIC" if ok else "NOT DETERMINISTIC")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
