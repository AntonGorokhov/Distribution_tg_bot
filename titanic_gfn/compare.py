#!/usr/bin/env python3
"""Controls for the GFlowNet run: random search and Optuna TPE over the same discrete space,
the same seeded CV evaluator and the same budget of unique evaluations.

Outputs best-so-far curves and diversity counts so the three methods can be compared per
evaluation, not per wall-clock second.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import optuna

from titanic.determinism import reexec_with_fixed_hashseed, seed_everything

reexec_with_fixed_hashseed()

from .env import CanvasEnv  # noqa: E402
from .ops import COLUMNS, OPS, VALID  # noqa: E402
from .reward import DEFAULTS, HPARAMS, RewardEvaluator  # noqa: E402
from .train import BatchedReward  # noqa: E402

optuna.logging.set_verbosity(optuna.logging.WARNING)
CELLS = [(c, o) for c in COLUMNS for o in OPS if VALID[c][o]]


def summarize(name: str, evaluated: list[tuple[tuple, float]], seconds: float) -> dict:
    """evaluated: list of (key, acc) in evaluation order, unique keys."""
    accs = np.array([a for _, a in evaluated])
    best_so_far = np.maximum.accumulate(accs).tolist()
    top = sorted(evaluated, key=lambda kv: -kv[1])[:10]
    return {
        "method": name, "unique_evaluations": len(evaluated), "seconds": round(seconds, 1),
        "best_acc": float(accs.max()), "mean_acc": round(float(accs.mean()), 4),
        "n_above_083": int((accs >= 0.83).sum()), "n_above_084": int((accs >= 0.84).sum()),
        "top20_mean": round(float(np.sort(accs)[-20:].mean()), 4),
        "best_so_far": best_so_far[:: max(1, len(best_so_far) // 400)] + [best_so_far[-1]],
        "top10": [{"acc": a, "cells": [f"{c}:{o}" for c, o in k[0]], "hp": dict(k[1])} for k, a in top],
    }


def random_search(R: BatchedReward, budget: int, rng: np.random.Generator) -> list:
    evaluated, seen = [], set()
    while len(evaluated) < budget:
        batch = []
        for _ in range(96):
            cells = [c for c in CELLS if rng.random() < 0.5]
            hp = {n: ch[rng.integers(len(ch))] for n, ch in HPARAMS}
            k = RewardEvaluator.key(cells, {**DEFAULTS, **hp})
            if k not in seen:
                seen.add(k); batch.append((cells, hp, k))
        accs = R.accuracies([(c, h) for c, h, _ in batch])
        for (_, _, k), a in zip(batch, accs):
            if len(evaluated) < budget:
                evaluated.append((k, a))
    return evaluated


def tpe_search(R: BatchedReward, budget: int, seed: int, max_trials: int) -> list:
    evaluated, seen = [], set()
    sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True, n_startup_trials=30)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    def objective(trial):
        cells = [cell for cell in CELLS if trial.suggest_categorical(f"{cell[0]}:{cell[1]}", [False, True])]
        hp = {n: trial.suggest_categorical(n, ch) for n, ch in HPARAMS}
        k = RewardEvaluator.key(cells, {**DEFAULTS, **hp})
        acc = R.accuracies([(cells, hp)])[0]
        if k not in seen:
            seen.add(k); evaluated.append((k, acc))
        if len(evaluated) >= budget:
            trial.study.stop()
        return acc

    study.optimize(objective, n_trials=max_trials, n_jobs=1)
    return evaluated


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--budget", type=int, default=None, help="unique evaluations; default = the GFlowNet run's")
    p.add_argument("--gfn-out", default="out/titanic_gfn")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--methods", default="random,tpe")
    p.add_argument("--out", default="out/titanic_gfn")
    args = p.parse_args()
    seed_everything(args.seed)
    os.makedirs(args.out, exist_ok=True)

    gfn = json.load(open(os.path.join(args.gfn_out, "results.json")))
    budget = args.budget or gfn["unique_evaluations"]
    R = BatchedReward(args.seed, args.workers, beta=gfn["beta"], base=gfn["base"])
    out = {}
    if "random" in args.methods:
        t = time.time(); ev = random_search(R, budget, np.random.default_rng(args.seed))
        out["random"] = summarize("random", ev, time.time() - t)
        print(f"random : best {out['random']['best_acc']:.4f} | >=0.83: {out['random']['n_above_083']} | {out['random']['seconds']:.0f}s", flush=True)
    if "tpe" in args.methods:
        t = time.time(); ev = tpe_search(R, budget, args.seed, max_trials=budget * 3)
        out["tpe"] = summarize("tpe", ev, time.time() - t)
        print(f"tpe    : best {out['tpe']['best_acc']:.4f} | >=0.83: {out['tpe']['n_above_083']} | {out['tpe']['seconds']:.0f}s", flush=True)
    R.close()
    # GFlowNet summary from its own evaluation cache (training order is lost; use history for the curve)
    ev_rows = json.load(open(os.path.join(args.gfn_out, "evaluations.json")))
    hist = json.load(open(os.path.join(args.gfn_out, "history.json")))
    accs = np.array([r["acc"] for r in ev_rows])
    out["gflownet"] = {
        "method": "gflownet", "unique_evaluations": len(ev_rows), "seconds": gfn["total_seconds"],
        "best_acc": float(accs.max()), "mean_acc": round(float(accs.mean()), 4),
        "n_above_083": int((accs >= 0.83).sum()), "n_above_084": int((accs >= 0.84).sum()),
        "top20_mean": round(float(np.sort(accs)[-20:].mean()), 4),
        "best_so_far_by_iter": [(h["evals"], h["acc_max"]) for h in hist],
        "sample_mean_acc": gfn["sample_mean_acc"], "sample_hyps_above_083": gfn["sample_hyps_above_083"],
    }
    out["gflownet"]["best_so_far_by_iter"] = np.maximum.accumulate([a for _, a in out["gflownet"]["best_so_far_by_iter"]]).tolist()
    json.dump(out, open(os.path.join(args.out, "comparison.json"), "w"), indent=1)
    print("\nmethod    uniq  best    mean    top20   >=0.83 >=0.84  sec")
    for m, s in out.items():
        print(f"{m:9s} {s['unique_evaluations']:5d} {s['best_acc']:.4f} {s['mean_acc']:.4f} {s['top20_mean']:.4f} {s['n_above_083']:6d} {s['n_above_084']:6d} {s['seconds']:5.0f}")


if __name__ == "__main__":
    main()
