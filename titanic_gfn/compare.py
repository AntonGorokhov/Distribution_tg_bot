#!/usr/bin/env python3
"""Controls for the GFlowNet run, all on the same seeded evaluator.

* random search with the untrained policy's action prior (each valid cell with probability
  1/2; each hyper-parameter row unset -> default, or one of its values, uniformly over the
  1 + k options), at the GFlowNet's training-phase budget;
* Optuna TPE over the same discrete space (30 booleans + 4 categoricals), seeded;
* mean-field: cells drawn independently at the trained generator's own marginals and
  hyper-parameters drawn from its empirical distribution (does the generator carry any
  structure beyond its marginals?).

Scores: "search CV" is the single seeded 5-fold accuracy every search optimised;
"re-scored" is the same pipeline on three fresh 5-fold splits (seeds 1-3), averaged.
Equal-budget rows are cut from the recorded evaluation order (no replay needed).
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import time
from collections import Counter

import numpy as np
import optuna
import pandas as pd

from titanic.determinism import reexec_with_fixed_hashseed, seed_everything

if __name__ == "__main__":
    reexec_with_fixed_hashseed()

from .ops import COLUMNS, OPS, VALID  # noqa: E402
from .reward import DEFAULTS, HPARAMS, RewardEvaluator  # noqa: E402
from .train import BatchedReward  # noqa: E402

optuna.logging.set_verbosity(optuna.logging.WARNING)
CELLS = [(c, o) for c in COLUMNS for o in OPS if VALID[c][o]]
RESCORE_SEEDS = (1, 2, 3)   # fresh CV splits, never used during any search
TOP = 10


def load_json(d: str, name: str):
    p = os.path.join(d, name)
    if not os.path.exists(p) and os.path.exists(p + ".gz"):
        return json.load(gzip.open(p + ".gz", "rt"))
    return json.load(open(p))


def rescore(hyps: list[tuple], X, y) -> list[float]:
    evs = [RewardEvaluator(X, y, seed=s) for s in RESCORE_SEEDS]
    return [float(np.mean([ev.cv_accuracy(list(c), dict(h)) for ev in evs])) for c, h in hyps]


def jaccard_diversity(cell_sets: list[frozenset]) -> float:
    d, n = [], len(cell_sets)
    for i in range(n):
        for j in range(i + 1, n):
            u = len(cell_sets[i] | cell_sets[j])
            d.append(1 - len(cell_sets[i] & cell_sets[j]) / u if u else 0.0)
    return round(float(np.mean(d)), 3) if d else 0.0


def summarize(name: str, evaluated: list[tuple[tuple, float]], seconds, X, y, budget_note: str = "") -> dict:
    """evaluated: list of (key, acc) in evaluation order, unique keys."""
    accs = np.array([a for _, a in evaluated])
    best_so_far = np.maximum.accumulate(accs).tolist()
    top = sorted(evaluated, key=lambda kv: -kv[1])[:TOP]
    resc = rescore([(k[0], k[1]) for k, _ in top], X, y)
    return {
        "method": name, "budget_note": budget_note, "unique_evaluations": len(evaluated),
        "seconds": None if seconds is None else round(seconds, 1),
        "best_acc": float(accs.max()), "mean_acc": round(float(accs.mean()), 4),
        "n_above_083": int((accs >= 0.83).sum()), "n_above_084": int((accs >= 0.84).sum()),
        "frac_above_083": round(float((accs >= 0.83).mean()), 4),
        "top20_mean": round(float(np.sort(accs)[-20:].mean()), 4),
        "rescored_of_search_best": round(resc[0], 4),
        "rescored_best": round(max(resc), 4), "rescored_top10_mean": round(float(np.mean(resc)), 4),
        "top10_diversity": jaccard_diversity([frozenset(k[0]) for k, _ in top]),
        "best_so_far": best_so_far[:: max(1, len(best_so_far) // 400)] + [best_so_far[-1]],
        "top10": [{"acc": a, "rescored": round(r, 4), "cells": [f"{c}:{o}" for c, o in k[0]], "hp": dict(k[1])} for (k, a), r in zip(top, resc)],
    }


def draw_hp_policy_prior(rng) -> dict:
    """Untrained policy's prior on a hyper-parameter row: unset (default) or any value, uniformly."""
    hp = {}
    for n, ch in HPARAMS:
        j = rng.integers(len(ch) + 1)
        if j < len(ch):
            hp[n] = ch[j]
    return hp


def batch_evaluate(R: BatchedReward, proposals, budget: int) -> list:
    evaluated, seen = [], set()
    while len(evaluated) < budget:
        batch = []
        for _ in range(96):
            cells, hp = proposals()
            k = RewardEvaluator.key(cells, {**DEFAULTS, **hp})
            if k not in seen:
                seen.add(k); batch.append((cells, hp, k))
        accs = R.accuracies([(c, h) for c, h, _ in batch])
        for (_, _, k), a in zip(batch, accs):
            if len(evaluated) < budget:
                evaluated.append((k, a))
    return evaluated


def random_search(R, budget, rng):
    return batch_evaluate(R, lambda: ([c for c in CELLS if rng.random() < 0.5], draw_hp_policy_prior(rng)), budget)


def mean_field(R, budget, rng, samples: list) -> list:
    n = len(samples)
    marg = {c: sum(c in s["cells"] for s in samples) / n for c in (f"{a}:{b}" for a, b in CELLS)}
    hps = [s["hp"] for s in samples]
    return batch_evaluate(R, lambda: ([c for c in CELLS if rng.random() < marg[f"{c[0]}:{c[1]}"]], dict(hps[rng.integers(n)])), budget)


def untrained_policy(R, n: int, seed: int) -> tuple[list, dict]:
    """Samples from the untrained U-Net policy (uniform over valid actions, STOP included), the
    distribution the GFlowNet starts from; evaluated like the trained generator's samples."""
    import torch
    from gfn.unet import UNetPolicy
    from .env import CanvasEnv
    from .train import rollout
    torch.manual_seed(seed); torch.set_num_threads(1)
    env, model, gen = CanvasEnv(), UNetPolicy(base=32), torch.Generator().manual_seed(seed)
    with torch.no_grad():
        canvas = torch.cat([rollout(model, env, 32, 0.0, gen)[0] for _ in range(int(np.ceil(n / 32)))])[:n]
    hyps = [env.decode(c) for c in canvas]
    accs = R.accuracies(hyps)
    seen, evaluated = set(), []
    for (cells, hp), a in zip(hyps, accs):
        k = RewardEvaluator.key(cells, {**DEFAULTS, **hp})
        if k not in seen:
            seen.add(k); evaluated.append((k, a))
    marg = {f"{a}:{b}": round(sum((a, b) in h[0] for h in hyps) / n, 3) for a, b in CELLS}
    extra = {"cell_marginals": marg, "cells_per_canvas": [round(float(np.mean([len(h[0]) for h in hyps])), 2), round(float(np.std([len(h[0]) for h in hyps])), 2)],
             "hp_marginals": {name: dict(Counter(str(h[1].get(name, "unset")) for h in hyps)) for name, _ in HPARAMS}, "mean_acc_all_samples": round(float(np.mean(accs)), 4)}
    return evaluated, extra


def tpe_search(R, budget, seed, max_trials):
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


def optuna_branch_reference(X, y) -> dict | None:
    """Top-10 trials of the Optuna branch (LightGBM etc., hand-written hypotheses), re-scored on
    the same fresh splits. Its search CV is a different protocol (2x5-fold, seed 42)."""
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from titanic.search import build_pipeline, split_params
    path = os.path.join("titanic", "results", "trials.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path).sort_values("value", ascending=False).head(TOP)
    resc = []
    for _, row in df.iterrows():
        params = {}
        for k, v in row.items():
            if k in ("number", "value", "cv_std", "n_features") or pd.isna(v):
                continue
            if k.startswith("h_") and k != "h_age_impute":
                v = str(v) == "True"
            elif isinstance(v, float) and v.is_integer() and not k.endswith(("learning_rate", "subsample", "colsample", "reg_lambda", "l2", "_C", "_gamma")):
                v = int(v)          # CSV turned ints into floats where other models left NaN
            elif k == "rf_max_features" and v not in ("sqrt", "None"):
                v = float(v)
            params[k] = None if v == "None" else v
        hyp, model, mp = split_params(params)
        pipe = build_pipeline(hyp, model, mp, 42)
        resc.append(float(np.mean([cross_val_score(pipe, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=s), scoring="accuracy", n_jobs=1).mean()
                                   for s in RESCORE_SEEDS])))
    return {"method": "Optuna branch (hand-written hypotheses, LightGBM et al.)", "search_protocol": "2x stratified 5-fold, seed 42",
            "unique_evaluations": 120, "best_acc": float(df["value"].max()),
            "rescored_of_search_best": round(resc[0], 4), "rescored_best": round(max(resc), 4), "rescored_top10_mean": round(float(np.mean(resc)), 4)}


def gfn_rows(gfn_out: str, X, y) -> dict:
    ev = load_json(gfn_out, "evaluations.json")
    train = [((tuple(tuple(c) for c in r["cells"]), tuple(sorted(r["hp"].items()))), r["acc"]) for r in ev if r["phase"] == "train"]
    res = load_json(gfn_out, "results.json")
    out = {"gflownet": summarize("gflownet (training phase)", train, res.get("train_seconds"), X, y, "all evaluations made during training")}
    if len(train) > 3000:
        out["gflownet@3000"] = summarize("gflownet (first 3 000 evaluations)", train[:3000], None, X, y, "training order")
    samples = load_json(gfn_out, "samples.json")
    sample_pairs = [(( tuple(tuple(c.split(":")) for c in s["cells"]), tuple(sorted(s["hp"].items()))), s["acc"]) for s in samples]
    out["gflownet-samples"] = summarize("gflownet generator (on-policy samples)", sample_pairs, None, X, y, f"{len(samples)} samples, eps = 0")
    n = len(samples)
    marg = {f"{a}:{b}": round(sum(f"{a}:{b}" in s["cells"] for s in samples) / n, 3) for a, b in CELLS}
    out["gflownet-samples"]["cell_marginals"] = marg
    out["gflownet-samples"]["hp_marginals"] = {name: dict(Counter(str(s["hp"].get(name, "unset")) for s in samples)) for name, _ in HPARAMS}
    out["gflownet-samples"]["cells_per_canvas"] = [round(float(np.mean([len(s["cells"]) for s in samples])), 2), round(float(np.std([len(s["cells"]) for s in samples])), 2)]
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gfn-out", default="out/titanic_gfn")
    p.add_argument("--budget", type=int, default=None, help="unique evaluations for random; default = GFlowNet training-phase count")
    p.add_argument("--tpe-budget", type=int, default=3000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--methods", default="random,meanfield,tpe,reference")
    p.add_argument("--out", default="out/titanic_gfn")
    args = p.parse_args()
    seed_everything(args.seed)
    os.makedirs(args.out, exist_ok=True)
    methods = set(args.methods.split(","))

    res = load_json(args.gfn_out, "results.json")
    budget = args.budget or res["train_evaluations"]
    R = BatchedReward(args.seed, args.workers, beta=res["beta"], base=res["base"])
    X, y = R.local.X, pd.Series(R.local.y)
    out = gfn_rows(args.gfn_out, X, y)
    rng = np.random.default_rng(args.seed)
    if "random" in methods:
        t = time.time(); ev = random_search(R, budget, rng)
        out["random"] = summarize("random search (untrained-policy prior)", ev, time.time() - t, X, y, "same budget as GFlowNet training")
        if budget > 3000:
            out["random@3000"] = summarize("random search (first 3 000)", ev[:3000], None, X, y, "evaluation order")
        print(f"random : best {out['random']['best_acc']:.4f} | >=0.83: {out['random']['n_above_083']} | {out['random']['seconds']:.0f}s", flush=True)
    if "meanfield" in methods:
        samples = load_json(args.gfn_out, "samples.json")
        t = time.time(); ev = mean_field(R, len(samples), rng, samples)
        out["meanfield"] = summarize("mean-field (independent cells at the generator's marginals)", ev, time.time() - t, X, y, f"{len(samples)} draws")
        print(f"meanfld: best {out['meanfield']['best_acc']:.4f} | >=0.83: {out['meanfield']['n_above_083']} | mean {out['meanfield']['mean_acc']}", flush=True)
    if "untrained" in methods:
        n = len(load_json(args.gfn_out, "samples.json"))
        ev, extra = untrained_policy(R, n, res["seed"])
        out["untrained"] = {**summarize("untrained policy (uniform over valid actions)", ev, None, X, y, f"{n} samples"), **extra}
        print(f"untrain: mean {extra['mean_acc_all_samples']} | >=0.83: {out['untrained']['n_above_083']} | cells {extra['cells_per_canvas']}", flush=True)
    if "tpe" in methods:
        t = time.time(); ev = tpe_search(R, args.tpe_budget, args.seed, max_trials=args.tpe_budget * 3)
        out["tpe"] = summarize("Optuna TPE", ev, time.time() - t, X, y, f"{args.tpe_budget} evaluations")
        print(f"tpe    : best {out['tpe']['best_acc']:.4f} | >=0.83: {out['tpe']['n_above_083']} | {out['tpe']['seconds']:.0f}s", flush=True)
    R.close()
    if "reference" in methods:
        ref = optuna_branch_reference(X, y)
        if ref:
            out["reference"] = ref
    json.dump(out, open(os.path.join(args.out, "comparison.json"), "w"), indent=1)
    print("\nmethod                               uniq  best(search) search-best re-scored  best re-scored  top10 re-scored  diversity  >=0.83 >=0.84")
    for k, s in out.items():
        print(f"{s['method'][:36]:36s} {s['unique_evaluations']:5d}  {s['best_acc']:.4f}       {s['rescored_of_search_best']:.4f}              "
              f"{s['rescored_best']:.4f}          {s['rescored_top10_mean']:.4f}         {s.get('top10_diversity', float('nan')):.3f}   {s.get('n_above_083', 0):6d} {s.get('n_above_084', 0):6d}")


if __name__ == "__main__":
    main()
