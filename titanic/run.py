#!/usr/bin/env python3
"""End-to-end: deterministic search -> refit best on full train -> submission.csv + results.json."""
from __future__ import annotations

import argparse
import json
import os
import time

from .determinism import (config_fingerprint, environment_info, reexec_with_fixed_hashseed,
                          seed_everything, sha256_file)

if __name__ == "__main__":  # re-exec only as the program, never when merely imported
    reexec_with_fixed_hashseed()

from . import data  # noqa: E402  (after the re-exec so the import happens once)
from .search import build_pipeline, run_study, split_params, trials_frame  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trials", type=int, default=120)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--repeats", type=int, default=2, help="repeats of stratified 5-fold CV per trial")
    p.add_argument("--out", default="out/titanic")
    args = p.parse_args()

    seed_everything(args.seed)
    X, y, X_test = data.load()
    t0 = time.time()
    study = run_study(X, y, seed=args.seed, n_trials=args.trials, out_dir=args.out, n_repeats=args.repeats)
    search_seconds = time.time() - t0

    df = trials_frame(study)
    df.to_csv(os.path.join(args.out, "trials.csv"), index=False)
    best = study.best_trial
    hyp, model, params = split_params(best.params)
    pipe = build_pipeline(hyp, model, params, args.seed).fit(X, y)
    pred = pipe.predict(X_test)
    sub_path = os.path.join(args.out, "submission.csv")
    import pandas as pd
    pd.DataFrame({"PassengerId": X_test["PassengerId"], "Survived": pred.astype(int)}).to_csv(sub_path, index=False)

    by_model = df.groupby("model")["value"].agg(["count", "max", "mean"]).round(4).to_dict("index")
    results = {
        "seed": args.seed, "trials": args.trials, "cv": f"{args.repeats}x stratified 5-fold",
        "best_trial": best.number, "best_cv_accuracy": round(best.value, 5), "best_cv_std": round(best.user_attrs["cv_std"], 5),
        "best_model": model, "best_hypotheses": hyp, "best_params": params,
        "best_fingerprint": config_fingerprint(best.params),
        "best_cv_se_naive": round(best.user_attrs["cv_std"] / (5 * args.repeats) ** 0.5, 5),
        "n_trials_within_1se": int((df["value"] >= best.value - best.user_attrs["cv_std"] / (5 * args.repeats) ** 0.5).sum()),
        "n_trials_within_1std": int((df["value"] >= best.value - best.user_attrs["cv_std"]).sum()),
        "submission_sha256": sha256_file(sub_path),
        "trials_sha256": sha256_file(os.path.join(args.out, "trials.csv")),
        "params_sha256": config_fingerprint(df.drop(columns=["value", "cv_std"]).to_dict("records")),
        "survived_rate_test": round(float(pred.mean()), 4),
        "by_model": by_model,
        "search_seconds": round(search_seconds, 1),
        "data_md5": data.verify(),
        "env": environment_info(),
    }
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"best trial #{best.number}: CV acc {best.value:.4f} (fold std {best.user_attrs['cv_std']:.4f}, "
          f"{results['n_trials_within_1std']} trials within one std) | {model} {params}")
    print(f"hypotheses: {hyp}")
    print(f"submission sha256 {results['submission_sha256'][:16]}… | trials sha256 {results['trials_sha256'][:16]}… | {search_seconds:.0f}s")
    print(df.sort_values("value", ascending=False).head(10)[["number", "value", "cv_std", "model", "n_features"]].to_string(index=False))


if __name__ == "__main__":
    main()
