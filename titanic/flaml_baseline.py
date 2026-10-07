#!/usr/bin/env python3
"""AutoML comparator: FLAML with a fixed iteration budget (no wall-clock budget) and a seed.

With time_budget=-1 and max_iter fixed, FLAML's search is a deterministic function of the
seed, so it can be checked the same way as the Optuna study (run twice, compare)."""
from __future__ import annotations

import argparse
import json
import os
import time
import warnings

from .determinism import environment_info, reexec_with_fixed_hashseed, seed_everything, sha256_file

reexec_with_fixed_hashseed()
warnings.filterwarnings("ignore")

from . import data  # noqa: E402
from .features import TitanicFeatures  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max-iter", type=int, default=60)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="out/titanic_flaml")
    args = p.parse_args()
    seed_everything(args.seed)
    os.makedirs(args.out, exist_ok=True)

    from flaml import AutoML
    X, y, X_test = data.load()
    feats = TitanicFeatures(title=True, family=True, deck=True, ticket_group=True, fare_log=True,
                            age_bins=True, age_impute="title_pclass").fit(X)
    Xf, Xt = feats.transform(X), feats.transform(X_test)

    automl = AutoML()
    t0 = time.time()
    automl.fit(Xf, y, task="classification", metric="accuracy", time_budget=-1, max_iter=args.max_iter,
               estimator_list=["lgbm", "rf", "extra_tree", "lrl1", "lrl2"], eval_method="cv", n_splits=5,
               split_type="stratified", seed=args.seed, n_jobs=1, verbose=0, log_file_name="")
    pred = automl.predict(Xt)
    import pandas as pd
    sub = os.path.join(args.out, "submission.csv")
    pd.DataFrame({"PassengerId": X_test["PassengerId"], "Survived": pred.astype(int)}).to_csv(sub, index=False)
    results = {
        "seed": args.seed, "max_iter": args.max_iter,
        "best_estimator": automl.best_estimator, "best_config": automl.best_config,
        "best_cv_accuracy": round(1 - automl.best_loss, 5),
        "submission_sha256": sha256_file(sub), "seconds": round(time.time() - t0, 1), "env": environment_info(),
    }
    json.dump(results, open(os.path.join(args.out, "results.json"), "w"), indent=2, default=str)
    print(f"FLAML best {automl.best_estimator} CV acc {1 - automl.best_loss:.4f} | sub sha256 {results['submission_sha256'][:16]}… | {results['seconds']}s")


if __name__ == "__main__":
    main()
