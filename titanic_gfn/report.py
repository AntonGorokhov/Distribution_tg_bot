#!/usr/bin/env python3
"""Build the interactive Titanic report: collects committed results of both branches into one
JSON and inlines it into viz/titanic_report_template.html -> out/titanic_report.html."""
from __future__ import annotations

import argparse
import gzip
import json
import os

from .env import HP_ROW0, H, W
from .ops import COLUMNS, OPS, OP_DOC, VALID
from .reward import DEFAULTS, HPARAMS


def load(path):
    if not os.path.exists(path) and os.path.exists(path + ".gz"):
        return json.load(gzip.open(path + ".gz", "rt"))
    return json.load(open(path))


def collect(gfn_dir: str, optuna_dir: str) -> dict:
    cmp = load(os.path.join(gfn_dir, "comparison.json"))
    gfn = load(os.path.join(gfn_dir, "results.json"))
    hist = load(os.path.join(gfn_dir, "history.json"))
    samples = load(os.path.join(gfn_dir, "samples.json"))
    opt = load(os.path.join(optuna_dir, "results.json"))
    flaml = load(os.path.join(optuna_dir, "flaml_results.json"))
    det = open(os.path.join(optuna_dir, "determinism_check.txt")).read().strip()
    for k, v in cmp.items():
        if "top10" in v:
            v["top10"] = v["top10"][:10]
    return {
        "canvas": {"H": H, "W": W, "columns": COLUMNS, "ops": OPS, "op_doc": OP_DOC, "hp_row0": HP_ROW0,
                   "valid": [[bool(VALID[c][o]) for o in OPS] for c in COLUMNS],
                   "hparams": [[n, [str(x) for x in ch]] for n, ch in HPARAMS], "defaults": {k: str(v) for k, v in DEFAULTS.items()}},
        "compare": cmp,
        "gfn": {k: gfn[k] for k in ("seed", "iters", "batch", "beta", "base", "unique_evaluations", "eval_seconds", "total_seconds",
                                    "train_evaluations", "train_seconds", "log_z", "train_best_acc", "train_hyps_above_083",
                                    "samples", "sample_unique", "sample_mean_acc", "sample_hyps_above_083")},
        "history": hist,
        "samples_top": sorted(samples, key=lambda s: -s["acc"])[:10],
        "optuna": {k: opt[k] for k in ("trials", "cv", "best_trial", "best_cv_accuracy", "best_cv_std", "best_model", "best_hypotheses",
                                       "best_params", "n_trials_within_1std", "n_trials_within_1se", "best_cv_se_naive", "by_model",
                                       "search_seconds", "submission_sha256", "survived_rate_test")},
        "flaml": {k: flaml.get(k) for k in ("best_estimator", "best_cv_accuracy", "best_cv_accuracy_per_fold_protocol", "max_iter", "seconds")},
        "determinism": det,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gfn-dir", default="titanic_gfn/results")
    p.add_argument("--optuna-dir", default="titanic/results")
    p.add_argument("--template", default="viz/titanic_report_template.html")
    p.add_argument("--out", default="out/titanic_report.html")
    p.add_argument("--body-out", default=None, help="also write the page without the document skeleton")
    args = p.parse_args()
    run = collect(args.gfn_dir, args.optuna_dir)
    body = open(args.template, encoding="utf-8").read().replace("/*__RUN_JSON__*/", json.dumps(run, separators=(",", ":")))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write('<!doctype html><html lang="ru"><head><meta charset="utf-8">'
                '<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"></head><body>\n' + body + "\n</body></html>\n")
    if args.body_out:
        with open(args.body_out, "w", encoding="utf-8") as f:
            f.write(body)
    print(f"wrote {args.out} ({os.path.getsize(args.out) // 1024} KB)")


if __name__ == "__main__":
    main()
