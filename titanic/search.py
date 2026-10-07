"""Seeded Optuna search over feature hypotheses x model x hyper-parameters.

Determinism comes from: TPESampler(seed), n_jobs=1 (sequential trials), seeded CV
splits, seeded single-threaded estimators, and a fixed category order in every
suggest_categorical call. The study is persisted to SQLite for inspection only: run_study
always starts from an empty study (study.db is deleted); resuming an existing study with a
freshly constructed sampler would re-seed the RNG and diverge from an uninterrupted run.
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import HYPOTHESES, TitanicFeatures
from .models import MODELS, NEEDS_SCALING, make_model, suggest_params

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)


def suggest_hypotheses(trial) -> dict:
    return {k: trial.suggest_categorical(f"h_{k}", v) for k, v in HYPOTHESES.items()}


def build_pipeline(hyp: dict, model: str, params: dict, seed: int) -> Pipeline:
    steps = [("features", TitanicFeatures(**hyp))]
    if model in NEEDS_SCALING:
        steps.append(("scale", StandardScaler()))
    steps.append(("model", make_model(model, params, seed)))
    return Pipeline(steps)


def make_cv(seed: int, n_splits: int = 5, n_repeats: int = 2):
    return RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)


def make_objective(X: pd.DataFrame, y: pd.Series, seed: int, n_repeats: int):
    def objective(trial: optuna.Trial) -> float:
        hyp = suggest_hypotheses(trial)
        model = trial.suggest_categorical("model", MODELS)
        params = suggest_params(trial, model)
        pipe = build_pipeline(hyp, model, params, seed)
        scores = cross_val_score(pipe, X, y, cv=make_cv(seed, n_repeats=n_repeats), scoring="accuracy", n_jobs=1)
        trial.set_user_attr("cv_std", float(np.std(scores)))
        trial.set_user_attr("n_features", int(pipe.named_steps["features"].fit(X.iloc[:64]).transform(X.iloc[:64]).shape[1]))
        return float(np.mean(scores))
    return objective


def run_study(X, y, seed: int, n_trials: int, out_dir: str, n_repeats: int = 2, study_name: str = "titanic",
              n_startup_trials: int = 15) -> optuna.Study:
    os.makedirs(out_dir, exist_ok=True)
    db = os.path.join(out_dir, "study.db")
    if os.path.exists(db):
        os.remove(db)
    sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True, group=True, n_startup_trials=n_startup_trials)
    study = optuna.create_study(direction="maximize", sampler=sampler, study_name=study_name,
                                storage=f"sqlite:///{db}", load_if_exists=False)
    study.optimize(make_objective(X, y, seed, n_repeats), n_trials=n_trials, n_jobs=1, show_progress_bar=False)
    return study


def trials_frame(study: optuna.Study) -> pd.DataFrame:
    rows = []
    for t in study.trials:
        row = {"number": t.number, "value": t.value, "model": t.params.get("model"),
               "cv_std": t.user_attrs.get("cv_std"), "n_features": t.user_attrs.get("n_features")}
        row.update({k: v for k, v in sorted(t.params.items())})
        rows.append(row)
    return pd.DataFrame(rows)


def split_params(params: dict) -> tuple[dict, str, dict]:
    hyp = {k[2:]: v for k, v in params.items() if k.startswith("h_")}
    model = params["model"]
    mp = {k[len(model) + 1:]: v for k, v in params.items() if k.startswith(model + "_")}
    # restore the names make_model expects
    rename = {"l2": "l2_regularization", "colsample": "colsample_bytree"}
    mp = {rename.get(k, k): v for k, v in mp.items()}
    return hyp, model, mp
