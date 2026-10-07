"""Model space. Every estimator is built with an explicit seed and a single thread."""
from __future__ import annotations

from lightgbm import LGBMClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

MODELS = ["logreg", "rf", "hgb", "lgbm", "svc"]
NEEDS_SCALING = {"logreg", "svc"}


def suggest_params(trial, name: str) -> dict:
    if name == "logreg":
        return {"C": trial.suggest_float("logreg_C", 1e-3, 1e2, log=True)}
    if name == "rf":
        return {
            "n_estimators": trial.suggest_int("rf_n_estimators", 100, 400, step=50),
            "max_depth": trial.suggest_int("rf_max_depth", 3, 12),
            "min_samples_leaf": trial.suggest_int("rf_min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("rf_max_features", ["sqrt", 0.5, None]),
        }
    if name == "hgb":
        return {
            "learning_rate": trial.suggest_float("hgb_learning_rate", 0.02, 0.3, log=True),
            "max_iter": trial.suggest_int("hgb_max_iter", 50, 400, step=25),
            "max_leaf_nodes": trial.suggest_int("hgb_max_leaf_nodes", 4, 32),
            "min_samples_leaf": trial.suggest_int("hgb_min_samples_leaf", 5, 40),
            "l2_regularization": trial.suggest_float("hgb_l2", 1e-4, 10.0, log=True),
        }
    if name == "lgbm":
        return {
            "n_estimators": trial.suggest_int("lgbm_n_estimators", 50, 400, step=25),
            "learning_rate": trial.suggest_float("lgbm_learning_rate", 0.02, 0.3, log=True),
            "num_leaves": trial.suggest_int("lgbm_num_leaves", 4, 32),
            "min_child_samples": trial.suggest_int("lgbm_min_child_samples", 5, 40),
            "subsample": trial.suggest_float("lgbm_subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("lgbm_colsample", 0.6, 1.0),
            "reg_lambda": trial.suggest_float("lgbm_reg_lambda", 1e-3, 10.0, log=True),
        }
    if name == "svc":
        return {
            "C": trial.suggest_float("svc_C", 1e-2, 1e2, log=True),
            "gamma": trial.suggest_float("svc_gamma", 1e-3, 1.0, log=True),
        }
    raise ValueError(name)


def make_model(name: str, params: dict, seed: int):
    if name == "logreg":
        return LogisticRegression(max_iter=2000, random_state=seed, **params)
    if name == "rf":
        return RandomForestClassifier(random_state=seed, n_jobs=1, **params)
    if name == "hgb":
        return HistGradientBoostingClassifier(random_state=seed, early_stopping=False, **params)
    if name == "lgbm":
        return LGBMClassifier(random_state=seed, n_jobs=1, deterministic=True, force_row_wise=True,
                              subsample_freq=1, verbose=-1, **params)
    if name == "svc":
        return SVC(random_state=seed, **params)
    raise ValueError(name)
