"""Cached, seeded CV reward for a (feature cells, hyper-parameters) hypothesis."""
from __future__ import annotations

import math
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .ops import FeatureBuilder

# discretised hyper-parameter rows: (name, choices). One cell per row may be painted.
HPARAMS = [
    ("model", ["logreg", "hgb"]),
    ("depth", [2, 3, 4]),
    ("lr", [0.05, 0.1, 0.2]),
    ("iters", [50, 100, 200]),
]
DEFAULTS = {"model": "hgb", "depth": 3, "lr": 0.1, "iters": 100}


def make_model(hp: dict, seed: int):
    if hp["model"] == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=seed))
    return HistGradientBoostingClassifier(max_depth=hp["depth"], learning_rate=hp["lr"], max_iter=hp["iters"],
                                          early_stopping=False, random_state=seed)


class RewardEvaluator:
    """cv_accuracy(cells, hp) with an exact cache keyed by the hypothesis; every evaluation is
    a seeded stratified 5-fold CV with the feature builder refitted per fold."""

    def __init__(self, X, y, seed: int = 0, beta: float = 60.0, base: float = 0.78, n_splits: int = 5):
        self.X, self.y, self.seed, self.beta, self.base = X, y.values, seed, beta, base
        self.folds = list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(X, y))
        self.cache: dict[tuple, float] = {}
        self.evals = 0
        self.eval_seconds = 0.0

    @staticmethod
    def key(cells, hp) -> tuple:
        return (tuple(sorted(cells)), tuple(sorted(hp.items())))

    def cv_accuracy(self, cells, hp) -> float:
        hp = {**DEFAULTS, **hp}
        k = self.key(cells, hp)
        if k in self.cache:
            return self.cache[k]
        t = time.time()
        accs = []
        for tr, va in self.folds:
            fb = FeatureBuilder(cells).fit(self.X.iloc[tr])
            m = make_model(hp, self.seed).fit(fb.transform(self.X.iloc[tr]), self.y[tr])
            accs.append(float((m.predict(fb.transform(self.X.iloc[va])) == self.y[va]).mean()))
        acc = float(np.mean(accs))
        self.cache[k] = acc
        self.evals += 1
        self.eval_seconds += time.time() - t
        return acc

    def log_reward(self, cells, hp) -> float:
        return self.beta * (self.cv_accuracy(cells, hp) - self.base)

    def reward(self, cells, hp) -> float:
        return math.exp(self.log_reward(cells, hp))
