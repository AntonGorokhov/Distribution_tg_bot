"""Feature hypotheses as an sklearn transformer.

Every hypothesis is a boolean/categorical switch; the search treats them as parameters,
so "which features help" is explored with the same seeded sampler as the model
hyper-parameters. All statistics used for imputation are fit on the training fold only
(no leakage into CV), and every encoding uses a fixed category order so the column
layout never depends on the data or on dict/set ordering.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

TITLES = ["Mr", "Mrs", "Miss", "Master", "Rare"]
DECKS = ["A", "B", "C", "D", "E", "F", "G", "T", "U"]
PORTS = ["S", "C", "Q"]
AGE_BINS = [-np.inf, 12, 18, 30, 45, 60, np.inf]

HYPOTHESES = {
    "title": [False, True],              # Mr/Mrs/Miss/Master/Rare from Name
    "family": [False, True],             # FamilySize = SibSp + Parch + 1, IsAlone
    "deck": [False, True],               # first letter of Cabin, U = unknown
    "ticket_group": [False, True],       # passengers sharing the same ticket (fit data)
    "fare_log": [False, True],           # log1p(Fare)
    "age_bins": [False, True],           # ordinal age bucket next to raw age
    "age_impute": ["median", "title_pclass"],  # how missing Age is filled
}


def extract_title(name: str) -> str:
    m = re.search(r",\s*([^\.]+)\.", name)
    t = m.group(1).strip() if m else "Rare"
    t = {"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"}.get(t, t)
    return t if t in TITLES else "Rare"


class TitanicFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, title=True, family=True, deck=False, ticket_group=False,
                 fare_log=True, age_bins=False, age_impute="title_pclass"):
        self.title = title
        self.family = family
        self.deck = deck
        self.ticket_group = ticket_group
        self.fare_log = fare_log
        self.age_bins = age_bins
        self.age_impute = age_impute

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _base(df: pd.DataFrame) -> pd.DataFrame:
        b = pd.DataFrame(index=df.index)
        b["Pclass"] = df["Pclass"].astype(int)
        b["Sex"] = (df["Sex"] == "female").astype(int)
        b["SibSp"] = df["SibSp"].astype(int)
        b["Parch"] = df["Parch"].astype(int)
        b["Age"] = df["Age"].astype(float)
        b["Fare"] = df["Fare"].astype(float)
        b["Title"] = df["Name"].map(extract_title)
        b["Embarked"] = df["Embarked"].where(df["Embarked"].isin(PORTS), None)
        b["Deck"] = df["Cabin"].fillna("U").astype(str).str[0].where(lambda s: s.isin(DECKS), "U")
        b["Ticket"] = df["Ticket"].astype(str)
        return b

    # ------------------------------------------------------------------ fit
    def fit(self, X: pd.DataFrame, y=None):
        b = self._base(X)
        self.fare_median_by_pclass_ = b.groupby("Pclass")["Fare"].median().sort_index().to_dict()
        self.fare_median_ = float(b["Fare"].median())
        self.age_median_ = float(b["Age"].median())
        grp = b.groupby(["Title", "Pclass"])["Age"].median()
        self.age_median_by_title_pclass_ = {k: float(v) for k, v in sorted(grp.dropna().items())}
        self.embarked_mode_ = b["Embarked"].dropna().mode().sort_values().iloc[0]
        self.ticket_counts_ = b["Ticket"].value_counts().to_dict()
        return self

    # ------------------------------------------------------------ transform
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        b = self._base(X)
        out = pd.DataFrame(index=X.index)
        out["Pclass"] = b["Pclass"]
        out["Sex"] = b["Sex"]
        out["SibSp"] = b["SibSp"]
        out["Parch"] = b["Parch"]

        fare = b["Fare"].copy()
        miss = fare.isna()
        fare[miss] = b.loc[miss, "Pclass"].map(self.fare_median_by_pclass_).fillna(self.fare_median_)
        out["Fare"] = np.log1p(fare) if self.fare_log else fare

        age = b["Age"].copy()
        miss = age.isna()
        if self.age_impute == "title_pclass":
            fill = [self.age_median_by_title_pclass_.get((t, p), self.age_median_)
                    for t, p in zip(b.loc[miss, "Title"], b.loc[miss, "Pclass"])]
            age[miss] = fill
        else:
            age[miss] = self.age_median_
        out["Age"] = age
        out["AgeMissing"] = miss.astype(int)
        if self.age_bins:
            out["AgeBin"] = pd.cut(age, AGE_BINS, labels=False).astype(int)

        emb = b["Embarked"].fillna(self.embarked_mode_)
        for p in PORTS:
            out[f"Emb_{p}"] = (emb == p).astype(int)

        if self.title:
            for t in TITLES:
                out[f"Title_{t}"] = (b["Title"] == t).astype(int)
        if self.family:
            fam = b["SibSp"] + b["Parch"] + 1
            out["FamilySize"] = fam
            out["IsAlone"] = (fam == 1).astype(int)
        if self.deck:
            for d in DECKS:
                out[f"Deck_{d}"] = (b["Deck"] == d).astype(int)
        if self.ticket_group:
            out["TicketGroup"] = b["Ticket"].map(self.ticket_counts_).fillna(1).astype(int)
        return out

    def get_feature_names_out(self, input_features=None):
        return None
