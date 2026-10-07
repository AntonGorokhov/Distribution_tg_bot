"""Operation library: what the policy may do to a raw column.

Rows (raw columns) x columns (ops) form the feature part of the canvas. A cell is valid only
if the op applies to that column's type; invalid cells are masked for the policy forever.
Every op is a stateless or fit-on-train transform; the fitted statistics live in
`FeatureBuilder` (train fold only, no leakage), mirroring titanic.features.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from titanic.features import DECKS, PORTS, TITLES, extract_title

COLUMNS = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "Cabin", "Ticket", "Name"]
NUMERIC = {"Pclass", "Age", "SibSp", "Parch", "Fare"}
OPS = ["raw", "log", "qbin", "missing", "count", "onehot", "extract", "family"]
OP_DOC = {
    "raw": "use the column as is (numeric; Sex as 0/1)",
    "log": "log1p of a numeric column",
    "qbin": "quantile bin index (4 bins, edges fitted on train)",
    "missing": "indicator that the value is missing",
    "count": "frequency encoding (count of equal values in train)",
    "onehot": "one-hot with a fixed category list",
    "extract": "Name -> title, Cabin -> deck letter, Ticket -> prefix present",
    "family": "SibSp/Parch -> FamilySize and IsAlone",
}

# applicability mask: VALID[col][op]
VALID = {c: {o: False for o in OPS} for c in COLUMNS}
for c in NUMERIC:
    VALID[c]["raw"] = True
    VALID[c]["qbin"] = True
    VALID[c]["count"] = True
for c in ("Age", "Fare"):
    VALID[c]["log"] = True
    VALID[c]["missing"] = True
VALID["Sex"]["raw"] = True
VALID["Pclass"]["onehot"] = True
VALID["Embarked"]["onehot"] = True
VALID["Embarked"]["missing"] = True
VALID["Cabin"]["missing"] = True
VALID["Cabin"]["extract"] = True
VALID["Ticket"]["count"] = True
VALID["Ticket"]["extract"] = True
VALID["Name"]["extract"] = True
VALID["SibSp"]["family"] = True
VALID["Parch"]["family"] = True

N_FEATURE_CELLS = sum(v for c in COLUMNS for v in VALID[c].values())


def valid_mask() -> np.ndarray:
    """(len(COLUMNS), len(OPS)) boolean."""
    return np.array([[VALID[c][o] for o in OPS] for c in COLUMNS])


class FeatureBuilder:
    """Fit-on-train statistics for every op, then build the feature frame for a cell set."""

    def __init__(self, cells: list[tuple[str, str]]):
        self.cells = sorted(set(cells))

    @staticmethod
    def _title(df): return df["Name"].map(extract_title)

    def fit(self, df: pd.DataFrame):
        self.stats_ = {}
        for c, o in self.cells:
            if o == "qbin":
                q = df[c].dropna().quantile([0.25, 0.5, 0.75]).values
                self.stats_[(c, o)] = np.unique(q)
            elif o == "count":
                self.stats_[(c, o)] = df[c].value_counts().to_dict()
            if c in NUMERIC:
                self.stats_[(c, "median")] = float(df[c].median())
        self.title_age_ = {k: float(v) for k, v in sorted(df.assign(T=self._title(df)).groupby("T")["Age"].median().dropna().items())}
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = {}
        for c, o in self.cells:
            col = df[c]
            if c in NUMERIC:
                col = col.fillna(self.stats_[(c, "median")]).astype(float)
            if o == "raw":
                out[f"{c}"] = (df["Sex"] == "female").astype(int) if c == "Sex" else col
            elif o == "log":
                out[f"{c}_log"] = np.log1p(col.clip(lower=0))
            elif o == "qbin":
                out[f"{c}_qbin"] = np.searchsorted(self.stats_[(c, o)], col.values).astype(int)
            elif o == "missing":
                out[f"{c}_na"] = df[c].isna().astype(int)
            elif o == "count":
                out[f"{c}_cnt"] = df[c].map(self.stats_[(c, o)]).fillna(1).astype(int)
            elif o == "onehot":
                cats = PORTS if c == "Embarked" else [1, 2, 3]
                for k in cats:
                    out[f"{c}_{k}"] = (df[c] == k).astype(int)
            elif o == "extract":
                if c == "Name":
                    t = self._title(df)
                    for k in TITLES:
                        out[f"Title_{k}"] = (t == k).astype(int)
                elif c == "Cabin":
                    d = df["Cabin"].fillna("U").astype(str).str[0]
                    for k in DECKS:
                        out[f"Deck_{k}"] = (d == k).astype(int)
                elif c == "Ticket":
                    out["Ticket_prefix"] = (~df["Ticket"].astype(str).str.isdigit()).astype(int)
            elif o == "family":
                fam = df["SibSp"] + df["Parch"] + 1
                out["FamilySize"] = fam.astype(int)
                out["IsAlone"] = (fam == 1).astype(int)
        if not out:  # empty hypothesis: a constant column so the model still fits
            out["const"] = pd.Series(np.zeros(len(df)), index=df.index)
        return pd.DataFrame(out, index=df.index)
