"""Kaggle Titanic data. Files are committed under titanic/data and verified by md5 (the
official Kaggle train.csv/test.csv). download() fetches the same files from a GitHub mirror."""
from __future__ import annotations

import os
import urllib.request

import pandas as pd

from .determinism import md5_file

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MIRROR = "https://raw.githubusercontent.com/agconti/kaggle-titanic/master/data/"
EXPECTED_MD5 = {
    "train.csv": "61fdd54abdbf6a85b778e937122e1194",
    "test.csv": "029c9cd22461f6dbe8d9ab01def965c6",
}


def download(force: bool = False) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    for name in EXPECTED_MD5:
        path = os.path.join(DATA_DIR, name)
        if force or not os.path.exists(path):
            urllib.request.urlretrieve(MIRROR + name, path)


def verify() -> dict:
    out = {}
    for name, md5 in EXPECTED_MD5.items():
        path = os.path.join(DATA_DIR, name)
        got = md5_file(path)
        if got != md5:
            raise RuntimeError(f"{name}: md5 {got} != expected {md5}; not the Kaggle file")
        out[name] = got
    return out


def load() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Returns (train_X_raw, y, test_X_raw) with PassengerId kept as a column."""
    verify()
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    y = train.pop("Survived").astype(int)
    return train, y, test
