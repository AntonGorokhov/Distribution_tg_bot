import numpy as np
import pandas as pd

from titanic import data
from titanic.features import HYPOTHESES, TitanicFeatures, extract_title
from titanic.search import build_pipeline, make_cv, run_study, split_params, trials_frame


def test_data_is_kaggle_files():
    md5 = data.verify()
    assert set(md5) == {"train.csv", "test.csv"}
    X, y, X_test = data.load()
    assert X.shape == (891, 11) and X_test.shape == (418, 11) and y.isin([0, 1]).all()


def test_titles():
    assert extract_title("Braund, Mr. Owen Harris") == "Mr"
    assert extract_title("Heikkinen, Miss. Laina") == "Miss"
    assert extract_title("Andersson, Master. Gustaf") == "Master"
    assert extract_title("Byles, Rev. Thomas Roussel Davids") == "Rare"
    assert extract_title("Oliva y Ocana, Dona. Fermina") == "Rare"


def test_features_no_nan_fixed_layout_and_no_leak():
    X, y, X_test = data.load()
    f = TitanicFeatures(title=True, family=True, deck=True, ticket_group=True, fare_log=True,
                        age_bins=True, age_impute="title_pclass").fit(X)
    A, B = f.transform(X), f.transform(X_test)
    assert not A.isna().any().any() and not B.isna().any().any()
    assert list(A.columns) == list(B.columns)
    # statistics come from the fit data only: refitting on a subset changes the fill value
    g = TitanicFeatures(age_impute="median").fit(X.iloc[:100])
    assert g.age_median_ != f.age_median_ or g.fare_median_ != f.fare_median_
    assert all(hyp in f.get_params() for hyp in HYPOTHESES)
    # ticket group: a fit row counts itself, a new row on the same ticket adds itself
    fit = pd.DataFrame({"PassengerId": [1, 2, 3], "Pclass": [1, 1, 2], "Name": ["A, Mr. X", "B, Mrs. Y", "C, Mr. Z"], "Sex": ["male", "female", "male"],
                        "Age": [30, 28, 40], "SibSp": [0, 0, 0], "Parch": [0, 0, 0], "Ticket": ["T1", "T1", "T2"], "Fare": [10, 10, 7], "Cabin": [None] * 3, "Embarked": ["S"] * 3})
    new = fit.assign(PassengerId=[11, 12, 13])
    g = TitanicFeatures(ticket_group=True).fit(fit)
    assert g.transform(fit)["TicketGroup"].tolist() == [2, 2, 1]
    assert g.transform(new)["TicketGroup"].tolist() == [3, 3, 2]
    assert g.transform(new.assign(Ticket=["ZZ"] * 3))["TicketGroup"].tolist() == [1, 1, 1]


def test_cv_and_search_are_reproducible(tmp_path):
    X, y, _ = data.load()
    cv1 = [tr[:5].tolist() for tr, _ in make_cv(3).split(X, y)]
    cv2 = [tr[:5].tolist() for tr, _ in make_cv(3).split(X, y)]
    assert cv1 == cv2
    kw = dict(n_trials=6, n_repeats=1, n_startup_trials=2)  # 4 of 6 trials come from the seeded TPE phase
    s1 = run_study(X, y, seed=3, out_dir=str(tmp_path / "a"), **kw)
    s2 = run_study(X, y, seed=3, out_dir=str(tmp_path / "b"), **kw)
    pd.testing.assert_frame_equal(trials_frame(s1), trials_frame(s2))
    s3 = run_study(X, y, seed=4, out_dir=str(tmp_path / "c"), **kw)
    params = lambda s: trials_frame(s).drop(columns=["value", "cv_std"])
    assert not params(s1).equals(params(s3))
    hyp, model, params = split_params(s1.best_trial.params)
    pipe = build_pipeline(hyp, model, params, 3).fit(X, y)
    assert set(np.unique(pipe.predict(X))) <= {0, 1}
