# Titanic with a deterministic hypothesis search

Goal: a Kaggle Titanic solution where the **whole search** over feature hypotheses and
model hyper-parameters is a pure function of one seed. Two runs on the same machine
produce byte-identical trial tables and submissions; `verify_determinism.py` proves it.

## What is searched

Feature hypotheses (`titanic/features.py`, each a switch the sampler can flip):

| switch | meaning |
|---|---|
| `title` | Mr / Mrs / Miss / Master / Rare parsed from `Name` |
| `family` | `FamilySize = SibSp + Parch + 1`, `IsAlone` |
| `deck` | first letter of `Cabin`, `U` for unknown |
| `ticket_group` | passengers on the same ticket, counted among the training rows plus the row itself (same definition for fit rows and new rows) |
| `fare_log` | `log1p(Fare)` |
| `age_bins` | ordinal age bucket next to raw age |
| `age_impute` | missing `Age` by global median or by (Title, Pclass) median |

Models (`titanic/models.py`): logistic regression, random forest, sklearn
HistGradientBoosting, LightGBM, RBF SVM, each with its own hyper-parameter space.
The objective is mean accuracy over 2x repeated stratified 5-fold CV (Kaggle's metric),
all imputation statistics fitted inside each training fold.

## Why it is deterministic

| source of randomness | what pins it |
|---|---|
| sampler | `optuna.samplers.TPESampler(seed=…)`, `n_jobs=1` (strictly sequential trials) |
| CV splits | `RepeatedStratifiedKFold(random_state=seed)` |
| estimators | `random_state=seed`, one thread; LightGBM with `deterministic=True, force_row_wise=True` |
| BLAS / OpenMP | `OMP_NUM_THREADS=1` etc. set in `titanic/__init__.py` before numpy loads |
| hash ordering | the entry point re-execs itself with `PYTHONHASHSEED=0`; encodings use fixed category lists anyway |
| data | committed CSVs verified against the Kaggle md5 before every run |

Everything is written to `out/titanic/`: `study.db` (Optuna SQLite, for inspection; every run starts fresh),
`trials.csv`, `submission.csv`, `results.json` (best config, its fingerprint, sha256 of
the submission and of the trial table, library versions, git commit).

## Run

```bash
pip install -e ".[titanic]"                           # or: pip install -r requirements.txt
python -m titanic.run --trials 120 --seed 42          # ~3 min on 1 CPU core
python -m titanic.verify_determinism --trials 30 --flaml   # A == B, A != other seed
python -m titanic.flaml_baseline --max-iter 60        # AutoML comparator (FLAML, fixed iterations)
python -m pytest tests/test_titanic.py
```

## Results

Committed run: `python -m titanic.run --trials 120 --seed 42` (144 s on one core).
Outputs are in `titanic/results/`.

| | |
|---|---|
| best CV accuracy (2x5-fold) | **0.8406** ± 0.0170 (trial #110) |
| best model | lgbm: n_estimators=275, learning_rate=0.0642, num_leaves=12, min_child_samples=17, subsample=0.7822, colsample_bytree=0.6965, reg_lambda=4.1282 |
| hypotheses picked | title=True, family=True, deck=False, ticket_group=True, fare_log=True, age_bins=False, age_impute=median |
| submission sha256 | `2ffe1014f66e2bc8…` (predicted survival rate 0.373) |
| FLAML comparator (60 iterations) | lgbm, CV accuracy 0.8384 |

Per model family over the 120 trials:

| model | trials | best CV | mean CV |
|---|---|---|---|
| lgbm | 75 | 0.8406 | 0.8326 |
| logreg | 8 | 0.8316 | 0.8158 |
| svc | 10 | 0.8316 | 0.7533 |
| hgb | 15 | 0.8311 | 0.8222 |
| rf | 12 | 0.8305 | 0.8251 |

Determinism check (`python -m titanic.verify_determinism --trials 30 --flaml`), output in
`titanic/results/determinism_check.txt`:

```
optuna  seed 42 run A vs B: trials identical=True, submission identical=True; best acc 0.83894 vs 0.83894
optuna  seed 43 differs from seed 42: True (best acc 0.82943)
flaml   seed 42 run A vs B: config identical=True, submission identical=True; best acc 0.83839 vs 0.83839
DETERMINISTIC
```

Kaggle's public leaderboard was not queried from this environment (no Kaggle access);
pipelines in this CV range typically score 0.77–0.80 on the public test split.
