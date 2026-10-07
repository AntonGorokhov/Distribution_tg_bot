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
| `ticket_group` | how many passengers share the ticket |
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

Everything is written to `out/titanic/`: `study.db` (Optuna SQLite, resumable),
`trials.csv`, `submission.csv`, `results.json` (best config, its fingerprint, sha256 of
the submission and of the trial table, library versions, git commit).

## Run

```bash
python -m titanic.run --trials 120 --seed 42          # ~3 min on 1 CPU core
python -m titanic.verify_determinism --trials 30 --flaml   # A == B, A != other seed
python -m titanic.flaml_baseline --max-iter 60        # AutoML comparator (FLAML, fixed iterations)
python -m pytest tests/test_titanic.py
```

## Results

Filled in from `out/titanic/results.json` of the committed run; see the section below.
