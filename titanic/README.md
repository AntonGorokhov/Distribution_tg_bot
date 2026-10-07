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

Committed run: `python -m titanic.run --trials 120 --seed 42` (138 s on one core).
Outputs are in `titanic/results/`.

| | |
|---|---|
| best CV accuracy (2x5-fold) | **0.8417** (fold std 0.0199, trial #84) |
| best model | lgbm: n_estimators=75, learning_rate=0.1782, num_leaves=13, min_child_samples=9, subsample=0.9344, colsample_bytree=0.7785, reg_lambda=3.1391 |
| hypotheses picked | title=True, family=False, deck=False, ticket_group=False, fare_log=True, age_bins=True, age_impute=median |
| submission sha256 | `0d8408a169ffc97e…` (predicted survival rate 0.347) |
| FLAML comparator (60 iterations) | lgbm: 0.8384 by FLAML's own CV on pre-fitted features, **0.8277** re-scored under the per-fold protocol used above |

Per model family over the 120 trials:

| model | trials | best CV | mean CV |
|---|---|---|---|
| lgbm | 61 | 0.8417 | 0.8326 |
| hgb | 19 | 0.8378 | 0.8201 |
| rf | 12 | 0.8322 | 0.8233 |
| svc | 18 | 0.8316 | 0.7561 |
| logreg | 10 | 0.8277 | 0.8183 |

**Selection noise.** The fold std of the best trial is 0.020; 77 of 120 trials lie
within one fold std of the best and 29 within one naive standard error
(0.0063). The search identifies a plateau of LightGBM configurations at 0.83–0.84, not a
unique winner; the reported maximum over 120 trials is optimistic by roughly 0.005–0.01
relative to re-scoring the same configuration on fresh CV splits. Treat "hypotheses picked"
as one member of that plateau.

Determinism check (`python -m titanic.verify_determinism --trials 30 --flaml`), output in
`titanic/results/determinism_check.txt`:

```
optuna  seed 42 run A vs B: trials identical=True, submission identical=True; best acc 0.83894 vs 0.83894
optuna  seed 43 differs from seed 42 (sampled params and values): True (best acc 0.82829)
flaml   seed 42 run A vs B: config identical=True, submission identical=True; best acc 0.83839 vs 0.83839
DETERMINISTIC
```

Kaggle's public leaderboard was not queried from this environment (no Kaggle access);
pipelines in this CV range typically score 0.77–0.80 on the public test split.
