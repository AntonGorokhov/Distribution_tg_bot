"""Titanic (Kaggle) with a fully deterministic hypothesis + hyper-parameter search.

Importing this package pins every thread pool to one thread BEFORE numpy/BLAS load,
so floating-point reductions are reproducible across runs.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
