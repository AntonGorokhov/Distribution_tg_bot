"""Everything that makes a run reproducible lives here, so it can be audited in one place."""
from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import sys
import warnings

import numpy as np

HASH_SEED = "0"
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


def reexec_with_fixed_hashseed() -> None:
    """str/set iteration order depends on PYTHONHASHSEED, and BLAS thread counts on OMP_* vars,
    both of which must be set before the interpreter / numpy start. Entry points call this
    under `if __name__ == "__main__"` and get re-executed once with the pinned values."""
    pinned = sys.flags.hash_randomization == 0 and all(os.environ.get(v) == "1" for v in THREAD_VARS)
    if pinned:
        return
    argv = getattr(sys, "orig_argv", None)
    if not argv:
        warnings.warn("cannot re-exec with pinned PYTHONHASHSEED/thread vars; set them in the environment", RuntimeWarning)
        return
    env = {**os.environ, "PYTHONHASHSEED": HASH_SEED, **{v: "1" for v in THREAD_VARS}}
    os.execve(sys.executable, list(argv), env)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def md5_file(path: str) -> str:
    return hashlib.md5(open(path, "rb").read()).hexdigest()


def config_fingerprint(obj) -> str:
    """Stable hash of a (nested) config: sorted keys, repr floats."""
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def environment_info() -> dict:
    import lightgbm, optuna, pandas, sklearn
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False).stdout.strip()
    except OSError:
        commit = ""
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__, "pandas": pandas.__version__, "sklearn": sklearn.__version__,
        "optuna": optuna.__version__, "lightgbm": lightgbm.__version__,
        "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
        "hash_randomization": sys.flags.hash_randomization,
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "git_commit": commit,
        "git_dirty": bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, check=False).stdout.strip()),
    }
