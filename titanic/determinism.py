"""Everything that makes a run reproducible lives here, so it can be audited in one place."""
from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import sys

import numpy as np

HASH_SEED = "0"


def reexec_with_fixed_hashseed() -> None:
    """str/set iteration order depends on PYTHONHASHSEED, which can only be set before the
    interpreter starts. Re-exec ourselves once with a pinned value."""
    if os.environ.get("PYTHONHASHSEED") != HASH_SEED:
        env = {**os.environ, "PYTHONHASHSEED": HASH_SEED}
        spec = getattr(sys.modules.get("__main__"), "__spec__", None)
        argv = [sys.executable, "-m", spec.name, *sys.argv[1:]] if spec else [sys.executable, *sys.argv]
        os.execve(sys.executable, argv, env)


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
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "git_commit": commit,
    }
