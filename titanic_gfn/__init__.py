"""Titanic feature-engineering + hyper-parameter search driven by a GFlowNet.

The GFlowNet paints a canvas (raw column x operation, plus hyper-parameter rows); every
painted cell is a hypothesis "apply op to column". Reward = exp(beta * (cv_acc - base)).
Thread pools are pinned to one thread before numpy loads, as in the `titanic` package.
"""
import titanic  # noqa: F401  (pins OMP/BLAS threads)
