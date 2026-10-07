# GFlowNet on synthetic data with a U-Net policy (CPU)

Minimal, from-scratch GFlowNet (trajectory balance) that learns to sample binary
`H x W` images proportionally to a synthetic multimodal reward. The forward policy
is a small 2-level U-Net: the canvas goes in, a logit per pixel comes out
(plus a STOP logit from the bottleneck). Everything runs on CPU in ~4 minutes.

## Task

* **State**: binary canvas, starts empty.
* **Action**: paint one unpainted pixel, or STOP. The same canvas is reachable by
  many orderings, so the state graph is a DAG (the actual GFlowNet setting).
* **Reward** (synthetic): `R(x) = floor + sum_k exp(-hamming(x, proto_k) / tau)`
  with `K=3` random prototype images. Three sharp modes plus a long tail.
* **Backward policy**: fixed uniform over painted pixels, so
  `sum log P_B = -log(n!)` for a trajectory ending with `n` painted pixels.
* **Loss**: trajectory balance with a learned `log Z`.

## Why it is verifiable

For a `4 x 4` grid all `2^16 = 65536` terminal states are enumerable, so
`gfn/evaluate.py` computes the **exact** model distribution `P_T(x)` by dynamic
programming over the DAG and compares it with `R(x)/Z`. No sampling noise.

## Run

```bash
pip install -r requirements.txt
python train.py            # 4x4 grid, 1500 iters, ~4 min on 4 CPU cores
python -m pytest tests     # smoke tests, ~3 s
```

Default run result:

| metric                                | value            |
|---------------------------------------|------------------|
| TV distance `P_model` vs `R/Z`        | 0.045            |
| corr(`P_model`, `R/Z`) over 65536 states | 0.9996         |
| log Z learned / true                  | 2.821 / 2.842    |
| mass within 1 px of a prototype (model / target) | 0.402 / 0.407 |
| per-mode hit rate (3 modes, target 0.058 each) | 0.057 / 0.058 / 0.059 |

Plots go to `out/`: `training.png`, `samples.png`, `target_vs_model.png`, `scatter.png`.

Useful flags: `--size 8` (no exact eval, sample metrics only), `--iters`, `--batch`,
`--tau` (mode sharpness), `--floor`, `--eps` (exploration), `--base` (U-Net width).

## Layout

```
gfn/env.py       environment, reward, exact enumeration
gfn/unet.py      U-Net policy -> (H*W + 1) logits
gfn/trainer.py   vectorised rollouts + trajectory-balance loss + training loop
gfn/evaluate.py  exact DP distribution, sample-based metrics
train.py         CLI entry point
plot.py          diagnostic figures
tests/           smoke tests
```
