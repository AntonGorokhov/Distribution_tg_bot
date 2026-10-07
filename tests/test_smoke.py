import torch

from gfn.env import GridEnv
from gfn.evaluate import exact_terminal_distribution
from gfn.trainer import rollout, tb_loss, train
from gfn.unet import UNetPolicy


def test_unet_shapes():
    m = UNetPolicy(base=8)
    out = m(torch.zeros(5, 1, 4, 4))
    assert out.shape == (5, 17)


def test_rollout_and_loss():
    env, m = GridEnv(size=4), UNetPolicy(base=8)
    r = rollout(m, env, batch=16, eps=0.2)
    assert r.canvas.shape == (16, 1, 4, 4)
    assert torch.isfinite(r.log_pf).all()
    loss = tb_loss(r, torch.tensor(0.0), env)
    assert torch.isfinite(loss)


def test_exact_distribution_sums_to_one():
    env, m = GridEnv(size=4), UNetPolicy(base=8)
    p = exact_terminal_distribution(m, env)
    assert abs(p.sum().item() - 1.0) < 1e-6


def test_short_training_runs():
    env, m = GridEnv(size=4), UNetPolicy(base=8)
    log_z, hist = train(env, m, iters=5, batch=8, log_every=100)
    assert len(hist) == 5
