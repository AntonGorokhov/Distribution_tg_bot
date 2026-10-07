"""Exact and sample-based evaluation of a trained GFlowNet on the small grid."""
from __future__ import annotations

import torch

from .env import GridEnv
from .trainer import masked_log_softmax, rollout
from .unet import UNetPolicy


@torch.no_grad()
def exact_terminal_distribution(model: UNetPolicy, env: GridEnv, chunk: int = 4096) -> torch.Tensor:
    """P_T(x) for every x via dynamic programming over the DAG, ordered by popcount.

    F_in(s0) = 1 ; F_in(s) = sum_{b in s} F_in(s \\ b) * P_F(paint b | s \\ b)
    P_T(x)   = F_in(x) * P_F(STOP | x)
    """
    n = env.H * env.W
    states = env.all_states()
    N = states.shape[0]
    pf = torch.empty(N, n + 1)
    for i in range(0, N, chunk):
        s = states[i:i + chunk]
        pf[i:i + chunk] = masked_log_softmax(model(s), env.action_mask(s)).exp()
    ids = torch.arange(N)
    popcount = ((ids[:, None] >> torch.arange(n)[None, :]) & 1).sum(1)
    f_in = torch.zeros(N, dtype=torch.float64)
    f_in[0] = 1.0
    pf = pf.double()
    for level in range(1, n + 1):
        idx = ids[popcount == level]
        acc = torch.zeros(idx.shape[0], dtype=torch.float64)
        for b in range(n):
            has = ((idx >> b) & 1).bool()
            parent = idx[has] ^ (1 << b)
            acc[has] += f_in[parent] * pf[parent, b]
        f_in[idx] = acc
    return f_in * pf[:, n]


@torch.no_grad()
def exact_metrics(model: UNetPolicy, env: GridEnv, log_z: torch.Tensor) -> dict:
    states = env.all_states()
    log_r = env.log_reward(states).double()
    true_log_z = torch.logsumexp(log_r, 0)
    p_star = (log_r - true_log_z).exp()
    p_model = exact_terminal_distribution(model, env)
    return {
        "mass_check": p_model.sum().item(),                      # should be ~1
        "l1": (p_model - p_star).abs().sum().item(),             # in [0, 2]
        "tv": 0.5 * (p_model - p_star).abs().sum().item(),
        "corr": torch.corrcoef(torch.stack([p_model, p_star]))[0, 1].item(),
        "true_log_z": true_log_z.item(),
        "learned_log_z": log_z.item(),
        "p_model": p_model,
        "p_star": p_star,
    }


@torch.no_grad()
def sample_metrics(model: UNetPolicy, env: GridEnv, n: int = 20000, batch: int = 2000) -> dict:
    canvases, log_rs = [], []
    for _ in range(n // batch):
        r = rollout(model, env, batch, eps=0.0)
        canvases.append(r.canvas); log_rs.append(r.log_r)
    canvas = torch.cat(canvases); log_r = torch.cat(log_rs)
    flat = canvas.reshape(canvas.shape[0], -1).long()
    ids = (flat * (2 ** torch.arange(flat.shape[1]))[None]).sum(1)
    protos = env.protos.reshape(env.protos.shape[0], -1)
    hamming = (canvas.reshape(canvas.shape[0], -1)[:, None] - protos[None]).abs().sum(-1)
    return {
        "mean_reward": log_r.exp().mean().item(),
        "unique_states": ids.unique().numel(),
        "frac_within_1_of_proto": (hamming.min(1).values <= 1).float().mean().item(),
        "proto_hits": [(hamming[:, k] == 0).float().mean().item() for k in range(protos.shape[0])],
        "canvas": canvas,
    }
