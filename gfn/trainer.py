"""Trajectory-Balance training loop (Malkin et al., 2022) with a learned log Z."""
from __future__ import annotations

import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .env import GridEnv
from .unet import UNetPolicy


@dataclass
class Rollout:
    canvas: torch.Tensor     # (B,1,H,W) terminal states
    log_pf: torch.Tensor     # (B,) sum of log P_F along trajectory
    log_r: torch.Tensor      # (B,) log reward of terminal state
    n_painted: torch.Tensor  # (B,)


def masked_log_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return F.log_softmax(logits.masked_fill(~mask, float("-inf")), dim=-1)


@torch.no_grad()
def _sample_actions(log_probs: torch.Tensor, mask: torch.Tensor, eps: float) -> torch.Tensor:
    probs = log_probs.exp()
    if eps > 0:
        uniform = mask.float() / mask.float().sum(-1, keepdim=True)
        probs = (1 - eps) * probs + eps * uniform
    return torch.multinomial(probs, 1).squeeze(1)


def rollout(model: UNetPolicy, env: GridEnv, batch: int, eps: float = 0.0) -> Rollout:
    canvas = env.initial_state(batch)
    done = torch.zeros(batch, dtype=torch.bool)
    log_pf = torch.zeros(batch)
    for _ in range(env.H * env.W + 1):
        mask = env.action_mask(canvas)
        logp = masked_log_softmax(model(canvas), mask)
        a = _sample_actions(logp.detach(), mask, eps)
        log_pf = log_pf + torch.where(done, torch.zeros(batch), logp.gather(1, a[:, None]).squeeze(1))
        a = torch.where(done, torch.full_like(a, env.STOP), a)
        canvas = env.step(canvas, a)
        done = done | (a == env.STOP)
        if bool(done.all()):
            break
    n = canvas.reshape(batch, -1).sum(1)
    return Rollout(canvas, log_pf, env.log_reward(canvas), n)


def tb_loss(r: Rollout, log_z: torch.Tensor, env: GridEnv) -> torch.Tensor:
    log_pb = env.log_pb_sum(r.n_painted)
    residual = log_z + r.log_pf - r.log_r - log_pb
    return residual.pow(2).mean()


def train(env: GridEnv, model: UNetPolicy, iters: int = 1500, batch: int = 64,
          lr: float = 2e-3, lr_z: float = 1e-1, eps: float = 0.1,
          log_every: int = 100, log_z_init: float = 0.0):
    log_z = torch.nn.Parameter(torch.tensor(log_z_init))
    opt = torch.optim.Adam([
        {"params": model.parameters(), "lr": lr},
        {"params": [log_z], "lr": lr_z},
    ])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters)
    history = []
    t0 = time.time()
    for it in range(1, iters + 1):
        r = rollout(model, env, batch, eps=eps)
        loss = tb_loss(r, log_z, env)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        sched.step()
        history.append((it, loss.item(), log_z.item(), r.log_r.mean().item()))
        if it % log_every == 0 or it == 1:
            print(f"it {it:5d} | loss {loss.item():8.4f} | logZ {log_z.item():7.3f} "
                  f"| mean logR {r.log_r.mean().item():7.3f} | {time.time() - t0:5.1f}s")
    return log_z.detach(), history
