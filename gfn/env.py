"""Synthetic environment: build a binary H x W image one pixel at a time.

State  : binary canvas (B, 1, H, W).
Actions: paint one of the H*W unpainted pixels, or STOP (index H*W).
Reward : mixture of bumps around K random prototype images (synthetic, multimodal):
         R(x) = floor + sum_k exp(-hamming(x, proto_k) / tau)

Because the same canvas is reachable by many orderings, the state graph is a DAG,
not a tree - the setting where GFlowNets are actually needed.
Backward policy is fixed uniform over painted pixels: P_B(s' -> s) = 1 / n_painted(s'),
so the sum of log P_B along a trajectory ending with n painted pixels is -lgamma(n + 1).
"""
from __future__ import annotations

import math

import torch


class GridEnv:
    def __init__(self, size: int = 4, n_protos: int = 3, tau: float = 0.4,
                 reward_floor: float = 1e-4, seed: int = 0):
        self.H = self.W = size
        self.n_actions = size * size + 1  # pixels + STOP
        self.STOP = size * size
        self.tau = tau
        self.reward_floor = reward_floor
        g = torch.Generator().manual_seed(seed)
        self.protos = (torch.rand(n_protos, size, size, generator=g) < 0.5).float()

    # ------------------------------------------------------------------ reward
    def log_reward(self, canvas: torch.Tensor) -> torch.Tensor:
        """canvas: (B, 1, H, W) or (B, H, W) binary -> (B,) log R."""
        x = canvas.reshape(canvas.shape[0], -1)                 # (B, HW)
        p = self.protos.reshape(self.protos.shape[0], -1)       # (K, HW)
        hamming = (x[:, None, :] - p[None, :, :]).abs().sum(-1)  # (B, K)
        bumps = torch.exp(-hamming / self.tau).sum(-1)
        return torch.log(bumps + self.reward_floor)

    # --------------------------------------------------------------- dynamics
    def initial_state(self, batch: int) -> torch.Tensor:
        return torch.zeros(batch, 1, self.H, self.W)

    def action_mask(self, canvas: torch.Tensor) -> torch.Tensor:
        """True = allowed. Painted pixels are forbidden; STOP is always allowed."""
        flat = canvas.reshape(canvas.shape[0], -1) < 0.5
        stop = torch.ones(canvas.shape[0], 1, dtype=torch.bool)
        return torch.cat([flat, stop], dim=1)

    def step(self, canvas: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Apply paint actions in-place-free; STOP leaves the canvas untouched."""
        out = canvas.clone()
        B = canvas.shape[0]
        paint = action < self.STOP
        idx = torch.arange(B)[paint]
        a = action[paint]
        out.view(B, -1)[idx, a] = 1.0
        return out

    @staticmethod
    def log_pb_sum(n_painted: torch.Tensor) -> torch.Tensor:
        """Sum of log P_B over a trajectory that ends with n painted pixels."""
        return -torch.lgamma(n_painted.float() + 1.0)

    # ------------------------------------------------- exact enumeration (small)
    def all_states(self) -> torch.Tensor:
        """Every binary canvas, index i <-> bit pattern i. Only sane for H*W <= 16."""
        n = self.H * self.W
        assert n <= 20, "too many states to enumerate"
        ids = torch.arange(2 ** n)
        bits = (ids[:, None] >> torch.arange(n)[None, :]) & 1
        return bits.float().reshape(-1, 1, self.H, self.W)

    def exact_log_z(self) -> float:
        return torch.logsumexp(self.log_reward(self.all_states()), 0).item()

    def __repr__(self) -> str:
        return f"GridEnv(size={self.H}, protos={self.protos.shape[0]}, tau={self.tau})"
