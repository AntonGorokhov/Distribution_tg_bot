"""Canvas environment: rows = raw columns (feature cells) + one row per hyper-parameter.

Painting a feature cell = "apply op to column"; painting a hyper-parameter cell = choose that
value (at most one per row). STOP ends the trajectory. Invalid cells are masked for ever.
Because the constraints are order-independent, every painted cell could have been painted
last, so the uniform backward policy of gfn.env applies unchanged: sum log P_B = -lgamma(n+1).
"""
from __future__ import annotations

import torch

from .ops import COLUMNS, OPS, valid_mask
from .reward import HPARAMS

H = 16            # rows: 10 feature rows + 4 hparam rows + 2 padding rows
W = len(OPS)      # 8
N_FEAT_ROWS = len(COLUMNS)
HP_ROW0 = N_FEAT_ROWS


def static_valid() -> torch.Tensor:
    """(H, W) boolean mask of cells that may ever be painted."""
    m = torch.zeros(H, W, dtype=torch.bool)
    m[:N_FEAT_ROWS] = torch.from_numpy(valid_mask())
    for r, (_, choices) in enumerate(HPARAMS):
        m[HP_ROW0 + r, :len(choices)] = True
    return m


class CanvasEnv:
    def __init__(self):
        self.H, self.W = H, W
        self.n_actions = H * W + 1
        self.STOP = H * W
        self.valid = static_valid()
        self.max_steps = int(self.valid.sum()) + 1

    def initial_state(self, batch: int) -> torch.Tensor:
        return torch.zeros(batch, 1, H, W)

    def action_mask(self, canvas: torch.Tensor) -> torch.Tensor:
        B = canvas.shape[0]
        c = canvas[:, 0] > 0.5                                   # (B, H, W)
        allowed = self.valid[None].expand(B, -1, -1) & ~c
        hp = c[:, HP_ROW0:HP_ROW0 + len(HPARAMS)].any(-1)        # (B, n_hp) row already chosen
        allowed = allowed.clone()
        allowed[:, HP_ROW0:HP_ROW0 + len(HPARAMS)] &= ~hp[:, :, None]
        return torch.cat([allowed.reshape(B, -1), torch.ones(B, 1, dtype=torch.bool)], dim=1)

    def step(self, canvas: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        out = canvas.clone()
        B = canvas.shape[0]
        paint = action < self.STOP
        out.view(B, -1)[torch.arange(B)[paint], action[paint]] = 1.0
        return out

    @staticmethod
    def log_pb_sum(n_painted: torch.Tensor) -> torch.Tensor:
        return -torch.lgamma(n_painted.float() + 1.0)

    # ------------------------------------------------------------- decoding
    @staticmethod
    def decode(canvas: torch.Tensor) -> tuple[list[tuple[str, str]], dict]:
        """canvas (1, H, W) or (H, W) -> (feature cells, hyper-parameters)."""
        c = canvas.reshape(H, W) > 0.5
        cells = [(COLUMNS[r], OPS[k]) for r in range(N_FEAT_ROWS) for k in range(W) if c[r, k]]
        hp = {}
        for r, (name, choices) in enumerate(HPARAMS):
            row = c[HP_ROW0 + r]
            for k, v in enumerate(choices):
                if row[k]:
                    hp[name] = v
        return cells, hp

    @staticmethod
    def encode(cells, hp) -> torch.Tensor:
        c = torch.zeros(H, W)
        for col, op in cells:
            c[COLUMNS.index(col), OPS.index(op)] = 1.0
        for r, (name, choices) in enumerate(HPARAMS):
            if name in hp:
                c[HP_ROW0 + r, choices.index(hp[name])] = 1.0
        return c

    @staticmethod
    def canvas_key(canvas: torch.Tensor) -> int:
        flat = (canvas.reshape(-1) > 0.5).long()
        return int((flat << torch.arange(flat.numel())).sum())
