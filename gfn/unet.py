"""Tiny 2-level U-Net that maps a canvas to per-pixel action logits + a STOP logit."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(1, cout), nn.SiLU(),
        nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(1, cout), nn.SiLU(),
    )


class UNetPolicy(nn.Module):
    """Input (B,1,H,W) -> logits (B, H*W + 1). Last column is STOP.

    encoder: H x W (c) -> H/2 x W/2 (2c) -> bottleneck
    decoder: upsample + skip concat -> H x W -> 1x1 conv -> pixel logits
    STOP head: global-avg-pooled bottleneck -> MLP -> scalar
    """

    def __init__(self, in_ch: int = 1, base: int = 32):
        super().__init__()
        self.enc1 = _block(in_ch, base)
        self.enc2 = _block(base, base * 2)
        self.mid = _block(base * 2, base * 2)
        self.dec1 = _block(base * 2 + base, base)
        self.head = nn.Conv2d(base, 1, 1)
        self.stop = nn.Sequential(nn.Linear(base * 2, base), nn.SiLU(), nn.Linear(base, 1))
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)

    def forward_with_activations(self, canvas: torch.Tensor) -> tuple[torch.Tensor, dict]:
        B, _, H, W = canvas.shape
        e1 = self.enc1(canvas)                       # (B, c, H, W)
        e2 = self.enc2(F.max_pool2d(e1, 2))          # (B, 2c, H/2, W/2)
        m = self.mid(e2)                             # (B, 2c, H/2, W/2)
        up = F.interpolate(m, size=(H, W), mode="nearest")
        d1 = self.dec1(torch.cat([up, e1], dim=1))   # (B, c, H, W)
        pix = self.head(d1).reshape(B, H * W)
        stop = self.stop(m.mean(dim=(2, 3)))         # (B, 1)
        logits = torch.cat([pix, stop], dim=1)
        return logits, {"e1": e1, "e2": e2, "mid": m, "d1": d1, "pix": pix, "stop": stop}

    def forward(self, canvas: torch.Tensor) -> torch.Tensor:
        return self.forward_with_activations(canvas)[0]
