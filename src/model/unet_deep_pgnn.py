# ──────────────────────────────────────────────────────────────
# src/model/unet_pgnn.py
#   • PartialConv 기반 PGNN-UNet (mask-aware)
#   • 출력: Softplus ≥ 0
# ──────────────────────────────────────────────────────────────
from __future__ import annotations
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse, pathlib, sys
from pathlib import Path

ROOT = pathlib.Path(__file__).resolve().parents[1]  # …/src
sys.path.append(str(ROOT))
from model.partial_conv import StrictPartialConv2d as PartialConv2d

# -------------------------------------------------------------
# 2. U-Net Blocks
# -------------------------------------------------------------
class DownBlock(nn.Module):
    """PartialConv → MaxPool; returns 4-tuple."""
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.pconv = PartialConv2d(in_c, out_c)
        self.pool  = nn.MaxPool2d(2)

    def forward(self, x, m):
        x, m = self.pconv(x, m)
        xp, mp = self.pool(x), self.pool(m)
        return x, m, xp, mp                   # feat, mask, pooled_feat, pooled_mask

class UpBlock(nn.Module):
    """Up-conv + skip + PartialConv (mask 1-ch)."""
    def __init__(self, in_c: int, skip_c: int, out_c: int):
        super().__init__()
        self.up_feat   = nn.ConvTranspose2d(in_c, out_c, 2, stride=2)
        self.pconv     = PartialConv2d(out_c + skip_c, out_c)
        self.up_mask   = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, x, m, skip_x, skip_m):
        x = self.up_feat(x)
        m = self.up_mask(m)

        # size alignment (if needed)
        if x.shape[-2:] != skip_x.shape[-2:]:
            diffY = skip_x.size(2) - x.size(2)
            diffX = skip_x.size(3) - x.size(3)
            pad   = [diffX // 2, diffX - diffX // 2,
                     diffY // 2, diffY - diffY // 2]
            x = F.pad(x, pad);  m = F.pad(m, pad)

        x = torch.cat([x, skip_x], 1)   # feature concat
        m = torch.max(m, skip_m)        # mask union (1-channel)
        x, m = self.pconv(x, m)
        return x, m

# -------------------------------------------------------------
# 3. PGNN-UNet
# -------------------------------------------------------------
class PGNN_UNet(nn.Module):
    def __init__(self, in_c: int = 6, base: int = 64):
        super().__init__()
        # Encoder
        self.down1  = DownBlock(in_c,       base)         # (H)
        self.down2  = DownBlock(base,       base * 2)     # (H/2)
        self.down3  = DownBlock(base * 2,   base * 4)     # (H/4)
        self.down4  = DownBlock(base * 4,   base * 8)     # (H/8)

        # Bridge
        self.bridge = PartialConv2d(base * 8, base * 16)  # (H/16)

        # Decoder
        self.up4 = UpBlock(in_c=base * 16, skip_c=base * 8, out_c=base * 8)  # (H/8)
        self.up3 = UpBlock(in_c=base * 8,  skip_c=base * 4, out_c=base * 4)  # (H/4)
        self.up2 = UpBlock(in_c=base * 4,  skip_c=base * 2, out_c=base * 2)  # (H/2)
        self.up1 = UpBlock(in_c=base * 2,  skip_c=base,     out_c=base)      # (H)

        self.out_conv = nn.Conv2d(base, 1, kernel_size=1)

    def forward(self, x):
        mask = x[:, 1:2]

        d1, m1, p1, pm1 = self.down1(x,      mask)     # H
        d2, m2, p2, pm2 = self.down2(p1,     pm1)      # H/2
        d3, m3, p3, pm3 = self.down3(p2,     pm2)      # H/4
        d4, m4, p4, pm4 = self.down4(p3,     pm3)      # H/8

        b, mb = self.bridge(p4, pm4)                   # H/16

        u4, mu4 = self.up4(b,  mb,  d4, m4)             # H/8
        u3, mu3 = self.up3(u4, mu4, d3, m3)             # H/4
        u2, mu2 = self.up2(u3, mu3, d2, m2)             # H/2
        u1, mu1 = self.up1(u2, mu2, d1, m1)             # H

        return F.softplus(self.out_conv(u1))

# -------------------------------------------------------------
# 4. Laplacian Physics Loss
# -------------------------------------------------------------
def laplacian_tensor(x: torch.Tensor) -> torch.Tensor:
    """5-point Laplacian with replicate padding."""
    x_p = F.pad(x, (1, 1, 1, 1), mode="replicate")
    lap = (x_p[:, :, 1:-1, :-2] + x_p[:, :, 1:-1, 2:] +
           x_p[:, :, :-2, 1:-1] + x_p[:, :, 2:, 1:-1] - 4 * x)
    return lap

def laplacian_loss(pred: torch.Tensor) -> torch.Tensor:
    """Physics-guided smoothness loss."""
    return (laplacian_tensor(pred) ** 2).mean()
