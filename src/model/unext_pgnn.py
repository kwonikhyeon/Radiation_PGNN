"""
UNeXt-PGNN : CoordConv + U-NeXt backbone + Deep Supervision(optional)

입력  : (N, 2, H, W)   [r_meas, mask]
출력  : (N, 1, H, W)   dense radiation field
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────
# Basic blocks
# ─────────────────────────────────────
class UNeXtBlock(nn.Module):
    """DWConv(7×7) → PWConv×2 (MLP) + residual"""
    def __init__(self, dim: int):
        super().__init__()
        self.dwconv  = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.pwconv1 = nn.Conv2d(dim, dim * 4, 1)
        self.act     = nn.GELU()
        self.pwconv2 = nn.Conv2d(dim * 4, dim, 1)
        self.bn      = nn.BatchNorm2d(dim)

    def forward(self, x):
        h = self.dwconv(x)
        h = self.pwconv1(h)
        h = self.act(h)
        h = self.pwconv2(h)
        h = self.bn(h)
        return x + h


class Down(nn.Module):
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_c),
            nn.GELU(),
            UNeXtBlock(out_c),
        )

    def forward(self, x):  # H→H/2
        return self.block(x)


class Up(nn.Module):
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        # ① 업샘플
        self.up = nn.ConvTranspose2d(in_c, out_c, 2, stride=2)
        # ② concat 이후 처리용 블록
        self.block  = UNeXtBlock(out_c * 2)
        # ③ ***채널 축소*** : out_c*2 → out_c
        self.reduce = nn.Conv2d(out_c * 2, out_c, 1)

    def forward(self, x, skip):
        x = self.up(x)                  # (N, out_c, H*2, W*2)
        x = torch.cat([x, skip], dim=1) # (N, out_c*2, …)
        x = self.block(x)               # still out_c*2
        x = self.reduce(x)              # (N, out_c, …) ← ↓축소
        return x                        # (N, out_c, H*2, W*2) ← 업샘플링


# ─────────────────────────────────────
# Main network
# ─────────────────────────────────────
class UNEXT_PGNN(nn.Module):
    """
    U-NeXt variant for PGNN
    Args
    ----
    in_ch : 2  (r_meas, mask)
    base  : base channel width (default 32)
    use_coord : add (x,y) CoordConv channels
    """
    def __init__(self, in_ch: int = 2, out_ch: int = 1,
                 base: int = 32, use_coord: bool = True):
        super().__init__()
        self.use_coord = use_coord
        c_in = in_ch + (2 if use_coord else 0)
        dims = [base, base * 2, base * 4, base * 8]  # 32/64/128/256

        # Encoder
        self.stem = nn.Conv2d(c_in, base, 3, padding=1)
        self.enc1 = UNeXtBlock(base)
        self.down1 = Down(dims[0], dims[1])
        self.down2 = Down(dims[1], dims[2])
        self.down3 = Down(dims[2], dims[3])

        # Bottleneck
        self.bottle = UNeXtBlock(dims[3])

        # Decoder
        self.up3 = Up(dims[3], dims[2])
        self.up2 = Up(dims[2], dims[1])
        self.up1 = Up(dims[1], dims[0])

        # Heads (main + deep supervision)
        self.head = nn.Conv2d(dims[0], out_ch, 1)
        self.aux2 = nn.Conv2d(dims[1], out_ch, 1)   # 128×128
        self.aux3 = nn.Conv2d(dims[2], out_ch, 1)   # 64×64

    # ──────────────
    @staticmethod
    def _coord_tensor(H: int, W: int, device: torch.device):
        yy, xx = torch.meshgrid(
            torch.linspace(0, 1, H, device=device),
            torch.linspace(0, 1, W, device=device), indexing="ij")
        return torch.stack([xx, yy])  # (2,H,W)

    # ──────────────
    def forward(self, x):
        N, _, H, W = x.shape
        if self.use_coord:
            coord = self._coord_tensor(H, W, x.device).unsqueeze(0).repeat(N, 1, 1, 1)
            x = torch.cat([x, coord], dim=1)        # (N,4,H,W)

        s0 = F.gelu(self.stem(x))                   # 256×256
        e1 = self.enc1(s0)                         # skip 1

        e2 = self.down1(e1)                        # 128×128
        e3 = self.down2(e2)                        # 64×64
        e4 = self.down3(e3)                        # 32×32

        b  = self.bottle(e4)

        d3 = self.up3(b, e3)                       # 64×64
        d2 = self.up2(d3, e2)                      # 128×128
        d1 = self.up1(d2, e1)                      # 256×256

        out  = self.head(d1)
        # aux2 = F.interpolate(self.aux2(d2), scale_factor=2, mode="bilinear", align_corners=False)
        # aux3 = F.interpolate(self.aux3(d3), scale_factor=4, mode="bilinear", align_corners=False)
        aux2 = self.aux2(d2)   # 128×128
        aux3 = self.aux3(d3)   #  64×64
        return out, aux2, aux3                     # deep-sup outputs