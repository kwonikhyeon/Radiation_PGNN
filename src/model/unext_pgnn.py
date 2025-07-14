"""
UNeXt-PGNN (v3)
  • Partial-Conv Stem           : 관측치 없는 픽셀 무시
  • ASPP Bottleneck (기본)      : d=1/4/8 DWConv 병렬 → 전역 receptive field
  • 선택적 Swin Bottleneck      : use_swin=True 일 때만 timm 필요
  • CoordConv / deep-supervision / base 채널 토글
"""

from __future__ import annotations
import torch, torch.nn as nn, torch.nn.functional as F
import torch.cuda.amp as amp

# ════════════════════════════════════════════════════════════
# 0. Partial Convolution (Liu et al., CVPR’18)
# ════════════════════════════════════════════════════════════

class PartialConv2d(nn.Conv2d):
    def forward(self, x, mask):
        # 파라미터와 동일한 dtype·device 확보
        pdtype, pdev = self.weight.dtype, self.weight.device
        x_f   = x.to(dtype=pdtype, device=pdev)
        m_f   = mask.to(dtype=pdtype, device=pdev)

        with amp.autocast(enabled=False):              # FP32 연산
            k = torch.ones(1, 1, *self.kernel_size, device=pdev, dtype=pdtype)
            slide = F.conv2d(m_f, k, stride=self.stride,
                             padding=self.padding, dilation=self.dilation)
            slide = torch.clamp(slide, 1e-6)
            out   = super().forward(x_f * m_f) / slide

        new_mask = (slide > 0).to(mask.dtype)
        return out.to(x.dtype), new_mask               # 원래 AMP dtype 복원

# ════════════════════════════════════════════════════════════
# 1. U-NeXt 기본 블록
# ════════════════════════════════════════════════════════════
class UNeXtBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dw  = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.pw1 = nn.Conv2d(dim, dim*4, 1)
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(dim*4, dim, 1)
        self.bn  = nn.BatchNorm2d(dim)

    def forward(self, x):
        h = self.dw(x)
        h = self.act(self.pw1(h))
        h = self.bn(self.pw2(h))
        return x + h

class Down(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_c), nn.GELU(),
            UNeXtBlock(out_c)
        )
    def forward(self, x): return self.seq(x)

class Up(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up     = nn.ConvTranspose2d(in_c, out_c, 2, stride=2)
        self.block  = UNeXtBlock(out_c*2)
        self.reduce = nn.Conv2d(out_c*2, out_c, 1)
    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], 1)
        x = self.block(x)
        return self.reduce(x)

# ════════════════════════════════════════════════════════════
# 2. ASPP Bottleneck (Dilated DWConv ‖ concat)
# ════════════════════════════════════════════════════════════
class ASPP_Bottleneck(nn.Module):
    def __init__(self, dim, dilations=(1,4,8)):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Conv2d(dim, dim, 3, padding=d, dilation=d, groups=dim)
            for d in dilations
        ])
        self.point = nn.Conv2d(dim*len(dilations), dim, 1)
        self.act   = nn.GELU()
        self.bn    = nn.BatchNorm2d(dim)
    def forward(self, x):
        feats = [b(x) for b in self.branches]
        out   = self.point(torch.cat(feats, 1))
        out   = self.bn(self.act(out))
        return x + out                    # residual

# ════════════════════════════════════════════════════════════
# 3. 메인 네트워크
# ════════════════════════════════════════════════════════════
class UNEXT_PGNN(nn.Module):
    def __init__(self, in_ch=2, out_ch=1, *,
                 base=32, deep_supervision=True,
                 use_coord=False):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.use_coord        = use_coord

        in_ch += 2 if use_coord else 0
        dims = [base, base*2, base*4, base*8]

        # Stem : Partial-Conv
        self.stem = PartialConv2d(in_ch, base, 3, padding=1)

        # Encoder
        self.enc1  = UNeXtBlock(base)
        self.down1 = Down(dims[0], dims[1])
        self.down2 = Down(dims[1], dims[2])
        self.down3 = Down(dims[2], dims[3])

        # Bottleneck
        self.bottle = ASPP_Bottleneck(dims[3])

        # Decoder
        self.up3  = Up(dims[3], dims[2])
        self.up2  = Up(dims[2], dims[1])
        self.up1  = Up(dims[1], dims[0])

        self.head = nn.Conv2d(dims[0], out_ch, 1)
        if deep_supervision:
            self.aux2 = nn.Conv2d(dims[1], out_ch, 1)   # 128×128
            self.aux3 = nn.Conv2d(dims[2], out_ch, 1)   # 64×64

    # Coord-tensor 생성
    @staticmethod
    def _coord(H, W, device):
        yy, xx = torch.meshgrid(
            torch.linspace(0,1,H,device=device),
            torch.linspace(0,1,W,device=device), indexing="ij")
        return torch.stack([xx, yy])

    # ─ Forward ─
    def forward(self, x):
        N, _, H, W = x.shape
        if self.use_coord:
            coord = self._coord(H, W, x.device).unsqueeze(0).repeat(N,1,1,1)
            x = torch.cat([x, coord], 1)

        mask = x[:,1:2]
        x, mask = self.stem(x, mask)
        s0 = F.gelu(x)

        e1 = self.enc1(s0)
        e2 = self.down1(e1)
        e3 = self.down2(e2)
        e4 = self.down3(e3)
        b  = self.bottle(e4)

        d3 = self.up3(b, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)
        out = self.head(d1)

        if self.deep_supervision:
            aux2 = self.aux2(d2)
            aux3 = self.aux3(d3)
            return out, aux2, aux3
        else:
            dummy = torch.zeros_like(out)
            return out, dummy, dummy