# src/model/partial_conv.py
import torch, torch.nn as nn, torch.nn.functional as F
from typing import Tuple


class PartialConv2d(nn.Module):
    """Partial Convolution (NVIDIA 2018) : returns (out, new_mask)."""
    def __init__(self, in_c: int, out_c: int, k: int = 3, s: int = 1):
        super().__init__()
        p = k // 2
        self.conv = nn.Conv2d(in_c, out_c, k, s, padding=p, bias=False)
        self.bn   = nn.BatchNorm2d(out_c)
        self.act  = nn.LeakyReLU(0.1, inplace=True)

        self.register_buffer("weight_mask", torch.ones(1, 1, k, k))
        self.stride   = s
        self.padding  = p

    @torch.no_grad()
    def _update_mask(self, mask: torch.Tensor) -> torch.Tensor:
        update = F.conv2d(mask, self.weight_mask,
                          stride=self.stride, padding=self.padding) > 0
        return update.float()

    def forward(self, x: torch.Tensor, mask: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x * mask
        out = self.act(self.bn(self.conv(x)))
        new_mask = self._update_mask(mask)
        return out, new_mask

class StrictPartialConv2d(nn.Module):
    """Strict Partial Convolution: measured 위치는 그대로 통과, 미측정 위치만 업데이트."""
    def __init__(self, in_c: int, out_c: int, k: int = 3, s: int = 1):
        super().__init__()
        p = k // 2
        self.conv = nn.Conv2d(in_c, out_c, k, s, padding=p, bias=False)
        self.bn   = nn.BatchNorm2d(out_c)
        self.act  = nn.LeakyReLU(0.1, inplace=True)

        self.register_buffer("weight_mask", torch.ones(1, 1, k, k))
        self.stride  = s
        self.padding = p

    @torch.no_grad()
    def _update_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """Conv로 업데이트할 마스크 생성 (binary)."""
        update = F.conv2d(mask, self.weight_mask,
                          stride=self.stride, padding=self.padding) > 0
        return update.float()

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        # (1) measured 위치만 입력으로 사용
        x_masked = x * mask                        # (B, C_in, H, W)
        out_full = self.conv(x_masked)             # (B, C_out, H, W)
        out_full = self.act(self.bn(out_full))     # norm + act

        # (2) mask broadcasting: (B, 1, H, W) → (B, C_out, H, W)
        mask_exp = mask.expand_as(out_full)

        # (3) strict 처리: measured 위치는 원본 그대로 유지
        x_masked_up = x_masked
        if x_masked.shape[1] != out_full.shape[1]:
            # 예외 처리: x와 out_full의 채널 수가 다르면, measured는 0으로 둠
            x_masked_up = torch.zeros_like(out_full)
        else:
            x_masked_up = x_masked

        out = out_full * (1 - mask_exp) + x_masked_up  # measured = 원본, 나머지 = pred

        new_mask = self._update_mask(mask)
        return out, new_mask