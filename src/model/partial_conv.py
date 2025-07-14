# src/model/partial_conv.py
import torch, torch.nn as nn, torch.nn.functional as F

class PartialConv2d(nn.Conv2d):
    def forward(self, x, mask):
        with torch.no_grad():
            # mask 1→valid, 0→missing
            kernel = torch.ones(1, 1, self.kernel_size[0], self.kernel_size[1],
                                device=x.device)
            slide_in = F.conv2d(mask, kernel, stride=self.stride,
                                padding=self.padding, dilation=self.dilation)
            slide_in = torch.clamp(slide_in, min=1e-6)  # avoid /0
        out = super().forward(x * mask)
        out = out / slide_in               # 평균 보정
        new_mask = (slide_in > 0).float()  # 업데이트
        return out, new_mask