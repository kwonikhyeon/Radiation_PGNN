#!/usr/bin/env python3
"""
Simplified ConvNeXt PGNN Architecture
기존 특징을 유지하면서 복잡성을 대폭 축소한 버전

Key Simplifications:
1. 4개 브랜치 → 2개 브랜치 (Main + Confidence)
2. 복잡한 suppression → 핵심 3가지만
3. 과도한 정규화 → 필수 요소만
4. 99M → ~50M 파라미터 목표

Preserved Features:
- ConvNeXt backbone
- Mask attention
- Physics-guided constraints
- Anti-blur mechanisms
- Adaptive scaling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models import create_model

class SimplifiedDecoderBlock(nn.Module):
    """단순화된 디코더 블록 - 공간 어텐션 제거, 핵심 기능만"""
    def __init__(self, in_c, skip_c, out_c):
        super().__init__()
        
        # 간단한 skip connection 처리
        self.skip_conv = nn.Conv2d(skip_c, out_c//2, kernel_size=1) if skip_c > out_c else nn.Identity()
        skip_c_proc = out_c//2 if skip_c > out_c else skip_c
        
        # 핵심 convolution만
        self.block = nn.Sequential(
            nn.Conv2d(in_c + skip_c_proc, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        
        if not isinstance(self.skip_conv, nn.Identity):
            skip = self.skip_conv(skip)
        
        x = torch.cat([x, skip], dim=1)
        x = self.block(x)
        return x


class SimplifiedConvNeXtPGNN(nn.Module):
    """
    단순화된 ConvNeXt PGNN
    - 2개 브랜치: Main prediction + Confidence
    - 핵심 suppression 메커니즘만
    - 약 50M 파라미터 목표
    """
    def __init__(self, in_channels=6, decoder_channels=[256, 128, 64, 32], pred_scale=1.0):
        super().__init__()

        # 더 작은 ConvNeXt 모델 사용 (base → small)
        self.encoder = create_model(
            "convnext_small",  # base → small로 변경 (50M → 25M)
            pretrained=False,
            features_only=True,
            in_chans=in_channels
        )

        enc_channels = [f['num_chs'] for f in self.encoder.feature_info]

        # 단순화된 마스크 어텐션 (채널 수 맞춤)
        self.mask_proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, max(ch // 8, 16), kernel_size=3, padding=1),  # //4 → //8
                nn.ReLU(inplace=True),
                nn.Conv2d(max(ch // 8, 16), ch, kernel_size=1)  # 원본 채널 수와 맞춤
            ) for ch in enc_channels
        ])

        # 단순화된 디코더 (채널 수 절반)
        self.up4 = SimplifiedDecoderBlock(enc_channels[3], enc_channels[2], decoder_channels[0])
        self.up3 = SimplifiedDecoderBlock(decoder_channels[0], enc_channels[1], decoder_channels[1])
        self.up2 = SimplifiedDecoderBlock(decoder_channels[1], enc_channels[0], decoder_channels[2])
        self.up1 = nn.Sequential(
            nn.Conv2d(decoder_channels[2], decoder_channels[3], kernel_size=3, padding=1),
            nn.BatchNorm2d(decoder_channels[3]),
            nn.ReLU(inplace=True),
        )

        # 2개 브랜치만: Main + Confidence (기존 4개 → 2개)
        self.main_head = nn.Sequential(
            nn.Conv2d(decoder_channels[3], 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1)
        )
        
        self.confidence_head = nn.Sequential(
            nn.Conv2d(decoder_channels[3], 8, kernel_size=3, padding=1),  # 16 → 8
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 단순화된 적응적 스케일링 (파라미터 수 축소)
        self.adaptive_scale = nn.Parameter(torch.tensor(1.0))
        self.pred_scale = pred_scale

    def forward(self, x):
        mask = x[:, 1:2]
        measured_values = x[:, 0:1]
        distance_map = x[:, 5:6]
        coord_x = x[:, 4:5]
        coord_y = x[:, 3:4]

        # 인코더
        feats = self.encoder(x)
        feats_attn = []

        # 단순화된 마스크 어텐션
        for f, attn_conv in zip(feats, self.mask_proj):
            mask_resized = F.interpolate(mask, size=f.shape[2:], mode='bilinear', align_corners=False)
            mask_feat = attn_conv(mask_resized)
            
            # 간단한 어텐션 (복잡한 공식 제거)
            attention = torch.sigmoid(mask_feat) * 0.2
            feats_attn.append(f + attention)  # 곱셈 대신 덧셈만

        # 디코더
        c1, c2, c3, c4 = feats_attn
        u4 = self.up4(c4, c3)
        u3 = self.up3(u4, c2)
        u2 = self.up2(u3, c1)
        u1 = F.interpolate(u2, scale_factor=2, mode='bilinear', align_corners=False)
        u1 = self.up1(u1)
        out = F.interpolate(u1, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        # 2개 브랜치 예측
        main_pred = self.main_head(out)
        confidence = self.confidence_head(out)
        
        # 단순화된 적응적 스케일링
        B = x.shape[0]
        measured_max = torch.zeros(B, 1, 1, 1, device=x.device)
        
        for b in range(B):
            mask_b = mask[b, 0]
            if mask_b.sum() > 0:
                meas_vals = measured_values[b, 0][mask_b > 0]
                measured_max[b] = meas_vals.max() if len(meas_vals) > 0 else 1.0
            else:
                measured_max[b] = 1.0
        
        # 간단한 스케일링
        scale_factor = 0.5 + 1.5 * torch.sigmoid(self.adaptive_scale * measured_max)
        
        # 기본 예측
        pred_base = F.softplus(main_pred) * scale_factor * self.pred_scale
        
        # 핵심 3가지 suppression만 적용
        final_pred = self.apply_core_suppression(pred_base, confidence, distance_map, mask, measured_values)
        
        return final_pred

    def apply_core_suppression(self, pred, confidence, distance_map, mask, measured_values):
        """핵심 3가지 suppression 메커니즘만 적용"""
        
        # 1. 거리 기반 억제 (핵심)
        distance_suppression = torch.exp(-1.5 * distance_map)
        
        # 2. 신뢰도 기반 조절
        confidence_modulation = 0.5 + 0.4 * confidence  # [0.5, 0.9] 범위
        
        # 3. 측정값 보존 (소프트 제약)
        measurement_preservation = 1.0 - 0.8 * mask  # 측정 지점 근처 억제 감소
        
        # 단순한 곱셈 조합
        suppression = distance_suppression * confidence_modulation * measurement_preservation
        pred_suppressed = pred * suppression
        
        # 강도 제한 (간단한 클리핑)
        max_intensity = measured_values.max() * 2.0  # 2배 제한
        pred_final = torch.clamp(pred_suppressed, max=max_intensity)
        
        return pred_final


def count_parameters(model):
    """모델 파라미터 수 계산"""
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params


# 물리 손실 함수들도 단순화
def simplified_laplacian_loss(pred: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
    """단순화된 라플라시안 손실"""
    # 간단한 라플라시안 커널
    laplacian_kernel = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], 
                                  dtype=pred.dtype, device=pred.device).view(1, 1, 3, 3)
    
    laplacian = F.conv2d(pred, laplacian_kernel, padding=1)
    
    if mask is not None:
        # 측정되지 않은 영역에 더 강한 평활성 적용
        weight = 1.0 + (1 - mask)
        return (weight * laplacian ** 2).mean()
    
    return (laplacian ** 2).mean()


def simplified_physics_loss(pred: torch.Tensor, gt: torch.Tensor, input_batch: torch.Tensor) -> torch.Tensor:
    """단순화된 물리 손실 (기본 역제곱 법칙만)"""
    device = pred.device
    B = pred.shape[0]
    
    # GT에서 단순한 피크 찾기
    total_loss = 0.0
    valid_batches = 0
    
    for b in range(B):
        gt_field = gt[b, 0]
        mask = input_batch[b, 1]
        measured_vals = input_batch[b, 0]
        
        # 가장 높은 값들만 소스로 간주 (단순화)
        if gt_field.max() > 0.2:
            # 기본적인 물리 일관성만 체크
            physics_loss = F.mse_loss(pred[b] * mask, measured_vals.unsqueeze(0) * mask)
            total_loss += physics_loss
            valid_batches += 1
    
    return total_loss / max(valid_batches, 1)


# 테스트 함수
def test_simplified_model():
    """단순화된 모델 테스트"""
    print("🔍 단순화된 ConvNeXt PGNN 테스트")
    print("=" * 50)
    
    # 모델 생성
    model = SimplifiedConvNeXtPGNN(in_channels=6, pred_scale=1.0)
    
    # 파라미터 수 계산
    param_count = count_parameters(model)
    print(f"총 파라미터 수: {param_count:,} ({param_count/1e6:.1f}M)")
    
    # 샘플 입력
    batch_size = 2
    input_tensor = torch.randn(batch_size, 6, 256, 256)
    
    # Forward pass 테스트
    with torch.no_grad():
        output = model(input_tensor)
    
    print(f"입력 형태: {input_tensor.shape}")
    print(f"출력 형태: {output.shape}")
    print(f"출력 범위: [{output.min().item():.4f}, {output.max().item():.4f}]")
    print(f"출력 평균: {output.mean().item():.4f}")
    print(f"출력 표준편차: {output.std().item():.4f}")
    
    # 메모리 사용량 추정
    model_size_mb = param_count * 4 / (1024**2)  # float32 기준
    print(f"모델 크기: {model_size_mb:.1f} MB")
    
    return model


if __name__ == "__main__":
    model = test_simplified_model()