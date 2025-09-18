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
    def __init__(self, in_channels=6, decoder_channels=[256, 128, 64, 32], pred_scale=1.0, enable_radial_guide=True):
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
        
        # 원형 가이드 레이어 추가
        if enable_radial_guide:
            self.radial_guide = RadialBiasLayer(grid_size=256)
        else:
            self.radial_guide = None

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
        
        # 기본 예측 (unmeasured 영역만)
        pred_base = F.softplus(main_pred) * scale_factor * self.pred_scale
        
        # 핵심 3가지 suppression만 적용 (unmeasured 영역만)
        pred_suppressed = self.apply_core_suppression(pred_base, confidence, distance_map, mask, measured_values)
        
        # 🔥 핵심: 측정값 직접 보존 (스파이크 문제 완전 해결)
        final_pred = self.preserve_measured_values(pred_suppressed, mask, measured_values)
        
        # 🎯 원형 분포 가이드 적용
        if self.radial_guide is not None:
            source_locs = self.extract_source_locations(mask, measured_values)
            final_pred = self.radial_guide(final_pred, source_locs)
        
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
        
        # 강도 제한 (정규화된 범위 준수)
        # GT가 [0,1] 정규화된 범위이므로 예측도 동일 범위로 제한
        max_intensity = 1.0  # 정규화된 최대값
        pred_final = torch.clamp(pred_suppressed, min=0.0, max=max_intensity)
        
        return pred_final

    def preserve_measured_values(self, pred, mask, measured_values):
        """
        🔥 핵심: 측정값 직접 보존으로 스파이크 아티팩트 완전 제거
        
        Args:
            pred: 모델 예측값 [B, 1, H, W]
            mask: 측정 마스크 [B, 1, H, W]
            measured_values: 실제 측정값 [B, 1, H, W]
        
        Returns:
            final_pred: 측정점에서 실제값 보존된 최종 예측 [B, 1, H, W]
        """
        # 측정점(mask=1)에서는 예측값 대신 실제 측정값 직접 사용
        # 비측정점(mask=0)에서는 모델 예측값 사용
        final_pred = pred * (1 - mask) + measured_values * mask
        
        # 선택적: 측정점 주변 부드러운 전환 (optional smoothing)
        if hasattr(self, 'enable_measurement_smoothing') and self.enable_measurement_smoothing:
            final_pred = self._apply_measurement_smoothing(final_pred, pred, mask, measured_values)
        
        return final_pred
    
    def _apply_measurement_smoothing(self, final_pred, pred, mask, measured_values):
        """측정점 주변 부드러운 전환 (선택적 기능)"""
        # 3x3 커널로 측정점 주변 가중 평균
        kernel = torch.ones(1, 1, 3, 3, device=pred.device) / 9.0
        
        # 측정점 주변 영역 식별
        dilated_mask = F.conv2d(mask, kernel, padding=1)
        transition_mask = (dilated_mask > 0) & (dilated_mask < 1)  # 경계 영역
        
        # 경계 영역에서 가중 평균 적용
        smoothed_pred = F.conv2d(final_pred, kernel, padding=1)
        
        # 경계 영역에만 부드러운 전환 적용
        final_pred = final_pred * (1 - transition_mask.float()) + \
                    smoothed_pred * transition_mask.float()
        
        # 측정점은 항상 원본값 유지
        final_pred = final_pred * (1 - mask) + measured_values * mask
        
        return final_pred
    
    def extract_source_locations(self, mask, measured_values):
        """강한 측정값 위치 추출 (원형 가이드용)"""
        source_locs = []
        B = mask.shape[0]
        
        for b in range(B):
            mask_b = mask[b, 0]
            meas_b = measured_values[b, 0]
            
            # 임계값 이상인 측정점들 찾기
            strong_sources = torch.where((mask_b > 0) & (meas_b > 0.2))
            if len(strong_sources[0]) > 0:
                positions = torch.stack([strong_sources[0], strong_sources[1]], dim=1)
                intensities = meas_b[strong_sources]
                source_locs.append((positions, intensities))
            else:
                source_locs.append(None)
                
        return source_locs


class RadialBiasLayer(nn.Module):
    """원형 분포를 유도하는 바이어스 레이어"""
    def __init__(self, grid_size=256, strength=0.3):
        super().__init__()
        self.strength = nn.Parameter(torch.tensor(strength))
        self.grid_size = grid_size
        
        # 중심으로부터의 거리 맵 생성 (정적 바이어스)
        y, x = torch.meshgrid(torch.arange(grid_size), torch.arange(grid_size), indexing='ij')
        center = grid_size // 2
        static_bias = torch.exp(-((x - center)**2 + (y - center)**2) / (grid_size**2 * 0.1))
        self.register_buffer('static_radial_bias', static_bias)
        
    def forward(self, pred, source_locations=None):
        """
        원형 바이어스 적용
        
        Args:
            pred: 모델 예측값 [B, 1, H, W]
            source_locations: 각 배치별 소스 위치 정보 리스트
        """
        B, C, H, W = pred.shape
        device = pred.device
        
        if source_locations is not None:
            # 동적 원형 바이어스 (소스 위치 기반)
            bias = self.create_dynamic_bias(pred, source_locations)
        else:
            # 정적 원형 바이어스 (중심 기준)
            bias = self.static_radial_bias.unsqueeze(0).unsqueeze(0).expand(B, 1, -1, -1)
        
        # 원형 바이어스 적용 (곱셈으로 원형 형태 강화)
        enhanced_pred = pred * (1.0 + self.strength * bias)
        
        return enhanced_pred
    
    def create_dynamic_bias(self, pred, source_locations):
        """소스 위치 기반 동적 원형 바이어스 생성"""
        B, C, H, W = pred.shape
        device = pred.device
        
        # 전체 바이어스 맵 초기화
        bias_maps = torch.zeros(B, 1, H, W, device=device)
        
        for b, source_info in enumerate(source_locations):
            if source_info is None:
                # 소스가 없으면 정적 바이어스 사용
                bias_maps[b, 0] = self.static_radial_bias
                continue
                
            positions, intensities = source_info
            bias_map = torch.zeros(H, W, device=device)
            
            # 각 소스에 대해 원형 바이어스 생성
            y_grid, x_grid = torch.meshgrid(torch.arange(H, device=device), 
                                           torch.arange(W, device=device), indexing='ij')
            
            for i, (pos, intensity) in enumerate(zip(positions, intensities)):
                sy, sx = pos[0].item(), pos[1].item()
                
                # 해당 소스로부터의 거리 계산
                distances = torch.sqrt((y_grid - sy)**2 + (x_grid - sx)**2) + 1e-6
                
                # 원형 바이어스 생성 (가우시안 형태)
                sigma = min(H, W) * 0.2  # 적절한 확산 범위
                radial_bias = torch.exp(-(distances**2) / (2 * sigma**2))
                
                # 강도에 비례하여 바이어스 가중치 적용
                weight = intensity / intensities.max() if intensities.max() > 0 else 1.0
                bias_map += weight * radial_bias
            
            # 정규화
            if bias_map.max() > 0:
                bias_map = bias_map / bias_map.max()
            
            bias_maps[b, 0] = bias_map
        
        return bias_maps


def radial_physics_loss(pred: torch.Tensor, measured_values: torch.Tensor, 
                       mask: torch.Tensor, alpha=0.1) -> torch.Tensor:
    """원형 분포를 강제하는 물리 손실"""
    device = pred.device
    B, C, H, W = pred.shape
    total_loss = 0.0
    valid_samples = 0
    
    for b in range(B):
        pred_field = pred[b, 0]
        mask_b = mask[b, 0] 
        measured_b = measured_values[b, 0]
        
        # 측정점들에서 소스 위치 추정
        source_positions = torch.where(mask_b > 0)
        if len(source_positions[0]) == 0:
            continue
            
        for i in range(len(source_positions[0])):
            sy, sx = source_positions[0][i].item(), source_positions[1][i].item()
            source_intensity = measured_b[sy, sx].item()
            
            if source_intensity > 0.1:  # 유의미한 소스만
                # 해당 소스로부터의 이론적 원형 분포 계산
                y_grid, x_grid = torch.meshgrid(torch.arange(H, device=device), 
                                               torch.arange(W, device=device), indexing='ij')
                
                # 거리 계산
                distances = torch.sqrt((y_grid - sy)**2 + (x_grid - sx)**2) + 1e-6
                
                # 이론적 원형 분포 (역제곱 법칙 + 지수 감쇠)
                theoretical = source_intensity * torch.exp(-distances / (min(H, W) * 0.3)) / (distances + 1.0)
                theoretical = torch.clamp(theoretical, min=0.0, max=source_intensity)
                
                # 원형성 손실 (측정점 주변 영역만 고려)
                mask_region = (distances < min(H, W) * 0.4)  # 영향 범위 제한
                
                if mask_region.sum() > 0:
                    radial_loss = F.mse_loss(pred_field[mask_region], theoretical[mask_region])
                    total_loss += radial_loss * alpha
                    valid_samples += 1
    
    return total_loss / max(valid_samples, 1) if valid_samples > 0 else torch.tensor(0.0, device=device)


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
    model = SimplifiedConvNeXtPGNN(in_channels=6, pred_scale=1.0, enable_radial_guide=True)
    
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