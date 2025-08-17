import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models import create_model
import numpy as np


class DecoderBlock(nn.Module):
    def __init__(self, in_c, skip_c, out_c):
        super().__init__()
        # Enhanced skip connection processing
        self.skip_conv = nn.Sequential(
            nn.Conv2d(skip_c, skip_c // 2, kernel_size=1),
            nn.BatchNorm2d(skip_c // 2),
            nn.ReLU(inplace=True)
        ) if skip_c > 128 else nn.Identity()
        
        skip_c_proc = skip_c // 2 if skip_c > 128 else skip_c
        
        self.block = nn.Sequential(
            nn.Conv2d(in_c + skip_c_proc, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )
        
        # Spatial attention for better feature fusion
        self.spatial_att = nn.Sequential(
            nn.Conv2d(out_c, out_c // 8, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c // 8, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x, skip):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        
        # Process skip connection
        if not isinstance(self.skip_conv, nn.Identity):
            skip = self.skip_conv(skip)
        
        x = torch.cat([x, skip], dim=1)
        x = self.block(x)
        
        # Apply spatial attention
        att = self.spatial_att(x)
        x = x * att
        
        return x


class ConvNeXtUNetWithMaskAttention(nn.Module):
    def __init__(self, in_channels=6, decoder_channels=[512, 256, 128, 64], pred_scale=4.0):
        super().__init__()

        # ConvNeXt encoder
        self.encoder = create_model(
            "convnext_base",
            pretrained=False,
            features_only=True,
            in_chans=in_channels
        )

        enc_channels = [f['num_chs'] for f in self.encoder.feature_info]

        # Enhanced mask attention projection layers
        self.mask_proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, max(ch // 4, 32), kernel_size=3, padding=1),
                nn.BatchNorm2d(max(ch // 4, 32)),
                nn.ReLU(inplace=True),
                nn.Conv2d(max(ch // 4, 32), ch, kernel_size=1)
            ) for ch in enc_channels
        ])

        # decoder with residual connections
        self.up4 = DecoderBlock(enc_channels[3], enc_channels[2], decoder_channels[0])
        self.up3 = DecoderBlock(decoder_channels[0], enc_channels[1], decoder_channels[1])
        self.up2 = DecoderBlock(decoder_channels[1], enc_channels[0], decoder_channels[2])
        self.up1 = nn.Sequential(
            nn.Conv2d(decoder_channels[2], decoder_channels[3], kernel_size=3, padding=1),
            nn.BatchNorm2d(decoder_channels[3]),
            nn.ReLU(inplace=True),
        )

        # Multi-branch prediction heads with anti-blur modifications
        self.intensity_head = nn.Sequential(
            nn.Conv2d(decoder_channels[3], 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1)
        )
        
        # Sharp detail enhancement head - anti-blur mechanism
        self.sharpness_head = nn.Sequential(
            nn.Conv2d(decoder_channels[3], 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid()  # 0~1 sharpness enhancement factor
        )
        
        # Confidence/suppression branch for unmeasured regions
        self.confidence_branch = nn.Sequential(
            nn.Conv2d(decoder_channels[3], 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid()  # 0~1 confidence score
        )
        
        # Peak location refinement head
        self.peak_head = nn.Sequential(
            nn.Conv2d(decoder_channels[3], 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, kernel_size=1),
            nn.Sigmoid()  # Peak probability
        )
        
        # Adaptive scaling parameters  
        self.adaptive_scale = nn.Parameter(torch.tensor(1.0))
        self.min_scale = nn.Parameter(torch.tensor(0.5))
        self.max_scale = nn.Parameter(torch.tensor(3.5))
        
        self.pred_scale = pred_scale
        self.suppression_threshold = 0.1  # 측정점으로부터의 거리 기반 임계값

    def forward(self, x):
        mask = x[:, 1:2]   # measured location mask
        measured_values = x[:, 0:1]  # actual measured values
        distance_map = x[:, 5:6]  # distance from measured points
        # Fix coordinate assignment: dataset has [Y, X] in channels [3, 4]
        coord_x = x[:, 4:5]  # X coordinates (columns) - was incorrectly x[:, 3:4]
        coord_y = x[:, 3:4]  # Y coordinates (rows) - was incorrectly x[:, 4:5]

        feats = self.encoder(x)  # c1, c2, c3, c4
        feats_attn = []

        # Rebalanced mask attention - moderate enhancement
        for f, attn_conv in zip(feats, self.mask_proj):
            mask_resized = F.interpolate(mask, size=f.shape[2:], mode='bilinear', align_corners=False)
            mask_feat = attn_conv(mask_resized)
            # Balanced attention to guide without over-enhancement
            attention = torch.sigmoid(mask_feat) * 0.3  # Increased from 0.2 to 0.3
            feats_attn.append(f * (1 + attention) + 0.1 * mask_feat)  # Increased additive term

        c1, c2, c3, c4 = feats_attn
        u4 = self.up4(c4, c3)
        u3 = self.up3(u4, c2)
        u2 = self.up2(u3, c1)
        u1 = F.interpolate(u2, scale_factor=2, mode='bilinear', align_corners=False)
        u1 = self.up1(u1)
        out = F.interpolate(u1, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        # Multi-branch predictions
        intensity_raw = self.intensity_head(out)
        confidence = self.confidence_branch(out)
        peak_prob = self.peak_head(out)
        
        # Advanced adaptive scaling
        B, _, H, W = x.shape
        measured_stats = torch.zeros(B, 3, device=x.device)  # max, mean, std
        
        for b in range(B):
            mask_b = mask[b, 0]  # [H, W]
            if mask_b.sum() > 0:
                meas_vals = measured_values[b, 0][mask_b > 0]  # [N_measured]
                measured_stats[b, 0] = meas_vals.max()
                measured_stats[b, 1] = meas_vals.mean()
                measured_stats[b, 2] = meas_vals.std() if len(meas_vals) > 1 else 0.0
            else:
                measured_stats[b, 0] = 1.0
                measured_stats[b, 1] = 0.5
                measured_stats[b, 2] = 0.1
        
        # Learnable adaptive scaling
        scale_factor = (
            self.min_scale + 
            (self.max_scale - self.min_scale) * torch.sigmoid(
                self.adaptive_scale * measured_stats[:, 0:1].unsqueeze(-1).unsqueeze(-1)
            )
        )
        
        # Anti-blur: Sharper intensity prediction with gradient-preserving activation
        sharpness_factor = self.sharpness_head(out)
        
        # More aggressive scaling to prevent blur
        intensity_scaled = intensity_raw * (1.2 + 0.5 * sharpness_factor)  # Dynamic scaling
        pred_base = F.softplus(intensity_scaled) * scale_factor * self.pred_scale
        
        # Enhanced peak sharpening with gradient preservation
        peak_enhancement = 1.0 + (0.4 * peak_prob + 0.3 * sharpness_factor) * (1 - mask)
        pred_enhanced = pred_base * peak_enhancement
        
        # Anti-blur: Local contrast enhancement
        # Apply unsharp masking to enhance local details
        kernel_blur = torch.ones(1, 1, 3, 3, device=x.device) / 9.0  # 3x3 blur kernel
        pred_blurred = F.conv2d(pred_enhanced, kernel_blur, padding=1)
        unsharp_mask = pred_enhanced - pred_blurred
        pred_enhanced = pred_enhanced + 0.3 * unsharp_mask * (1 - mask)  # Apply only to unmeasured regions
        
        # SIMPLIFIED: 3-mechanism suppression strategy
        # 1. CORE: Distance-based suppression with smooth falloff
        DISTANCE_DECAY = -1.5  # Standardized decay rate
        distance_suppression = torch.exp(DISTANCE_DECAY * distance_map)
        
        # 2. SELECTIVE: Confidence-guided modulation  
        CONFIDENCE_MIN = 0.4  # Minimum confidence factor
        CONFIDENCE_RANGE = 0.5  # Range [0.4, 0.9]
        confidence_modulation = CONFIDENCE_MIN + CONFIDENCE_RANGE * confidence
        
        # 3. PROXIMITY: Smart near-measurement handling
        NEAR_THRESHOLD = 0.08  # Distance threshold for "near" measurements
        TRANSITION_WIDTH = 0.12  # Smooth transition zone width
        
        proximity_modulation = torch.where(
            distance_map < NEAR_THRESHOLD,
            torch.ones_like(distance_map) * 0.8,  # Preserve near-measurement regions
            torch.where(
                distance_map < (NEAR_THRESHOLD + TRANSITION_WIDTH),
                # Smooth linear transition
                0.8 + 0.2 * (distance_map - NEAR_THRESHOLD) / TRANSITION_WIDTH,
                torch.ones_like(distance_map)  # Full suppression allowed far away
            )
        )
        
        # COMBINED: Simple multiplicative combination with feature preservation
        base_suppression = distance_suppression * confidence_modulation * proximity_modulation
        
        # FEATURE-AWARE: Preserve high-confidence or peak regions
        PEAK_THRESHOLD = 0.25  # Peak probability threshold
        CONFIDENCE_THRESHOLD = 0.7  # High confidence threshold
        
        feature_preservation_mask = (peak_prob > PEAK_THRESHOLD) | (confidence > CONFIDENCE_THRESHOLD)
        PRESERVATION_FACTOR = 0.95  # Minimal suppression for important features
        
        final_suppression = torch.where(
            feature_preservation_mask,
            torch.ones_like(base_suppression) * PRESERVATION_FACTOR,
            base_suppression
        )
        
        # Apply simplified suppression
        pred_suppressed = pred_enhanced * final_suppression
        
        # GRADIENT-PRESERVING measurement constraint
        # Use soft constraint instead of hard substitution to preserve gradients
        MEASUREMENT_LOSS_WEIGHT = 1000.0  # Strong constraint but preserves gradients
        
        # Soft measurement preservation (allows gradients to flow)
        measurement_constraint = (pred_suppressed - measured_values) * mask
        pred_final = pred_suppressed - 0.01 * measurement_constraint  # Gentle correction
        
        # Note: The strong measurement constraint will be enforced via loss function
        # This preserves gradient flow while still maintaining measurement accuracy
        
        # REGIONAL CONSTRAINTS: Anti-blur sharp constraints with documented thresholds
        # FAR REGIONS: Distance > 60% of max distance, likely background with some features
        FAR_DISTANCE_THRESHOLD = 0.6    # 60% of normalized distance (empirically determined)
        FAR_PEAK_THRESHOLD = 0.2        # Low peak probability for background regions
        FAR_SUPPRESSION_FACTOR = 0.25   # Moderate suppression preserving potential features
        
        far_regions = (distance_map > FAR_DISTANCE_THRESHOLD) & (mask == 0) & (peak_prob < FAR_PEAK_THRESHOLD)
        pred_final = torch.where(far_regions, pred_final * FAR_SUPPRESSION_FACTOR, pred_final)
        
        # VERY FAR REGIONS: Distance > 85% of max distance, definite background
        VERY_FAR_DISTANCE_THRESHOLD = 0.85  # 85% of normalized distance
        VERY_FAR_PEAK_THRESHOLD = 0.1       # Very low peak probability
        VERY_FAR_SUPPRESSION_FACTOR = 0.1   # Strong suppression for definite background
        
        very_far_regions = (distance_map > VERY_FAR_DISTANCE_THRESHOLD) & (mask == 0) & (peak_prob < VERY_FAR_PEAK_THRESHOLD)
        pred_final = torch.where(very_far_regions, pred_final * VERY_FAR_SUPPRESSION_FACTOR, pred_final)
        
        # INTENSITY CLAMPING: Prevent unrealistic intensity predictions
        INTENSITY_MULTIPLIER = 2.5  # Allow predictions up to 2.5x max measured value
                                   # Rationale: Physical systems can have local amplification
                                   # but extreme values (>2.5x) are likely artifacts
        max_allowed_intensity = measured_values.max() * INTENSITY_MULTIPLIER
        pred_final = torch.clamp(pred_final, max=max_allowed_intensity)
        
        return pred_final


# -------------------------------------------------------------
# Enhanced Physics-Guided Loss Functions
# -------------------------------------------------------------

def peak_location_loss(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Loss specifically for peak location accuracy"""
    # Find peaks in ground truth (local maxima above threshold)
    kernel_size = 5
    gt_maxpool = F.max_pool2d(gt, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
    gt_peaks = ((gt == gt_maxpool) & (gt > 0.3)).float()
    
    # Find peaks in prediction
    pred_maxpool = F.max_pool2d(pred, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
    pred_peaks = ((pred == pred_maxpool) & (pred > 0.2)).float()
    
    # Peak location loss: encourage peaks to align
    peak_loss = F.mse_loss(pred_peaks, gt_peaks)
    
    # Peak intensity loss: ensure peak intensities match
    intensity_loss = F.mse_loss(pred * gt_peaks, gt * gt_peaks)
    
    return 0.6 * peak_loss + 0.4 * intensity_loss


def spatial_consistency_loss(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Enforce spatial consistency in gradients"""
    # Calculate gradients
    pred_grad_x = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
    pred_grad_y = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
    gt_grad_x = torch.abs(gt[:, :, :, 1:] - gt[:, :, :, :-1])
    gt_grad_y = torch.abs(gt[:, :, 1:, :] - gt[:, :, :-1, :])
    
    # Weight gradients more near measurements
    mask_x = F.interpolate(mask, size=pred_grad_x.shape[2:], mode='bilinear', align_corners=False)
    mask_y = F.interpolate(mask, size=pred_grad_y.shape[2:], mode='bilinear', align_corners=False)
    
    weight_x = 1.0 + torch.exp(-3.0 * mask_x[:, :, :, :-1])
    weight_y = 1.0 + torch.exp(-3.0 * mask_y[:, :, :-1, :])
    
    grad_loss_x = F.mse_loss(pred_grad_x * weight_x, gt_grad_x * weight_x)
    grad_loss_y = F.mse_loss(pred_grad_y * weight_y, gt_grad_y * weight_y)
    
    return (grad_loss_x + grad_loss_y) * 0.5


def regional_adaptive_loss(pred: torch.Tensor, gt: torch.Tensor, distance_map: torch.Tensor) -> torch.Tensor:
    """Adaptive loss weighting based on distance from measurements"""
    # Near region (high importance for accuracy)
    near_mask = (distance_map <= 0.3).float()
    near_weight = 2.0
    near_loss = F.mse_loss(pred * near_mask, gt * near_mask) * near_weight
    
    # Medium region (moderate importance)
    medium_mask = ((distance_map > 0.3) & (distance_map <= 0.6)).float()
    medium_weight = 1.5
    medium_loss = F.mse_loss(pred * medium_mask, gt * medium_mask) * medium_weight
    
    # Far region (low importance, focus on suppression)
    far_mask = (distance_map > 0.6).float()
    far_weight = 0.8
    far_loss = F.mse_loss(pred * far_mask, gt * far_mask) * far_weight
    
    total_pixels = near_mask.sum() + medium_mask.sum() + far_mask.sum()
    if total_pixels > 0:
        return (near_loss + medium_loss + far_loss) / total_pixels
    else:
        return torch.tensor(0.0, device=pred.device)


def intensity_preservation_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Preserve high-intensity regions accurately"""
    # Focus on high-intensity regions
    high_intensity_mask = (gt > 0.5).float()
    
    if high_intensity_mask.sum() > 0:
        # Intensity preservation
        intensity_loss = F.mse_loss(pred * high_intensity_mask, gt * high_intensity_mask)
        
        # Ensure peak values are not underestimated
        peak_ratio = (pred.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0] + 1e-8) / \
                     (gt.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0] + 1e-8)
        ratio_loss = F.mse_loss(peak_ratio, torch.ones_like(peak_ratio))
        
        return intensity_loss + 0.3 * ratio_loss
    else:
        return torch.tensor(0.0, device=pred.device)


def anti_blur_loss(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Anti-blur loss to prevent foggy/blurry predictions"""
    # 1. High-frequency detail preservation
    # Compute Laplacian for both pred and gt to detect edges/details
    pred_laplacian = laplacian_tensor(pred)
    gt_laplacian = laplacian_tensor(gt)
    
    # Focus on regions with high detail in GT
    detail_mask = (torch.abs(gt_laplacian) > 0.01).float()
    detail_loss = F.mse_loss(pred_laplacian * detail_mask, gt_laplacian * detail_mask)
    
    # 2. Gradient magnitude preservation
    pred_grad_x = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
    pred_grad_y = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
    gt_grad_x = torch.abs(gt[:, :, :, 1:] - gt[:, :, :, :-1])
    gt_grad_y = torch.abs(gt[:, :, 1:, :] - gt[:, :, :-1, :])
    
    gradient_loss = F.mse_loss(pred_grad_x, gt_grad_x) + F.mse_loss(pred_grad_y, gt_grad_y)
    
    # 3. Local contrast preservation
    # Use 3x3 local standard deviation as contrast measure
    kernel_size = 3
    padding = kernel_size // 2
    
    # Calculate local standard deviation
    pred_unfold = F.unfold(pred, kernel_size, padding=padding)
    gt_unfold = F.unfold(gt, kernel_size, padding=padding)
    
    pred_local_std = pred_unfold.std(dim=1, keepdim=True)
    gt_local_std = gt_unfold.std(dim=1, keepdim=True)
    
    # Reshape back to spatial dimensions
    B, _, H, W = pred.shape
    pred_local_std = pred_local_std.view(B, 1, H, W)
    gt_local_std = gt_local_std.view(B, 1, H, W)
    
    contrast_loss = F.mse_loss(pred_local_std, gt_local_std)
    
    # 4. Penalty for over-smoothing in non-measured regions
    unmeasured_mask = (1 - mask).float()
    pred_smooth_penalty = torch.where(
        unmeasured_mask > 0,
        -torch.log(torch.clamp(pred_local_std + 1e-8, min=1e-8, max=1.0)),  # Penalty for low variance
        torch.zeros_like(pred_local_std)
    ).mean()
    
    # Combine all anti-blur components
    total_anti_blur_loss = (
        0.4 * detail_loss +      # Edge/detail preservation
        0.3 * gradient_loss +    # Gradient magnitude preservation  
        0.2 * contrast_loss +    # Local contrast preservation
        0.1 * pred_smooth_penalty # Anti-smoothing penalty
    )
    
    return total_anti_blur_loss


def sharpness_enhancement_loss(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Encourage sharp, well-defined peaks instead of blurry regions"""
    # 1. Peak sharpness loss - encourage narrow, high peaks
    # Find peak regions in GT
    kernel_size = 5
    gt_maxpool = F.max_pool2d(gt, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
    gt_peaks = ((gt == gt_maxpool) & (gt > 0.2)).float()
    
    if gt_peaks.sum() > 0:
        # For peak regions, encourage sharp falloff around peaks
        # Calculate distance to nearest peak
        peak_coords = torch.nonzero(gt_peaks.squeeze(), as_tuple=False)
        
        if len(peak_coords) > 0:
            B, _, H, W = pred.shape
            y_grid, x_grid = torch.meshgrid(
                torch.arange(H, device=pred.device),
                torch.arange(W, device=pred.device),
                indexing='ij'
            )
            
            distance_to_peaks = torch.full((H, W), float('inf'), device=pred.device)
            
            for peak_coord in peak_coords:
                py, px = peak_coord[0], peak_coord[1]
                dist = torch.sqrt((y_grid - py)**2 + (x_grid - px)**2)
                distance_to_peaks = torch.min(distance_to_peaks, dist)
            
            # Normalize distance
            distance_to_peaks = distance_to_peaks / max(H, W)
            
            # Expected falloff (exponential decay from peaks)
            expected_falloff = torch.exp(-3.0 * distance_to_peaks.unsqueeze(0).unsqueeze(0))
            gt_peak_intensity = gt * gt_peaks
            expected_intensity = gt_peak_intensity.max() * expected_falloff
            
            # Loss for regions around peaks
            peak_region_mask = (distance_to_peaks < 0.3).float().unsqueeze(0).unsqueeze(0)
            sharpness_loss = F.mse_loss(
                pred * peak_region_mask, 
                torch.minimum(gt, expected_intensity) * peak_region_mask
            )
            
            return sharpness_loss
    
    return torch.tensor(0.0, device=pred.device)


def total_variation_loss(pred: torch.Tensor, mask: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """Total variation loss to reduce noise while preserving edges"""
    # Only apply TV loss to unmeasured regions to avoid affecting measured values
    unmeasured_mask = (1 - mask).float()
    
    # Compute gradients
    diff_x = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    diff_y = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    
    # Apply mask to gradients
    mask_x = F.interpolate(unmeasured_mask, size=diff_x.shape[2:], mode='bilinear', align_corners=False)
    mask_y = F.interpolate(unmeasured_mask, size=diff_y.shape[2:], mode='bilinear', align_corners=False)
    
    # Total variation with masking
    tv_loss = (
        torch.mean(torch.abs(diff_x) * mask_x[:, :, :, :-1]) +
        torch.mean(torch.abs(diff_y) * mask_y[:, :, :-1, :])
    )
    
    return alpha * tv_loss


# -------------------------------------------------------------
# Original Physics-Guided Loss Functions  
# -------------------------------------------------------------

def laplacian_tensor(x: torch.Tensor) -> torch.Tensor:
    """5-point Laplacian with replicate padding."""
    x_p = F.pad(x, (1, 1, 1, 1), mode="replicate")
    lap = (x_p[:, :, 1:-1, :-2] + x_p[:, :, 1:-1, 2:] +
           x_p[:, :, :-2, 1:-1] + x_p[:, :, 2:, 1:-1] - 4 * x)
    return lap

def laplacian_loss(pred: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
    """Physics-guided smoothness loss with optional masking."""
    laplacian = laplacian_tensor(pred)
    if mask is not None:
        # Apply stronger smoothness constraint away from measured points
        weight = 1.0 + 2.0 * (1 - mask)  # Higher weight for unmeasured regions
        return (weight * laplacian ** 2).mean()
    return (laplacian ** 2).mean()


def physics_loss_unified(pred: torch.Tensor, gt: torch.Tensor, input_batch: torch.Tensor,
                         background_level: float = 0.005, beta: float = 0.03) -> torch.Tensor:
    """
    통합된 물리 손실 함수 - 안정화되고 표준화된 버전
    
    Args:
        pred: 예측된 방사선 필드 [B, 1, H, W]
        gt: Ground truth 방사선 필드 [B, 1, H, W]
        input_batch: 입력 배치 [B, 6, H, W]
        background_level: 배경 방사선 수준 (표준값: 0.005)
        beta: 공기 중 감쇠 계수 (표준값: 0.03)
        
    Returns:
        physics_loss: 안정화된 물리 손실
    """
    device = pred.device
    B, _, H, W = pred.shape
    
    # GT에서 신뢰할 수 있는 소스 추정
    gt_source_estimates = estimate_sources_from_field(gt, threshold=0.15, max_sources=4)
    
    # 측정 데이터 추출
    try:
        measured_positions, measured_values = extract_measured_data(input_batch)
    except Exception:
        return torch.tensor(0.0, device=device)
    
    total_loss = 0.0
    valid_batches = 0
    
    for b in range(B):
        if (len(measured_positions) <= b or len(measured_values) <= b or 
            measured_positions[b].shape[0] == 0 or gt_source_estimates[b].shape[0] == 0):
            continue
            
        meas_pos = measured_positions[b]  # [N_meas, 2]
        meas_vals = measured_values[b]    # [N_meas]
        gt_sources = gt_source_estimates[b]  # [N_sources, 3]
        
        # 측정값 유효성 검사
        if (torch.any(torch.isnan(meas_vals)) or torch.any(torch.isinf(meas_vals)) or 
            torch.any(meas_vals < 0)):
            continue
            
        # GT 소스 기반 물리적 예측값 계산
        physics_predictions = torch.full_like(meas_vals, background_level)
        
        for s_idx in range(gt_sources.shape[0]):
            source_x, source_y, intensity = gt_sources[s_idx]
            
            # 소스 유효성 검사
            if intensity < 0.01:  # 너무 약한 소스 제외
                continue
                
            # 안정적인 강도 제한
            intensity = torch.clamp(intensity, min=0.01, max=5.0)
            
            # 거리 계산 (안정화됨)
            distances = torch.sqrt((meas_pos[:, 0] - source_x)**2 + 
                                 (meas_pos[:, 1] - source_y)**2)
            distances = torch.clamp(distances, min=0.03, max=1.5)  # 표준화된 범위
            
            # 역제곱 법칙 + 감쇠 (안정화됨)
            inv_sq_term = intensity / (distances**2 + 1e-7)
            attenuation = torch.exp(-beta * distances)
            contribution = inv_sq_term * attenuation
            
            # 안정적인 기여도 제한
            contribution = torch.clamp(contribution, max=3.0)
            physics_predictions += contribution
        
        # 최종 물리 예측값 제한
        physics_predictions = torch.clamp(physics_predictions, min=0.0, max=2.0)
        
        # 안정화된 손실 계산 (Smooth L1 Loss)
        diff = meas_vals - physics_predictions
        batch_physics_loss = F.smooth_l1_loss(meas_vals, physics_predictions, reduction='mean')
        
        # 안전성 검사
        if torch.isnan(batch_physics_loss) or torch.isinf(batch_physics_loss):
            continue
            
        total_loss += batch_physics_loss
        valid_batches += 1
    
    if valid_batches == 0:
        return torch.tensor(0.0, device=device)
    
    # 안정적인 스케일링
    final_loss = total_loss / valid_batches
    return torch.clamp(final_loss, max=5.0)  # 표준화된 최대값


def gt_based_inverse_square_law_loss(pred: torch.Tensor, gt: torch.Tensor, 
                                    measured_positions: torch.Tensor, measured_values: torch.Tensor,
                                    background_level: float = 0.00, beta: float = 0.01) -> torch.Tensor:
    """
    Ground Truth 기반 역제곱 법칙 물리 손실 함수
    
    GT에서 소스를 추정하고, 이를 바탕으로 계산한 물리적 예측값과 모델 예측값을 비교
    
    Args:
        pred: 예측된 방사선 필드 [B, 1, H, W]
        gt: Ground Truth 방사선 필드 [B, 1, H, W]
        measured_positions: 측정 위치 좌표 [B, N_meas, 2] (normalized [-1, 1])
        measured_values: 측정 값 [B, N_meas]
        background_level: 배경 방사선 수준
        beta: 공기 중 감쇠 계수
        
    Returns:
        physics_loss: GT 기반 물리 예측값과 모델 예측값 간의 차이
    """
    device = pred.device
    B, _, H, W = pred.shape
    
    # GT에서 소스 추정 (더 정확한 소스 정보)
    gt_source_estimates = estimate_sources_from_field(gt, threshold=0.15, max_sources=6)
    
    total_loss = 0.0
    valid_batches = 0
    
    for b in range(B):
        meas_pos = measured_positions[b]  # [N_meas, 2]
        meas_vals = measured_values[b]    # [N_meas]
        gt_sources = gt_source_estimates[b]  # [N_sources, 3]
        
        if meas_pos.shape[0] == 0 or gt_sources.shape[0] == 0:
            continue
            
        # 측정값 유효성 검사
        if torch.any(torch.isnan(meas_vals)) or torch.any(torch.isinf(meas_vals)):
            continue
        
        # GT 소스를 바탕으로 각 측정 위치에서 물리적 예측값 계산
        gt_physics_predictions = torch.full_like(meas_vals, background_level)
        
        for s_idx in range(gt_sources.shape[0]):
            source_x, source_y, intensity = gt_sources[s_idx]
            
            if intensity < 0.01:  # 매우 약한 소스는 제외
                continue
                
            # 소스 강도 제한
            intensity = torch.clamp(intensity, min=0.001, max=10.0)
            
            # 측정 위치와 GT 소스 간의 거리 계산
            distances = torch.sqrt((meas_pos[:, 0] - source_x)**2 + 
                                 (meas_pos[:, 1] - source_y)**2)
            
            # 거리 제한 (singularity 방지)
            distances = torch.clamp(distances, min=0.03, max=2.0)
            
            # 역제곱 법칙 + 공기 감쇠
            inv_sq_term = intensity / (distances**2 + 1e-7)
            attenuation = torch.exp(-beta * distances)
            contribution = inv_sq_term * attenuation
            
            # 기여도 제한
            contribution = torch.clamp(contribution, max=8.0)
            gt_physics_predictions += contribution
        
        # GT 물리 예측값 범위 제한
        gt_physics_predictions = torch.clamp(gt_physics_predictions, min=0.0, max=5.0)
        
        # 예측 필드에서 동일한 측정 위치의 값 추출
        pred_b = pred[b, 0]  # [H, W]
        pred_at_measurements = []
        
        for m_idx in range(meas_pos.shape[0]):
            # 정규화된 좌표를 픽셀 좌표로 변환
            norm_x, norm_y = meas_pos[m_idx]
            pixel_x = int((norm_x + 1) * (W - 1) / 2)
            pixel_y = int((norm_y + 1) * (H - 1) / 2)
            pixel_x = torch.clamp(torch.tensor(pixel_x), 0, W-1)
            pixel_y = torch.clamp(torch.tensor(pixel_y), 0, H-1)
            
            pred_val = pred_b[pixel_y, pixel_x]
            pred_at_measurements.append(pred_val)
        
        pred_at_measurements = torch.stack(pred_at_measurements)
        
        # GT 기반 물리 예측값과 모델 예측값 간의 차이 (Smooth L1 Loss)
        diff = pred_at_measurements - gt_physics_predictions
        physics_loss = F.smooth_l1_loss(pred_at_measurements, gt_physics_predictions, reduction='mean')
        
        # NaN/Inf 검사
        if torch.isnan(physics_loss) or torch.isinf(physics_loss):
            continue
            
        total_loss += physics_loss
        valid_batches += 1
    
    # 유효한 배치가 없으면 0 반환
    if valid_batches == 0:
        return torch.tensor(0.0, device=device)
    
    return total_loss / valid_batches


def estimate_sources_from_field(pred_field: torch.Tensor, threshold: float = 0.2, 
                               max_sources: int = 4) -> torch.Tensor:
    """
    예측된 방사선 필드로부터 소스 위치와 강도 추정
    
    Args:
        pred_field: 예측된 방사선 필드 [B, 1, H, W]
        threshold: 소스 감지 임계값
        max_sources: 최대 소스 개수
        
    Returns:
        source_estimates: [B, max_sources, 3] (x, y, intensity)
    """
    B, _, H, W = pred_field.shape
    device = pred_field.device
    
    source_estimates = torch.zeros(B, max_sources, 3, device=device)
    
    for b in range(B):
        field = pred_field[b, 0]  # [H, W]
        
        # 로컬 맥시마 찾기
        maxpool = F.max_pool2d(field.unsqueeze(0).unsqueeze(0), 
                              kernel_size=5, stride=1, padding=2)
        local_maxima = (field == maxpool.squeeze()) & (field > threshold)
        
        # 상위 max_sources개 선택
        y_coords, x_coords = torch.where(local_maxima)
        intensities = field[local_maxima]
        
        if len(intensities) > 0:
            # 강도 순으로 정렬
            sorted_idx = torch.argsort(intensities, descending=True)
            n_sources = min(len(sorted_idx), max_sources)
            
            for i in range(n_sources):
                idx = sorted_idx[i]
                # 좌표를 [-1, 1] 범위로 정규화
                norm_x = 2.0 * x_coords[idx] / (W - 1) - 1.0
                norm_y = 2.0 * y_coords[idx] / (H - 1) - 1.0
                intensity = intensities[idx]
                
                source_estimates[b, i] = torch.tensor([norm_x, norm_y, intensity])
    
    return source_estimates


def extract_measured_data(input_batch: torch.Tensor) -> tuple:
    """
    입력 배치에서 측정 위치와 값 추출
    
    Args:
        input_batch: [B, 6, H, W] - [measured_values, mask, log_measured, X, Y, distance]
        
    Returns:
        measured_positions: [B, N_meas, 2] 측정 위치 좌표
        measured_values: [B, N_meas] 측정 값
    """
    B, _, H, W = input_batch.shape
    device = input_batch.device
    
    measured_vals = input_batch[:, 0]  # [B, H, W]
    mask = input_batch[:, 1]          # [B, H, W]
    # Fix coordinate assignment: dataset has [Y, X] in channels [3, 4]
    coord_x = input_batch[:, 4]       # [B, H, W] 정규화된 X 좌표 (columns)
    coord_y = input_batch[:, 3]       # [B, H, W] 정규화된 Y 좌표 (rows)
    
    all_positions = []
    all_values = []
    
    for b in range(B):
        # 측정된 위치 찾기
        measured_mask = mask[b] > 0
        if measured_mask.sum() > 0:
            y_coords, x_coords = torch.where(measured_mask)
            
            # 정규화된 좌표 사용
            positions = torch.stack([coord_x[b][measured_mask], 
                                   coord_y[b][measured_mask]], dim=1)  # [N_meas, 2]
            values = measured_vals[b][measured_mask]  # [N_meas]
            
            all_positions.append(positions)
            all_values.append(values)
        else:
            # 측정이 없는 경우 빈 텐서
            all_positions.append(torch.empty(0, 2, device=device))
            all_values.append(torch.empty(0, device=device))
    
    return all_positions, all_values