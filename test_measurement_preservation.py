#!/usr/bin/env python3
"""
측정값 직접 보존 기능 테스트 스크립트
스파이크 아티팩트 제거 효과 검증

Test Features:
1. 단순화된 모델의 측정값 보존 확인
2. 스파이크 아티팩트 제거 효과 검증
3. 자연스러운 필드 연속성 확인
4. 기존 방식과 비교 분석
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# 모델 임포트
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from src.model.simplified_conv_next_pgnn import SimplifiedConvNeXtPGNN
from uncertainty_aware_conv_next_pgnn import UncertaintyAwareConvNeXtPGNN

def create_test_data(batch_size=1, height=256, width=256):
    """테스트용 합성 데이터 생성"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 기본 입력 구조 생성
    inp = torch.zeros(batch_size, 6, height, width, device=device)
    
    # GT 필드 생성 (가우시안 소스)
    y_center, x_center = height // 2, width // 2
    y_coords, x_coords = torch.meshgrid(
        torch.linspace(0, height-1, height, device=device),
        torch.linspace(0, width-1, width, device=device),
        indexing='ij'
    )
    
    # 두 개의 가우시안 소스
    source1 = 0.8 * torch.exp(-((y_coords - y_center + 30)**2 + (x_coords - x_center - 40)**2) / (2 * 25**2))
    source2 = 0.6 * torch.exp(-((y_coords - y_center - 35)**2 + (x_coords - x_center + 50)**2) / (2 * 20**2))
    gt_field = source1 + source2 + 0.02  # 배경 방사선
    gt = gt_field.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    
    # 측정점 생성 (랜덤하게 15개)
    num_measurements = 15
    mask = torch.zeros(batch_size, 1, height, width, device=device)
    
    for _ in range(num_measurements):
        y_pos = torch.randint(20, height-20, (1,)).item()
        x_pos = torch.randint(20, width-20, (1,)).item()
        mask[0, 0, y_pos, x_pos] = 1.0
    
    # 측정값 = GT에서 노이즈 추가
    measured_values = gt * mask
    noise_scale = 0.05
    measured_values += torch.randn_like(measured_values) * noise_scale * mask
    
    # 6채널 입력 구성
    inp[:, 0:1] = measured_values  # 측정값
    inp[:, 1:2] = mask            # 마스크
    inp[:, 2:3] = torch.log1p(measured_values + 1e-8)  # 로그 측정값
    
    # 좌표 채널
    y_norm = (y_coords / (height - 1) * 2 - 1).unsqueeze(0).unsqueeze(0)
    x_norm = (x_coords / (width - 1) * 2 - 1).unsqueeze(0).unsqueeze(0)
    inp[:, 3:4] = y_norm
    inp[:, 4:5] = x_norm
    
    # 거리 맵
    distance_map = torch.zeros_like(mask[0, 0])
    measured_points = torch.where(mask[0, 0] > 0)
    if len(measured_points[0]) > 0:
        for y, x in zip(measured_points[0], measured_points[1]):
            distances = torch.sqrt((y_coords - y)**2 + (x_coords - x)**2)
            distance_map = torch.minimum(distance_map, distances) if distance_map.max() > 0 else distances
    
    distance_map = distance_map / distance_map.max() if distance_map.max() > 0 else distance_map
    inp[:, 5:6] = distance_map.unsqueeze(0).unsqueeze(0)
    
    return inp, gt, mask

def test_measurement_preservation():
    """측정값 보존 기능 테스트"""
    print("🔍 측정값 직접 보존 기능 테스트")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 테스트 데이터 생성
    inp, gt, mask = create_test_data()
    measured_values = inp[:, 0:1]
    
    print(f"📊 테스트 데이터:")
    print(f"   입력 크기: {inp.shape}")
    print(f"   GT 크기: {gt.shape}")
    print(f"   측정점 수: {mask.sum().int().item()}")
    print(f"   측정값 범위: [{measured_values[mask > 0].min():.4f}, {measured_values[mask > 0].max():.4f}]")
    
    # 모델 생성 및 테스트
    models = {
        "Simplified ConvNeXt": SimplifiedConvNeXtPGNN(in_channels=6, pred_scale=1.0),
        "Uncertainty-Aware": UncertaintyAwareConvNeXtPGNN(in_channels=6, pred_scale=1.0, max_prediction_distance=50)
    }
    
    results = {}
    
    with torch.no_grad():
        for model_name, model in models.items():
            model.to(device).eval()
            
            print(f"\n🔧 {model_name} 모델 테스트")
            
            if "Uncertainty" in model_name:
                pred, confidence, uncertainty, distance_maps = model(inp)
                results[model_name] = {
                    'prediction': pred,
                    'confidence': confidence,
                    'uncertainty': uncertainty,
                    'extra_outputs': True
                }
            else:
                pred = model(inp)
                results[model_name] = {
                    'prediction': pred,
                    'extra_outputs': False
                }
            
            # 측정값 보존 검증
            measured_positions = mask > 0
            pred_at_measurements = pred[measured_positions]
            actual_measurements = measured_values[measured_positions]
            
            # 완벽한 보존 확인 (오차가 거의 0이어야 함)
            preservation_error = F.mse_loss(pred_at_measurements, actual_measurements)
            max_preservation_error = torch.abs(pred_at_measurements - actual_measurements).max()
            
            print(f"   측정값 보존 MSE: {preservation_error:.8f}")
            print(f"   최대 보존 오차: {max_preservation_error:.8f}")
            
            if preservation_error < 1e-6:
                print("   ✅ 측정값 완벽 보존 확인!")
            else:
                print("   ❌ 측정값 보존 실패")
            
            # 예측 통계
            print(f"   예측값 범위: [{pred.min():.4f}, {pred.max():.4f}]")
            print(f"   예측값 평균: {pred.mean():.4f}")
            
            # 스파이크 검출 (측정점과 주변의 급격한 차이)
            spike_count = detect_spikes(pred, mask, threshold=0.1)
            print(f"   감지된 스파이크: {spike_count}개")
    
    # 시각화
    save_comparison_visualization(inp, gt, results, Path("test_measurement_preservation_results.png"))
    print(f"\n📋 결과 시각화 저장: test_measurement_preservation_results.png")
    
    return results

def detect_spikes(pred, mask, threshold=0.1):
    """측정점에서 스파이크 감지"""
    # 3x3 평균 필터
    kernel = torch.ones(1, 1, 3, 3, device=pred.device) / 9.0
    pred_smooth = F.conv2d(pred, kernel, padding=1)
    
    # 측정점에서의 급격한 차이
    spike_intensity = torch.abs(pred - pred_smooth) * mask
    spikes = spike_intensity > threshold
    
    return spikes.sum().item()

def save_comparison_visualization(inp, gt, results, save_path):
    """비교 시각화 저장"""
    mask = inp[:, 1:2]
    measured_values = inp[:, 0:1]
    
    num_models = len(results)
    fig, axes = plt.subplots(2, num_models + 2, figsize=(4 * (num_models + 2), 8))
    
    # 상단 행: 입력 및 각 모델 예측
    axes[0, 0].imshow(measured_values[0, 0].cpu().numpy(), cmap='hot', origin='lower')
    axes[0, 0].set_title('Input: Measured Values')
    axes[0, 0].axis('off')
    
    # 측정점 표시
    y_coords, x_coords = torch.where(mask[0, 0] > 0)
    axes[0, 0].scatter(x_coords.cpu(), y_coords.cpu(), c='blue', s=20, alpha=0.8)
    
    axes[0, 1].imshow(gt[0, 0].cpu().numpy(), cmap='hot', origin='lower')
    axes[0, 1].set_title('Ground Truth')
    axes[0, 1].axis('off')
    
    for i, (model_name, result) in enumerate(results.items()):
        pred = result['prediction']
        axes[0, i + 2].imshow(pred[0, 0].cpu().numpy(), cmap='hot', origin='lower')
        axes[0, i + 2].set_title(f'{model_name}')
        axes[0, i + 2].axis('off')
        
        # 측정점 표시
        axes[0, i + 2].scatter(x_coords.cpu(), y_coords.cpu(), c='blue', s=20, alpha=0.8)
    
    # 하단 행: 오차 분석
    axes[1, 0].axis('off')  # 빈 공간
    
    axes[1, 1].imshow(gt[0, 0].cpu().numpy(), cmap='hot', origin='lower')
    axes[1, 1].set_title('Ground Truth (Reference)')
    axes[1, 1].axis('off')
    
    for i, (model_name, result) in enumerate(results.items()):
        pred = result['prediction']
        error_map = torch.abs(pred - gt)[0, 0].cpu().numpy()
        
        im = axes[1, i + 2].imshow(error_map, cmap='Reds', origin='lower')
        axes[1, i + 2].set_title(f'{model_name}\nAbsolute Error')
        axes[1, i + 2].axis('off')
        plt.colorbar(im, ax=axes[1, i + 2], fraction=0.046, pad=0.04)
        
        # 측정점 표시
        axes[1, i + 2].scatter(x_coords.cpu(), y_coords.cpu(), c='blue', s=15, alpha=0.8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # CUDA 메모리 정리
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 측정값 보존 테스트 실행
    results = test_measurement_preservation()
    
    print("\n🎉 측정값 직접 보존 기능 테스트 완료!")
    print("=" * 50)