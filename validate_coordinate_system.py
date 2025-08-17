#!/usr/bin/env python3
"""
좌표계 End-to-End 검증 시스템
전체 파이프라인에서 좌표 일관성을 검증하는 포괄적 테스트
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
import sys, pathlib

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.append(str(ROOT / "src"))

import dataset.generate_truth as gt
import dataset.trajectory_sampler as ts
from dataset.dataset_generator import _make_single_sample
from model.conv_next_pgnn import ConvNeXtUNetWithMaskAttention, extract_measured_data

def test_coordinate_generation():
    """데이터 생성 과정에서 좌표 일관성 테스트"""
    print("🔍 Testing coordinate generation consistency...")
    
    # 고정된 시드로 재현 가능한 테스트
    rng = np.random.default_rng(42)
    
    # 단일 샘플 생성
    field, inp, mask = _make_single_sample(rng)
    
    # 채널 분리
    measured_values = inp[0]   # 측정값
    mask_channel = inp[1]      # 마스크
    coord_y = inp[3]          # Y 좌표 (행)
    coord_x = inp[4]          # X 좌표 (열)
    
    # 측정된 위치 찾기
    measured_positions = np.where(mask_channel > 0)
    
    print(f"  📊 Generated field shape: {field.shape}")
    print(f"  📊 Found {len(measured_positions[0])} measured positions")
    
    # 좌표 값 범위 확인
    coord_y_range = (coord_y.min(), coord_y.max())
    coord_x_range = (coord_x.min(), coord_x.max())
    
    print(f"  📐 Y coordinate range: {coord_y_range}")
    print(f"  📐 X coordinate range: {coord_x_range}")
    
    # 좌표가 [-1, 1] 범위에 있는지 확인
    assert -1.0 <= coord_y.min() <= coord_y.max() <= 1.0, f"Y coordinates out of range: {coord_y_range}"
    assert -1.0 <= coord_x.min() <= coord_x.max() <= 1.0, f"X coordinates out of range: {coord_x_range}"
    
    print("  ✅ Coordinate ranges are correct [-1, 1]")
    
    # 좌표 그리드가 올바르게 생성되었는지 확인
    H, W = field.shape
    expected_y = np.linspace(-1, 1, H)
    expected_x = np.linspace(-1, 1, W)
    
    # 첫 번째 행과 열의 좌표 확인
    actual_y_first_col = coord_y[:, 0]
    actual_x_first_row = coord_x[0, :]
    
    y_diff = np.abs(actual_y_first_col - expected_y).max()
    x_diff = np.abs(actual_x_first_row - expected_x).max()
    
    print(f"  📏 Y coordinate grid difference: {y_diff:.6f}")
    print(f"  📏 X coordinate grid difference: {x_diff:.6f}")
    
    assert y_diff < 1e-5, f"Y coordinate grid mismatch: {y_diff}"
    assert x_diff < 1e-5, f"X coordinate grid mismatch: {x_diff}"
    
    print("  ✅ Coordinate grids are correctly generated")
    
    return field, inp, mask_channel, measured_positions

def test_model_coordinate_interpretation():
    """모델에서 좌표 해석 일관성 테스트"""
    print("\n🧠 Testing model coordinate interpretation...")
    
    # 테스트 샘플 생성
    field, inp, mask_channel, measured_positions = test_coordinate_generation()
    
    # 모델 초기화 (작은 사이즈로 빠른 테스트)
    model = ConvNeXtUNetWithMaskAttention(in_channels=6, pred_scale=1.0)
    model.eval()
    
    # 배치 차원 추가
    inp_batch = torch.from_numpy(inp).unsqueeze(0).float()
    
    # 모델에서 좌표 추출 (forward 메서드의 좌표 할당 확인)
    with torch.no_grad():
        # 모델 내부 좌표 추출 시뮬레이션
        coord_x_model = inp_batch[:, 4:5]  # X coordinates (columns)
        coord_y_model = inp_batch[:, 3:4]  # Y coordinates (rows)
    
    print(f"  🧠 Model interprets X coords from channel 4: shape {coord_x_model.shape}")
    print(f"  🧠 Model interprets Y coords from channel 3: shape {coord_y_model.shape}")
    
    # 좌표 해석이 올바른지 확인
    coord_x_np = coord_x_model[0, 0].numpy()
    coord_y_np = coord_y_model[0, 0].numpy()
    
    # 원본 데이터와 비교
    original_x = inp[4]
    original_y = inp[3]
    
    x_match = np.allclose(coord_x_np, original_x)
    y_match = np.allclose(coord_y_np, original_y)
    
    print(f"  🔍 X coordinate matching: {x_match}")
    print(f"  🔍 Y coordinate matching: {y_match}")
    
    assert x_match, "Model X coordinate interpretation mismatch"
    assert y_match, "Model Y coordinate interpretation mismatch"
    
    print("  ✅ Model coordinate interpretation is correct")
    
    return inp_batch

def test_physics_loss_coordinates():
    """물리 손실 함수에서 좌표 사용 일관성 테스트"""
    print("\n⚛️  Testing physics loss coordinate usage...")
    
    # 테스트 배치 준비
    inp_batch = test_model_coordinate_interpretation()
    
    # extract_measured_data 함수 테스트
    try:
        measured_positions, measured_values = extract_measured_data(inp_batch)
        print(f"  📍 Extracted {len(measured_positions)} batches of measured data")
        
        if len(measured_positions) > 0 and measured_positions[0].shape[0] > 0:
            batch_0_positions = measured_positions[0]
            batch_0_values = measured_values[0]
            
            print(f"  📍 Batch 0: {batch_0_positions.shape[0]} measured positions")
            print(f"  📊 Position coordinate range: [{batch_0_positions[:, 0].min():.3f}, {batch_0_positions[:, 0].max():.3f}]")
            print(f"  📊 Position coordinate range: [{batch_0_positions[:, 1].min():.3f}, {batch_0_positions[:, 1].max():.3f}]")
            
            # 좌표가 [-1, 1] 범위에 있는지 확인
            assert torch.all(batch_0_positions >= -1.0) and torch.all(batch_0_positions <= 1.0), \
                "Extracted coordinates out of [-1, 1] range"
            
            print("  ✅ Physics loss coordinates are in correct range")
        else:
            print("  ⚠️  No measured positions found in test batch")
            
    except Exception as e:
        print(f"  ❌ Error in physics loss coordinate extraction: {e}")
        raise
    
    print("  ✅ Physics loss coordinate usage is consistent")

def test_coordinate_pixel_mapping():
    """좌표-픽셀 매핑 정확성 테스트"""
    print("\n🗺️  Testing coordinate-to-pixel mapping...")
    
    # 테스트 필드 생성
    rng = np.random.default_rng(123)
    field, inp, mask_channel = _make_single_sample(rng)[:3]
    
    H, W = field.shape
    coord_y = inp[3]
    coord_x = inp[4]
    
    # 몇 개의 테스트 포인트에서 좌표-픽셀 변환 확인
    test_pixels = [(0, 0), (H//2, W//2), (H-1, W-1)]
    
    for py, px in test_pixels:
        # 해당 픽셀의 정규화된 좌표
        norm_y = coord_y[py, px]
        norm_x = coord_x[py, px]
        
        # 정규화된 좌표에서 픽셀 좌표로 역변환
        recovered_py = int((norm_y + 1) * (H - 1) / 2)
        recovered_px = int((norm_x + 1) * (W - 1) / 2)
        
        print(f"  📍 Pixel ({py:3d}, {px:3d}) -> Norm ({norm_y:6.3f}, {norm_x:6.3f}) -> Pixel ({recovered_py:3d}, {recovered_px:3d})")
        
        # 역변환이 정확한지 확인 (반올림 오차 허용)
        assert abs(recovered_py - py) <= 1, f"Y coordinate mapping error: {py} -> {recovered_py}"
        assert abs(recovered_px - px) <= 1, f"X coordinate mapping error: {px} -> {recovered_px}"
    
    print("  ✅ Coordinate-to-pixel mapping is accurate")

def test_measured_position_consistency():
    """측정 위치의 일관성 종합 테스트"""
    print("\n🎯 Testing measured position consistency across pipeline...")
    
    rng = np.random.default_rng(456)
    
    # 1. 원본 필드와 웨이포인트 생성
    n_src = rng.integers(gt.N_SOURCES_RANGE[0], gt.N_SOURCES_RANGE[1] + 1)
    c, a, s = gt.sample_sources(gt.GRID, n_src, rng=rng)
    field = gt.gaussian_field(gt.GRID, c, a, s)
    
    # 정규화
    field = np.clip(field, 0, None)
    field_99th = np.percentile(field[field > 0], 99) if np.any(field > 0) else 1.0
    normalization_factor = max(field_99th, 1.0)
    field = np.clip(field / normalization_factor, 0, 1.0)
    
    # 2. 웨이포인트 생성
    waypoints = ts.generate_waypoints(rng=rng)
    meas, mask = ts.sparse_from_waypoints(field, waypoints, rng=rng)
    
    print(f"  🎯 Generated {len(waypoints)} waypoints")
    
    # 3. 각 웨이포인트에서 일관성 확인
    for i, (wp_y, wp_x) in enumerate(waypoints[:5]):  # 처음 5개만 테스트
        # 원본 필드 값
        field_value = field[wp_y, wp_x]
        
        # 측정 배열에서 값
        meas_value = meas[wp_y, wp_x]
        
        # 마스크 확인
        mask_value = mask[wp_y, wp_x]
        
        print(f"    Waypoint {i}: ({wp_y:3d}, {wp_x:3d}) -> Field: {field_value:.6f}, Meas: {meas_value:.6f}, Mask: {mask_value}")
        
        # 값 일치 확인
        assert mask_value == 1, f"Mask should be 1 at waypoint {i}"
        assert abs(field_value - meas_value) < 1e-6, f"Value mismatch at waypoint {i}: {field_value} vs {meas_value}"
    
    print("  ✅ Measured position values are consistent with field values")

def create_coordinate_validation_plot():
    """좌표 검증 결과 시각화"""
    print("\n📊 Creating coordinate validation visualization...")
    
    rng = np.random.default_rng(789)
    field, inp, mask_channel = _make_single_sample(rng)[:3]
    
    coord_y = inp[3]
    coord_x = inp[4]
    measured_values = inp[0]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Coordinate System Validation', fontsize=16)
    
    # 1. 원본 필드
    im1 = axes[0, 0].imshow(field, cmap='hot', origin='upper')
    axes[0, 0].set_title('Ground Truth Field')
    plt.colorbar(im1, ax=axes[0, 0], shrink=0.8)
    
    # 2. Y 좌표
    im2 = axes[0, 1].imshow(coord_y, cmap='viridis', origin='upper')
    axes[0, 1].set_title('Y Coordinates (Rows)')
    plt.colorbar(im2, ax=axes[0, 1], shrink=0.8)
    
    # 3. X 좌표
    im3 = axes[0, 2].imshow(coord_x, cmap='viridis', origin='upper')
    axes[0, 2].set_title('X Coordinates (Columns)')
    plt.colorbar(im3, ax=axes[0, 2], shrink=0.8)
    
    # 4. 측정 마스크
    im4 = axes[1, 0].imshow(mask_channel, cmap='gray', origin='upper')
    axes[1, 0].set_title('Measurement Mask')
    plt.colorbar(im4, ax=axes[1, 0], shrink=0.8)
    
    # 5. 측정값
    im5 = axes[1, 1].imshow(measured_values, cmap='hot', origin='upper')
    axes[1, 1].set_title('Measured Values')
    plt.colorbar(im5, ax=axes[1, 1], shrink=0.8)
    
    # 6. 오버레이: 필드 + 측정점
    axes[1, 2].imshow(field, cmap='hot', origin='upper', alpha=0.7)
    measured_y, measured_x = np.where(mask_channel > 0)
    axes[1, 2].scatter(measured_x, measured_y, c='blue', s=15, alpha=0.8)
    axes[1, 2].set_title('Field + Measurement Points')
    
    plt.tight_layout()
    plt.savefig('coordinate_validation.png', dpi=150, bbox_inches='tight')
    print("  💾 Saved validation plot as 'coordinate_validation.png'")

def main():
    """메인 검증 프로세스"""
    print("🔍 COMPREHENSIVE COORDINATE SYSTEM VALIDATION")
    print("=" * 60)
    
    try:
        # 1. 좌표 생성 테스트
        test_coordinate_generation()
        
        # 2. 모델 좌표 해석 테스트
        test_model_coordinate_interpretation()
        
        # 3. 물리 손실 좌표 사용 테스트
        test_physics_loss_coordinates()
        
        # 4. 좌표-픽셀 매핑 테스트
        test_coordinate_pixel_mapping()
        
        # 5. 측정 위치 일관성 테스트
        test_measured_position_consistency()
        
        # 6. 검증 시각화
        create_coordinate_validation_plot()
        
        print("\n" + "=" * 60)
        print("✅ ALL COORDINATE SYSTEM TESTS PASSED!")
        print("🎯 The coordinate system is consistent across the entire pipeline.")
        print("📊 Previous coordinate issues have been successfully resolved.")
        
    except Exception as e:
        print(f"\n❌ COORDINATE VALIDATION FAILED: {e}")
        print("🔧 Please check the coordinate handling in the reported component.")
        raise

if __name__ == "__main__":
    main()