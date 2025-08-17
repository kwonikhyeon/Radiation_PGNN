#!/usr/bin/env python3
"""
훈련 문제 진단 도구
모델, 데이터, 손실 함수의 문제점을 체계적으로 진단
"""
import numpy as np
import torch
import torch.nn.functional as F
import sys, pathlib
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.append(str(ROOT / "src"))

from model.conv_next_pgnn import ConvNeXtUNetWithMaskAttention
from dataset.dataset_generator import RadiationDataset

def diagnose_data():
    """데이터셋 진단"""
    print("🔍 DIAGNOSING DATASET...")
    
    try:
        train_ds = RadiationDataset("train")
        val_ds = RadiationDataset("val")
        
        print(f"  📊 Train samples: {len(train_ds)}")
        print(f"  📊 Val samples: {len(val_ds)}")
        
        # 샘플 데이터 확인
        inp, gt = train_ds[0]
        print(f"  📐 Input shape: {inp.shape}")
        print(f"  📐 GT shape: {gt.shape}")
        
        # 데이터 범위 확인
        print(f"  📊 Input ranges:")
        for i in range(inp.shape[0]):
            print(f"    Ch{i}: [{inp[i].min():.6f}, {inp[i].max():.6f}]")
        
        print(f"  📊 GT range: [{gt.min():.6f}, {gt.max():.6f}]")
        
        # NaN/Inf 확인
        has_nan_inp = torch.isnan(inp).any()
        has_inf_inp = torch.isinf(inp).any()
        has_nan_gt = torch.isnan(gt).any()
        has_inf_gt = torch.isinf(gt).any()
        
        print(f"  🔍 Input NaN: {has_nan_inp}, Inf: {has_inf_inp}")
        print(f"  🔍 GT NaN: {has_nan_gt}, Inf: {has_inf_gt}")
        
        if has_nan_inp or has_inf_inp or has_nan_gt or has_inf_gt:
            print("  ❌ FOUND NaN/Inf IN DATA!")
            return False
        
        # 측정값 일관성 재확인
        mask = inp[1]
        measured_values = inp[0]
        measured_positions = torch.where(mask > 0)
        
        if len(measured_positions[0]) > 0:
            consistency_errors = 0
            for i in range(min(10, len(measured_positions[0]))):
                y, x = measured_positions[0][i], measured_positions[1][i]
                field_val = gt[0, y, x]
                meas_val = measured_values[y, x]
                if abs(field_val - meas_val) > 1e-5:
                    consistency_errors += 1
            
            print(f"  🎯 Measurement consistency errors: {consistency_errors}/10")
            
            if consistency_errors > 0:
                print("  ❌ MEASUREMENT INCONSISTENCY DETECTED!")
                return False
        
        print("  ✅ Dataset appears healthy")
        return True
        
    except Exception as e:
        print(f"  ❌ Dataset error: {e}")
        return False

def diagnose_model():
    """모델 진단"""
    print("\n🧠 DIAGNOSING MODEL...")
    
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = ConvNeXtUNetWithMaskAttention(in_channels=6, pred_scale=2.0).to(device)
        
        # 모델 파라미터 확인
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"  📊 Total parameters: {total_params:,}")
        print(f"  📊 Trainable parameters: {trainable_params:,}")
        print(f"  💾 Model size: {total_params * 4 / 1024**2:.1f} MB")
        
        if total_params > 150_000_000:  # 150M 파라미터 이상
            print("  ⚠️  Model is very large, may cause memory issues")
        
        # 더미 입력으로 순전파 테스트
        dummy_input = torch.randn(2, 6, 256, 256).to(device)
        
        print(f"  🔄 Testing forward pass...")
        model.eval()
        with torch.no_grad():
            try:
                output = model(dummy_input)
                print(f"  📐 Output shape: {output.shape}")
                print(f"  📊 Output range: [{output.min():.6f}, {output.max():.6f}]")
                
                # NaN/Inf 확인
                has_nan = torch.isnan(output).any()
                has_inf = torch.isinf(output).any()
                
                if has_nan or has_inf:
                    print(f"  ❌ Model output contains NaN: {has_nan}, Inf: {has_inf}")
                    return False
                
                print("  ✅ Forward pass successful")
                
            except Exception as e:
                print(f"  ❌ Forward pass failed: {e}")
                return False
        
        # 그래디언트 테스트
        model.train()
        dummy_target = torch.randn_like(output)
        
        try:
            loss = F.mse_loss(output, dummy_target)
            loss.backward()
            
            # 그래디언트 확인
            grad_norms = []
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    grad_norms.append(grad_norm)
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        print(f"  ❌ NaN/Inf gradient in {name}")
                        return False
            
            avg_grad_norm = np.mean(grad_norms) if grad_norms else 0
            max_grad_norm = np.max(grad_norms) if grad_norms else 0
            
            print(f"  📊 Average gradient norm: {avg_grad_norm:.6f}")
            print(f"  📊 Max gradient norm: {max_grad_norm:.6f}")
            
            if max_grad_norm > 100:
                print(f"  ⚠️  Very large gradients detected, may cause instability")
            elif max_grad_norm < 1e-7:
                print(f"  ⚠️  Very small gradients detected, may indicate vanishing gradients")
            
            print("  ✅ Gradient computation successful")
            
        except Exception as e:
            print(f"  ❌ Gradient computation failed: {e}")
            return False
        
        return True
        
    except Exception as e:
        print(f"  ❌ Model initialization error: {e}")
        return False

def diagnose_training_step():
    """훈련 스텝 진단"""
    print("\n🎯 DIAGNOSING TRAINING STEP...")
    
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = ConvNeXtUNetWithMaskAttention(in_channels=6, pred_scale=2.0).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        # 실제 데이터로 테스트
        train_ds = RadiationDataset("train")
        inp, gt = train_ds[0]
        inp = inp.unsqueeze(0).to(device)  # 배치 차원 추가
        gt = gt.unsqueeze(0).to(device)
        
        print(f"  📊 Batch input range: [{inp.min():.6f}, {inp.max():.6f}]")
        print(f"  📊 Batch GT range: [{gt.min():.6f}, {gt.max():.6f}]")
        
        model.train()
        
        # 여러 스텝 실행하여 손실 변화 확인
        losses = []
        for step in range(5):
            optimizer.zero_grad()
            
            pred = model(inp)
            
            # 단순한 MSE 손실만 사용
            loss = F.mse_loss(pred, gt)
            
            print(f"  Step {step}: Loss = {loss.item():.6f}, Pred range = [{pred.min():.6f}, {pred.max():.6f}]")
            
            loss.backward()
            
            # 그래디언트 노름 확인
            total_grad_norm = 0
            for param in model.parameters():
                if param.grad is not None:
                    total_grad_norm += param.grad.norm().item() ** 2
            total_grad_norm = total_grad_norm ** 0.5
            
            print(f"    Gradient norm: {total_grad_norm:.6f}")
            
            optimizer.step()
            losses.append(loss.item())
        
        # 손실 변화 분석
        loss_changes = [losses[i+1] - losses[i] for i in range(len(losses)-1)]
        print(f"  📈 Loss changes: {loss_changes}")
        
        if all(change >= 0 for change in loss_changes):
            print("  ❌ Loss is not decreasing - optimization problem!")
            return False
        elif all(abs(change) < 1e-8 for change in loss_changes):
            print("  ❌ Loss is not changing - no learning!")
            return False
        else:
            print("  ✅ Loss is decreasing - training step working")
            return True
            
    except Exception as e:
        print(f"  ❌ Training step error: {e}")
        return False

def diagnose_measurement_preservation():
    """측정값 보존 메커니즘 진단"""
    print("\n🎯 DIAGNOSING MEASUREMENT PRESERVATION...")
    
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = ConvNeXtUNetWithMaskAttention(in_channels=6, pred_scale=2.0).to(device)
        
        # 실제 데이터 사용
        train_ds = RadiationDataset("train")
        inp, gt = train_ds[0]
        inp = inp.unsqueeze(0).to(device)
        gt = gt.unsqueeze(0).to(device)
        
        mask = inp[:, 1:2]  # 측정 마스크
        measured_values = inp[:, 0:1]  # 측정값
        
        model.eval()
        with torch.no_grad():
            pred = model(inp)
        
        # 측정 위치에서 값 확인
        measured_positions = torch.where(mask[0, 0] > 0)
        
        if len(measured_positions[0]) > 0:
            preservation_errors = 0
            max_error = 0.0
            
            for i in range(min(10, len(measured_positions[0]))):
                y, x = measured_positions[0][i], measured_positions[1][i]
                
                gt_val = gt[0, 0, y, x].item()
                pred_val = pred[0, 0, y, x].item()
                meas_val = measured_values[0, 0, y, x].item()
                
                error_gt_pred = abs(gt_val - pred_val)
                error_meas_pred = abs(meas_val - pred_val)
                
                max_error = max(max_error, error_meas_pred)
                
                if error_meas_pred > 1e-4:  # 허용 오차
                    preservation_errors += 1
                
                if i < 3:  # 처음 3개만 출력
                    print(f"    Pos ({y:3d},{x:3d}): GT={gt_val:.6f}, Pred={pred_val:.6f}, Meas={meas_val:.6f}")
            
            print(f"  📊 Measurement preservation errors: {preservation_errors}/10")
            print(f"  📊 Max preservation error: {max_error:.6f}")
            
            if preservation_errors > 3:
                print("  ❌ Measurement preservation is failing!")
                return False
            else:
                print("  ✅ Measurement preservation working")
                return True
        else:
            print("  ⚠️  No measured positions found")
            return True
            
    except Exception as e:
        print(f"  ❌ Measurement preservation check error: {e}")
        return False

def create_diagnostic_plot():
    """진단 결과 시각화"""
    print("\n📊 CREATING DIAGNOSTIC VISUALIZATION...")
    
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = ConvNeXtUNetWithMaskAttention(in_channels=6, pred_scale=2.0).to(device)
        
        train_ds = RadiationDataset("train")
        inp, gt = train_ds[0]
        inp_batch = inp.unsqueeze(0).to(device)
        gt_batch = gt.unsqueeze(0).to(device)
        
        model.eval()
        with torch.no_grad():
            pred = model(inp_batch)
        
        # CPU로 이동
        inp_np = inp.numpy()
        gt_np = gt.numpy()
        pred_np = pred[0].cpu().numpy()
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Training Issue Diagnostic Visualization', fontsize=16)
        
        # 1. GT
        im1 = axes[0, 0].imshow(gt_np[0], cmap='hot', origin='upper')
        axes[0, 0].set_title('Ground Truth')
        plt.colorbar(im1, ax=axes[0, 0], shrink=0.8)
        
        # 2. 초기 예측 (훈련되지 않은 모델)
        im2 = axes[0, 1].imshow(pred_np[0], cmap='hot', origin='upper')
        axes[0, 1].set_title('Model Prediction (Untrained)')
        plt.colorbar(im2, ax=axes[0, 1], shrink=0.8)
        
        # 3. 측정 마스크
        mask_np = inp_np[1]
        im3 = axes[0, 2].imshow(mask_np, cmap='gray', origin='upper')
        axes[0, 2].set_title('Measurement Mask')
        plt.colorbar(im3, ax=axes[0, 2], shrink=0.8)
        
        # 4. 측정값
        measured_np = inp_np[0]
        im4 = axes[1, 0].imshow(measured_np, cmap='hot', origin='upper')
        axes[1, 0].set_title('Measured Values')
        plt.colorbar(im4, ax=axes[1, 0], shrink=0.8)
        
        # 5. 차이 맵
        diff_np = np.abs(gt_np[0] - pred_np[0])
        im5 = axes[1, 1].imshow(diff_np, cmap='plasma', origin='upper')
        axes[1, 1].set_title('|GT - Prediction| Difference')
        plt.colorbar(im5, ax=axes[1, 1], shrink=0.8)
        
        # 6. 히스토그램 비교
        axes[1, 2].hist(gt_np[0].flatten(), bins=50, alpha=0.7, label='GT', density=True)
        axes[1, 2].hist(pred_np[0].flatten(), bins=50, alpha=0.7, label='Pred', density=True)
        axes[1, 2].set_title('Value Distribution')
        axes[1, 2].legend()
        axes[1, 2].set_xlabel('Value')
        axes[1, 2].set_ylabel('Density')
        
        plt.tight_layout()
        plt.savefig('training_diagnostic.png', dpi=150, bbox_inches='tight')
        print("  💾 Saved diagnostic plot as 'training_diagnostic.png'")
        
    except Exception as e:
        print(f"  ❌ Visualization error: {e}")

def main():
    """메인 진단 프로세스"""
    print("🚨 COMPREHENSIVE TRAINING ISSUE DIAGNOSIS")
    print("=" * 60)
    
    results = []
    
    # 1. 데이터 진단
    data_ok = diagnose_data()
    results.append(("Data", data_ok))
    
    # 2. 모델 진단
    model_ok = diagnose_model()
    results.append(("Model", model_ok))
    
    # 3. 훈련 스텝 진단
    training_ok = diagnose_training_step()
    results.append(("Training Step", training_ok))
    
    # 4. 측정값 보존 진단
    preservation_ok = diagnose_measurement_preservation()
    results.append(("Measurement Preservation", preservation_ok))
    
    # 5. 시각화 생성
    create_diagnostic_plot()
    
    # 결과 요약
    print("\n" + "=" * 60)
    print("📋 DIAGNOSIS SUMMARY:")
    
    all_passed = True
    for component, status in results:
        status_icon = "✅" if status else "❌"
        print(f"  {status_icon} {component}: {'PASS' if status else 'FAIL'}")
        if not status:
            all_passed = False
    
    print("\n🎯 RECOMMENDATIONS:")
    if not all_passed:
        print("  1. Use the simplified training script: train_conv_next_simple.py")
        print("  2. Start with basic MSE loss only")
        print("  3. Check for gradient clipping and learning rate")
        print("  4. Monitor for NaN/Inf values during training")
    else:
        print("  All components appear healthy. The issue may be in loss function complexity.")
        print("  Try the simplified training approach.")

if __name__ == "__main__":
    main()