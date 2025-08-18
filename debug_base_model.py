#!/usr/bin/env python3
"""
베이스 모델 자체의 문제점 진단
실제 데이터를 사용하여 ConvNeXt 모델의 출력과 그래디언트 분석
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT / "src"))

from model.conv_next_pgnn import ConvNeXtUNetWithMaskAttention as ConvNeXtUNetBase
from torch.utils.data import DataLoader

# Dataset loading
class RadFieldDataset:
    def __init__(self, npz_path: Path):
        self.data = np.load(npz_path)
        if "inp" in self.data:
            self.inp = self.data["inp"].astype(np.float32)
        else:
            chans = [self.data[k].astype(np.float32)
                      for k in ["M", "mask", "logM", "X", "Y", "distance"]]
            self.inp = np.stack(chans, axis=1)
        self.gt = self.data["gt"].astype(np.float32)
        
    def get_batch(self, batch_size=2):
        """배치 샘플 반환"""
        indices = np.random.choice(len(self.inp), batch_size, replace=False)
        inp_batch = torch.from_numpy(self.inp[indices])
        gt_batch = torch.from_numpy(self.gt[indices])
        return inp_batch, gt_batch

def test_model_with_different_scales():
    """다양한 pred_scale로 모델 테스트"""
    print("🔍 다양한 pred_scale로 베이스 모델 테스트")
    print("=" * 60)
    
    # 실제 데이터 로드
    try:
        dataset = RadFieldDataset(Path("data/train.npz"))
        inp_batch, gt_batch = dataset.get_batch(batch_size=2)
        print(f"✅ 실제 데이터 로드 성공: {inp_batch.shape}")
    except Exception as e:
        print(f"❌ 데이터 로드 실패: {e}")
        return
    
    print(f"입력 통계:")
    for i in range(6):
        channel = inp_batch[:, i]
        print(f"  Ch{i}: [{channel.min().item():.3f}, {channel.max().item():.3f}] (mean: {channel.mean().item():.3f})")
    
    print(f"GT 통계: [{gt_batch.min().item():.3f}, {gt_batch.max().item():.3f}] (mean: {gt_batch.mean().item():.3f})")
    print()
    
    # 다양한 pred_scale 테스트
    scales = [0.1, 0.5, 1.0, 1.5, 2.0, 3.0]
    
    for scale in scales:
        print(f"📊 pred_scale = {scale}")
        print("-" * 30)
        
        model = ConvNeXtUNetBase(in_channels=6, pred_scale=scale)
        
        # 출력 분석
        with torch.no_grad():
            output = model(inp_batch)
            
        print(f"출력 형태: {output.shape}")
        print(f"출력 범위: [{output.min().item():.6f}, {output.max().item():.6f}]")
        print(f"출력 평균: {output.mean().item():.6f}")
        print(f"출력 표준편차: {output.std().item():.6f}")
        
        # 고유값 분석
        output_flat = output.flatten().numpy()
        unique_vals = np.unique(output_flat)
        print(f"고유값 개수: {len(unique_vals)}")
        
        if len(unique_vals) < 10:
            print(f"⚠️  고유값 샘플: {unique_vals[:5]}")
        
        # 그래디언트 분석
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        
        optimizer.zero_grad()
        output = model(inp_batch)
        loss = nn.MSELoss()(output, gt_batch)
        loss.backward()
        
        # 그래디언트 통계
        grad_norms = []
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norms.append(param.grad.norm().item())
        
        total_grad = sum(grad_norms)
        avg_grad = total_grad / len(grad_norms) if grad_norms else 0.0
        
        print(f"손실: {loss.item():.6f}")
        print(f"총 그래디언트 노름: {total_grad:.6f}")
        print(f"평균 그래디언트 노름: {avg_grad:.2e}")
        
        if avg_grad < 1e-8:
            print("🚨 그래디언트 소멸!")
        elif len(unique_vals) < 10:
            print("🚨 출력 saturation!")
        else:
            print("✅ 정상 동작")
        
        print()

def analyze_model_components():
    """모델 구성요소별 분석"""
    print("🔍 모델 구성요소별 분석")
    print("=" * 60)
    
    # 실제 데이터 로드
    try:
        dataset = RadFieldDataset(Path("data/train.npz"))
        inp_batch, gt_batch = dataset.get_batch(batch_size=2)
    except Exception as e:
        print(f"❌ 데이터 로드 실패: {e}")
        return
    
    model = ConvNeXtUNetBase(in_channels=6, pred_scale=1.0)
    
    # Hook으로 중간 출력 캡처
    activations = {}
    
    def save_activation(name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                activations[name] = output.detach()
        return hook
    
    # 주요 레이어에 hook 등록
    model.convnext.stem.register_forward_hook(save_activation('stem'))
    model.convnext.stages[0].register_forward_hook(save_activation('stage0'))
    model.convnext.stages[1].register_forward_hook(save_activation('stage1'))
    model.convnext.stages[2].register_forward_hook(save_activation('stage2'))
    model.convnext.stages[3].register_forward_hook(save_activation('stage3'))
    
    if hasattr(model, 'decoder_head'):
        model.decoder_head.register_forward_hook(save_activation('decoder_head'))
    
    # Forward pass
    with torch.no_grad():
        output = model(inp_batch)
    
    print("중간 activation 분석:")
    for name, activation in activations.items():
        act_flat = activation.flatten()
        unique_vals = len(torch.unique(act_flat))
        
        print(f"{name:15} - Shape: {str(activation.shape):20} | "
              f"Range: [{activation.min().item():.4f}, {activation.max().item():.4f}] | "
              f"Unique: {unique_vals:6} | "
              f"Std: {activation.std().item():.4f}")
        
        if unique_vals < 10:
            print(f"  ⚠️  {name}에서 saturation 발생!")
    
    print(f"\n최종 출력:")
    print(f"Range: [{output.min().item():.6f}, {output.max().item():.6f}]")
    print(f"Unique values: {len(torch.unique(output.flatten()))}")

def main():
    print("🚨 베이스 모델 문제 진단")
    print("=" * 60)
    
    # 모델 가중치 초기화 상태 확인
    model = ConvNeXtUNetBase(in_channels=6, pred_scale=1.0)
    
    print("모델 파라미터 통계:")
    total_params = 0
    zero_params = 0
    
    for name, param in model.named_parameters():
        param_count = param.numel()
        param_zeros = (param == 0).sum().item()
        total_params += param_count
        zero_params += param_zeros
        
        if param_count < 100:  # 작은 파라미터만 출력
            print(f"{name:30} - Shape: {str(param.shape):15} | "
                  f"Range: [{param.min().item():.4f}, {param.max().item():.4f}] | "
                  f"Zeros: {param_zeros}/{param_count}")
    
    print(f"\n총 파라미터: {total_params:,}")
    print(f"0인 파라미터: {zero_params:,} ({100*zero_params/total_params:.1f}%)")
    
    print("\n" + "="*60)
    test_model_with_different_scales()
    
    print("="*60)
    analyze_model_components()

if __name__ == "__main__":
    main()