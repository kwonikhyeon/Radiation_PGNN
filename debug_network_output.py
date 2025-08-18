#!/usr/bin/env python3
"""
네트워크 출력 디버깅 스크립트
각 constraint method의 실제 출력 분포와 그래디언트 흐름 분석
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT / "src"))

from model.conv_next_pgnn import ConvNeXtUNetWithMaskAttention as ConvNeXtUNetBase
from torch.utils.data import DataLoader, Dataset

# 샘플 입력 생성 (실제 데이터 형태와 동일)
def create_sample_input(batch_size=2):
    """실제 데이터와 유사한 샘플 입력 생성"""
    # 6채널 입력: [M, mask, logM, X, Y, distance]
    sample_input = torch.randn(batch_size, 6, 256, 256)
    
    # 실제 데이터 분포에 맞게 조정
    sample_input[:, 0] = torch.clamp(sample_input[:, 0] * 0.1, 0, 1)  # M: 측정값 [0,1]
    sample_input[:, 1] = (sample_input[:, 1] > 0).float()  # mask: 이진값
    sample_input[:, 2] = torch.log(sample_input[:, 0] + 1e-8)  # logM
    
    # X, Y 좌표 생성 (올바른 형태로)
    x_coords = torch.linspace(-1, 1, 256).view(1, 1, 256).repeat(batch_size, 256, 1)
    y_coords = torch.linspace(-1, 1, 256).view(1, 256, 1).repeat(batch_size, 1, 256)
    
    sample_input[:, 3] = x_coords  # X
    sample_input[:, 4] = y_coords  # Y
    sample_input[:, 5] = torch.randn(batch_size, 256, 256) * 0.5 + 0.5  # distance
    
    return sample_input

# V2의 각 제약 방법 구현
class Method1_HardClip(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        
    def forward(self, x):
        pred = self.base_model(x)
        print(f"Hard Clip - Base output range: [{pred.min().item():.3f}, {pred.max().item():.3f}]")
        pred = pred / 4.0  # 스케일링
        print(f"Hard Clip - After scaling: [{pred.min().item():.3f}, {pred.max().item():.3f}]")
        pred = torch.clamp(pred, min=0.0, max=1.0)
        print(f"Hard Clip - After clamp: [{pred.min().item():.3f}, {pred.max().item():.3f}]")
        return pred

class Method2_Sigmoid(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        
    def forward(self, x):
        pred = self.base_model(x)
        print(f"Sigmoid - Base output range: [{pred.min().item():.3f}, {pred.max().item():.3f}]")
        pred = pred / 6.0  # 스케일링
        print(f"Sigmoid - After scaling: [{pred.min().item():.3f}, {pred.max().item():.3f}]")
        pred = torch.sigmoid(pred)
        print(f"Sigmoid - After sigmoid: [{pred.min().item():.3f}, {pred.max().item():.3f}]")
        return pred

class Method3_Tanh(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        
    def forward(self, x):
        pred = self.base_model(x)
        print(f"Tanh - Base output range: [{pred.min().item():.3f}, {pred.max().item():.3f}]")
        pred = pred / 3.0  # 스케일링
        print(f"Tanh - After scaling: [{pred.min().item():.3f}, {pred.max().item():.3f}]")
        pred = (torch.tanh(pred) + 1.0) / 2.0
        print(f"Tanh - After tanh+shift: [{pred.min().item():.3f}, {pred.max().item():.3f}]")
        return pred

def analyze_gradients(model, input_tensor, target):
    """그래디언트 흐름 분석"""
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    optimizer.zero_grad()
    output = model(input_tensor)
    loss = nn.MSELoss()(output, target)
    loss.backward()
    
    # 그래디언트 통계
    total_grad_norm = 0.0
    param_count = 0
    grad_info = []
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            total_grad_norm += grad_norm
            param_count += 1
            grad_info.append((name, grad_norm, param.grad.mean().item(), param.grad.std().item()))
    
    avg_grad_norm = total_grad_norm / param_count if param_count > 0 else 0.0
    
    return {
        'loss': loss.item(),
        'total_grad_norm': total_grad_norm,
        'avg_grad_norm': avg_grad_norm,
        'param_count': param_count,
        'grad_info': grad_info[-5:]  # 마지막 5개 레이어만
    }

def main():
    print("🔍 네트워크 출력 및 그래디언트 분석")
    print("=" * 60)
    
    # 샘플 입력 생성
    input_tensor = create_sample_input(batch_size=2)
    target = torch.rand(2, 1, 256, 256)  # 샘플 타겟
    
    print(f"입력 형태: {input_tensor.shape}")
    print(f"타겟 형태: {target.shape}")
    print()
    
    # 베이스 모델 생성
    base_model = ConvNeXtUNetBase(in_channels=6, pred_scale=1.0)
    
    methods = [
        ("Method1_HardClip", Method1_HardClip(base_model)),
        ("Method2_Sigmoid", Method2_Sigmoid(base_model)),
        ("Method3_Tanh", Method3_Tanh(base_model))
    ]
    
    results = {}
    
    for method_name, model in methods:
        print(f"\n📊 {method_name} 분석")
        print("-" * 40)
        
        # 출력 분석
        with torch.no_grad():
            output = model(input_tensor)
            
        print(f"출력 통계:")
        print(f"  - 형태: {output.shape}")
        print(f"  - 평균: {output.mean().item():.6f}")
        print(f"  - 표준편차: {output.std().item():.6f}")
        print(f"  - 최소값: {output.min().item():.6f}")
        print(f"  - 최대값: {output.max().item():.6f}")
        
        # 분포 분석
        output_flat = output.flatten().numpy()
        unique_values = len(np.unique(output_flat))
        print(f"  - 고유값 개수: {unique_values}")
        
        if unique_values < 10:
            print(f"  - ⚠️  너무 적은 고유값! (saturation 의심)")
        
        # 그래디언트 분석
        grad_info = analyze_gradients(model, input_tensor, target)
        print(f"\n그래디언트 분석:")
        print(f"  - 손실: {grad_info['loss']:.6f}")
        print(f"  - 총 그래디언트 노름: {grad_info['total_grad_norm']:.6f}")
        print(f"  - 평균 그래디언트 노름: {grad_info['avg_grad_norm']:.6f}")
        print(f"  - 파라미터 개수: {grad_info['param_count']}")
        
        if grad_info['avg_grad_norm'] < 1e-6:
            print(f"  - ⚠️  매우 작은 그래디언트! (학습 불가 상태)")
        
        results[method_name] = {
            'output_stats': {
                'mean': output.mean().item(),
                'std': output.std().item(), 
                'min': output.min().item(),
                'max': output.max().item(),
                'unique_values': unique_values
            },
            'grad_stats': grad_info
        }
        
        print()
    
    # 비교 분석
    print("\n🔬 방법별 비교")
    print("=" * 60)
    print(f"{'Method':<20} {'Output Range':<15} {'Std':<10} {'Unique':<8} {'Grad Norm':<12}")
    print("-" * 65)
    
    for method_name, data in results.items():
        out_stats = data['output_stats']
        grad_stats = data['grad_stats']
        
        range_str = f"[{out_stats['min']:.3f}, {out_stats['max']:.3f}]"
        std_str = f"{out_stats['std']:.4f}"
        unique_str = f"{out_stats['unique_values']}"
        grad_str = f"{grad_stats['avg_grad_norm']:.2e}"
        
        print(f"{method_name:<20} {range_str:<15} {std_str:<10} {unique_str:<8} {grad_str:<12}")
    
    # 문제 진단
    print("\n🚨 문제 진단")
    print("=" * 60)
    
    for method_name, data in results.items():
        out_stats = data['output_stats']
        grad_stats = data['grad_stats']
        
        issues = []
        
        # Saturation 체크
        if out_stats['unique_values'] < 100:
            issues.append("출력 saturation (고유값 부족)")
        
        # 그래디언트 소멸 체크
        if grad_stats['avg_grad_norm'] < 1e-5:
            issues.append("그래디언트 소멸 (학습 불가)")
        
        # 범위 체크
        output_range = out_stats['max'] - out_stats['min']
        if output_range < 0.01:
            issues.append("출력 범위 매우 좁음")
        
        if issues:
            print(f"{method_name}: {', '.join(issues)}")
        else:
            print(f"{method_name}: ✅ 정상")

if __name__ == "__main__":
    main()