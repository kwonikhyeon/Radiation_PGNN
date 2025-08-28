#!/usr/bin/env python3
"""
단순화된 모델에서 제약 방법들 테스트
기존 complex model에서 saturation 문제가 있었던 방법들을 간단한 모델에서 재검증
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))

from model.simplified_conv_next_pgnn import SimplifiedConvNeXtPGNN

# 제약 방법들 정의
class ConstraintMethod1_HardClip(nn.Module):
    """Hard Clipping - 검증된 방법"""
    def __init__(self):
        super().__init__()
        self.base_model = SimplifiedConvNeXtPGNN(in_channels=6, pred_scale=1.0)
        
    def forward(self, x):
        pred = self.base_model(x)
        pred = pred / 3.0  # 스케일링
        pred = torch.clamp(pred, min=0.0, max=1.0)
        return pred

class ConstraintMethod2_Sigmoid(nn.Module):
    """Sigmoid - 이전에 saturation 문제"""
    def __init__(self):
        super().__init__()
        self.base_model = SimplifiedConvNeXtPGNN(in_channels=6, pred_scale=1.0)
        
    def forward(self, x):
        pred = self.base_model(x)
        pred = pred / 4.0  # 더 강한 스케일링
        pred = torch.sigmoid(pred)
        return pred

class ConstraintMethod3_Tanh(nn.Module):
    """Tanh + Shift"""
    def __init__(self):
        super().__init__()
        self.base_model = SimplifiedConvNeXtPGNN(in_channels=6, pred_scale=1.0)
        
    def forward(self, x):
        pred = self.base_model(x)
        pred = pred / 3.0
        pred = (torch.tanh(pred) + 1.0) / 2.0
        return pred

class ConstraintMethod4_Softplus(nn.Module):
    """Softplus + Normalization (새로운 방법)"""
    def __init__(self):
        super().__init__()
        self.base_model = SimplifiedConvNeXtPGNN(in_channels=6, pred_scale=1.0)
        
    def forward(self, x):
        pred = self.base_model(x)
        pred = F.softplus(pred) / 3.0  # Softplus로 양수 보장 후 스케일링
        pred = torch.clamp(pred, max=1.0)  # 상한만 제한
        return pred

def test_constraint_methods():
    """모든 제약 방법들의 출력 특성 테스트"""
    print("🔬 단순화된 모델에서 제약 방법 테스트")
    print("=" * 60)
    
    methods = [
        ("Hard Clip", ConstraintMethod1_HardClip()),
        ("Sigmoid", ConstraintMethod2_Sigmoid()), 
        ("Tanh+Shift", ConstraintMethod3_Tanh()),
        ("Softplus+Clamp", ConstraintMethod4_Softplus())
    ]
    
    # 샘플 입력 (실제 데이터와 유사하게)
    batch_size = 4
    input_tensor = torch.randn(batch_size, 6, 256, 256)
    
    # 입력을 실제 데이터 분포에 맞게 조정
    input_tensor[:, 0] = torch.clamp(input_tensor[:, 0] * 0.1, 0, 1)  # 측정값
    input_tensor[:, 1] = (input_tensor[:, 1] > 0).float()  # 마스크
    input_tensor[:, 2] = torch.log(input_tensor[:, 0] + 1e-8)  # log 측정값
    
    # 좌표 텐서 올바르게 생성
    y_coords = torch.linspace(-1, 1, 256).view(256, 1).repeat(1, 256).unsqueeze(0).repeat(batch_size, 1, 1)
    x_coords = torch.linspace(-1, 1, 256).view(1, 256).repeat(256, 1).unsqueeze(0).repeat(batch_size, 1, 1)
    
    input_tensor[:, 3] = y_coords  # Y 좌표
    input_tensor[:, 4] = x_coords  # X 좌표
    input_tensor[:, 5] = torch.rand(batch_size, 256, 256) * 0.8 + 0.1  # 거리
    
    results = {}
    
    for method_name, model in methods:
        print(f"\n📊 {method_name} 테스트")
        print("-" * 30)
        
        # 출력 특성 분석
        with torch.no_grad():
            output = model(input_tensor)
            
        # 통계 계산
        stats = {
            'mean': output.mean().item(),
            'std': output.std().item(),
            'min': output.min().item(),
            'max': output.max().item(),
            'unique_count': len(torch.unique(output.flatten())),
            'range_width': (output.max() - output.min()).item()
        }
        
        # 출력
        print(f"평균: {stats['mean']:.4f}")
        print(f"표준편차: {stats['std']:.4f}")
        print(f"범위: [{stats['min']:.4f}, {stats['max']:.4f}]")
        print(f"고유값 개수: {stats['unique_count']:,}")
        print(f"범위 폭: {stats['range_width']:.4f}")
        
        # 문제 진단
        issues = []
        if stats['unique_count'] < 1000:
            issues.append("낮은 다양성")
        if stats['range_width'] < 0.01:
            issues.append("범위 너무 좁음")
        if stats['std'] < 0.001:
            issues.append("낮은 변동성")
        if abs(stats['mean'] - 0.5) < 0.01 and stats['std'] < 0.01:
            issues.append("⚠️ Saturation 의심")
            
        if issues:
            print(f"⚠️ 문제: {', '.join(issues)}")
        else:
            print("✅ 정상 동작")
            
        results[method_name] = stats
    
    # 종합 비교
    print(f"\n📋 종합 비교")
    print("=" * 60)
    print(f"{'Method':<15} {'Mean':<8} {'Std':<8} {'Range':<15} {'Unique':<8} {'Status'}")
    print("-" * 70)
    
    for method_name, stats in results.items():
        range_str = f"[{stats['min']:.3f},{stats['max']:.3f}]"
        unique_str = f"{stats['unique_count']//1000}K" if stats['unique_count'] > 1000 else str(stats['unique_count'])
        
        # 상태 판정
        if (stats['unique_count'] > 1000 and stats['range_width'] > 0.1 and 
            stats['std'] > 0.01 and not (abs(stats['mean'] - 0.5) < 0.01 and stats['std'] < 0.01)):
            status = "✅ Good"
        else:
            status = "⚠️ Issue"
            
        print(f"{method_name:<15} {stats['mean']:<8.3f} {stats['std']:<8.3f} {range_str:<15} {unique_str:<8} {status}")

def test_gradient_flow():
    """그래디언트 흐름 테스트"""
    print(f"\n🔍 그래디언트 흐름 테스트")
    print("=" * 40)
    
    methods = [
        ("Hard Clip", ConstraintMethod1_HardClip()),
        ("Sigmoid", ConstraintMethod2_Sigmoid()),
        ("Softplus+Clamp", ConstraintMethod4_Softplus())
    ]
    
    input_tensor = torch.randn(2, 6, 256, 256)
    target = torch.rand(2, 1, 256, 256)
    
    for method_name, model in methods:
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        
        optimizer.zero_grad()
        output = model(input_tensor)
        loss = F.mse_loss(output, target)
        loss.backward()
        
        # 그래디언트 통계
        grad_norms = []
        for param in model.parameters():
            if param.grad is not None:
                grad_norms.append(param.grad.norm().item())
        
        total_grad = sum(grad_norms)
        avg_grad = total_grad / len(grad_norms) if grad_norms else 0.0
        
        print(f"{method_name:<15} - Loss: {loss.item():.4f}, Avg Grad: {avg_grad:.2e}")
        
        if avg_grad < 1e-8:
            print(f"  ⚠️ 그래디언트 소멸 감지!")
        elif avg_grad > 1e2:
            print(f"  ⚠️ 그래디언트 폭발 감지!")
        else:
            print(f"  ✅ 정상 그래디언트 흐름")

def main():
    print("🚀 단순화된 ConvNeXt PGNN 제약 방법 검증")
    print("=" * 60)
    print("목표: 복잡한 모델에서 saturation 문제를 단순한 모델에서 해결")
    print()
    
    test_constraint_methods()
    test_gradient_flow()
    
    print(f"\n🎯 결론:")
    print("- 단순화된 모델(53M)에서 제약 방법들이 정상 동작하는지 확인")
    print("- Sigmoid saturation 문제가 해결되었는지 검증")
    print("- 가장 안정적인 제약 방법 선택을 위한 기초 데이터 제공")

if __name__ == "__main__":
    main()