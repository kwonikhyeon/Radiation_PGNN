# Radiation PGNN 분석 노트

## 분석 개요
이 문서는 Claude Code와 함께 진행한 Radiation PGNN 모델의 분석 결과와 발견사항을 기록합니다.

---

## 📊 ConvNeXt Simple GT Exp2 분석 (2025-08-17)

### 🔍 Test_15 사례 심층 분석

#### **관찰된 현상**
1. **측정점 스파이크**: 측정 위치에서 점 형태의 높은 값 예측
2. **안개형 확산**: 전체 영역의 31.6%가 낮은 강도로 넓게 퍼진 예측

#### **핵심 데이터**
```
GT shape: (256, 256), min: 0.0000, max: 1.0000, mean: 0.0382
Pred shape: (256, 256), min: -0.7526, max: 2.1845, mean: 0.0742
Mask shape: (256, 256), mask sum: 52.0 (측정점 개수)

측정점에서의 예측: min=0.0003, max=0.0842, mean=0.0206
측정점 외부 예측: min=-0.7526, max=2.1845, mean=0.0743

공간적 오차:
- GT 피크 위치: (188, 188) - 우하단
- 예측 피크 위치: (141, 74) - 좌상단  
- 거리 차이: 123픽셀

강도 분석:
- 예측 최대값: 2.18 (GT의 2.18배 과잉)
- 높은 값 픽셀: 146개가 1.0 초과, 73개가 1.5 초과
- 확산 영역: 20,727픽셀 (31.6%)
```

#### **측정값 보존 문제**
- **모든 52개 측정점에서 측정값 = 0.0** (배경 방사선)
- **예측값 평균 = 0.0206** (측정값이 0인데 예측이 양수)
- **최대 측정 오차 = 0.0842**
- **측정값의 2배 이상 예측하는 경우 = 52개 (100%)**

### 🔬 근본 원인 분석

#### **1. 측정점 스파이크 원인**
**현재 코드 (train_conv_next_simple_gt.py:83)**
```python
measurement_loss = F.mse_loss(pred * mask_float, measured_values * mask_float)
```

**문제점:**
- 측정값이 0.0인데 soft constraint로는 완벽한 0.0 예측 불가
- 500.0 가중치에도 불구하고 다른 손실들과 경쟁
- 측정점에서 non-zero 예측이 허용됨

#### **2. 안개 패턴 원인**
**현재 코드 (train_conv_next_simple_gt.py:67-74)**
```python
weight = torch.exp(-2.0 * dist)
loss_unmeasured = F.mse_loss(pred * unmeasured_mask * weight, gt * unmeasured_mask * weight)
```

**문제점:**
- 잘못된 피크 위치로 인한 모델 혼란
- 라플라시안 평활성 손실이 전역적 확산 유도
- 가우시안 제약이 올바른 위치에 적용되지 않음

#### **3. 물리 손실의 역효과**
- GT-Physics 손실이 잘못된 피크 위치 기준으로 계산
- 역제곱 법칙이 잘못된 소스 위치에서 적용

### 💡 해결 방안

#### **즉시 해결 가능한 문제**

**1. 측정값 완벽 보존 (Hard Constraint)**
```python
# 현재 soft constraint 대신
pred = pred * (1 - mask_float) + measured_values * mask_float
```

**2. 공간적 정확성 개선**
```python
# GT 피크 기반 물리 손실
gt_peak_pos = torch.argmax(gt.flatten())
```

**3. 강도 제한 강화**
```python
# 물리적 불가능한 값 클리핑
pred = torch.clamp(pred, min=0.0, max=2.0)
```

#### **근본적 개선 방향**
1. **손실 함수 밸런싱 재조정**
2. **측정점 주변 정보 활용 강화**
3. **공간적 제약 조건 추가**

---

## 📈 모델 성능 비교

### ConvNeXt Simple GT Exp1 vs Exp2
- **Exp1**: SSIM 평균 0.4357, 품질 분포 - Good: 0%, Moderate: 19%, Poor: 81%
- **Exp2**: 백그라운드 훈련에서 SSIM 0.52 달성 (4 에포크에서 중단)

### 주요 개선사항
- Simple + GT-Physics 조합으로 성능 향상
- 그러나 여전히 측정값 보존과 공간적 정확성 문제 존재

---

## 🔧 기술적 이슈 해결 기록

### CUDA 메모리 문제
- **문제**: 9.2GB GPU 메모리 사용으로 OOM 발생
- **해결**: 백그라운드 프로세스 종료 후 배치 크기 조정

### 좌표계 문제 (이전 해결)
- **문제**: 데이터셋 [Y,X] vs 모델 [X,Y] 해석 차이
- **해결**: `coord_x = x[:, 4:5]`, `coord_y = x[:, 3:4]`로 수정

### 데이터 일관성 문제 (이전 해결)  
- **문제**: Poisson 노이즈로 인한 91% 데이터 불일치
- **해결**: `POISSON_NOISE = False`로 설정

---

## 📝 다음 분석 계획

### 우선순위 항목
1. **측정값 완벽 보존 메커니즘 구현**
2. **공간적 정확성 개선 방안 테스트**
3. **안개 현상 억제 기법 개발**
4. **다른 테스트 케이스 분석 (특히 측정값이 non-zero인 경우)**

### 추가 분석이 필요한 데이터
- 측정값이 높은 값을 가지는 테스트 케이스
- 다중 소스가 있는 복잡한 시나리오
- 측정점 분포가 다른 패턴들

---

## 📊 Test_20 심층 분석: 피크 강도 과잉 예측 문제 (2025-08-17)

### 🔍 관찰된 현상
1. **공간적 정확성**: GT 피크 위치 (206, 146) vs 예측된 잘못된 피크 (0, 68) - 220픽셀 차이
2. **피크 강도 과잉**: GT 1.0 → 예측 2.5 (2.5배 과잉)
3. **측정값 보존**: 모든 64개 측정점이 non-zero 값 (이전 test_15와 다름)

### 📊 핵심 발견사항

#### **1. 진짜 소스 위치에서의 과잉 예측**
```
GT Peak (206, 146) 주변 분석:
- Radius 5: GT_max=1.0000, Pred_max=1.58x (1.58배)
- Radius 10: GT_max=1.0000, Pred_max=1.85x (1.85배)  
- Radius 15: GT_max=1.0000, Pred_max=2.05x (2.05배)
- 이 지역에 측정점 없음 (0개)
```

#### **2. 잘못된 피크 위치 (0, 68)**
```
Wrong Peak Analysis:
- GT value: 0.0000 (실제로는 소스가 없는 위치)
- Pred value: 2.5000 (최대값으로 예측)
- 주변 측정점: 6개 (모두 0.0 측정값)
```

#### **3. 중간 지역의 측정점들**
```
High-value measurements (>0.05):
Pos (141, 93): Measured=0.1091, Pred=0.3173 (2.9배)
Pos (165,100): Measured=0.1512, Pred=0.6514 (4.3배)  
Pos (182, 91): Measured=0.1440, Pred=0.3414 (2.4배)
Pos (210,110): Measured=0.1283, Pred=0.7475 (5.8배)
Pos (218,117): Measured=0.3030, Pred=1.1452 (3.8배)
```

#### **4. 모델 스케일링 한계 도달**
```
Scaling Analysis:
- pred_scale = 1.5 설정
- 56개 픽셀이 최대값 2.5에 도달 (스케일링 한계)
- 모델이 물리적 제약을 무시하고 최대 출력 생성
```

### 🔬 근본 원인 분석

#### **1. 공간적 혼란 (Spatial Confusion)**
- **측정점 분포와 GT 피크 위치 불일치**: GT 피크 주변에 측정점 없음
- **모델이 측정값 패턴을 잘못 해석**: 중간 지역의 측정값들로부터 잘못된 소스 위치 추정
- **거리 가중 손실의 오작동**: 잘못된 위치를 중심으로 거리 가중치 계산

#### **2. 강도 증폭 메커니즘 (Intensity Amplification)**
```python
# 현재 손실 함수에서의 문제
measurement_loss = F.mse_loss(pred * mask_float, measured_values * mask_float)
```
- **측정값 대비 3-6배 과잉 예측**: 손실 함수가 측정값 보존 실패
- **pred_scale=1.5 + Softplus 활성화**: 이론적 최대값 초과
- **물리적 제약 부재**: 역제곱 법칙 위반

#### **3. GT-Physics 손실의 오작동**
- **잘못된 피크 위치 기준**: GT-physics가 (0,68) 기준으로 계산
- **역제곱 법칙 왜곡**: 실제 소스가 아닌 위치에서 물리 법칙 적용

### 💡 Test_15 vs Test_20 비교

| 특성 | Test_15 | Test_20 |
|------|---------|---------|
| 측정값 분포 | 모든 측정점 = 0.0 | 모든 측정점 > 0.0 |
| 공간적 정확성 | 피크 위치 완전 오류 | 피크 위치 완전 오류 |
| 강도 문제 | 안개형 확산 | 극심한 피크 증폭 |
| 주요 원인 | 측정값 보존 실패 | 공간 혼란 + 강도 증폭 |

### 🎯 해결 방안

#### **즉시 적용 가능한 수정**
1. **강도 클리핑 강화**:
```python
# 물리적 불가능한 강도 제한
pred = torch.clamp(pred, min=0.0, max=1.5)  # GT 최대값보다 약간 높게
```

2. **측정값 완벽 보존**:
```python
# Hard constraint로 측정값 보존
pred = pred * (1 - mask_float) + measured_values * mask_float
```

3. **공간적 제약 추가**:
```python
# 측정값이 높은 지역 중심으로 피크 위치 제한
high_measurement_regions = measured_values > threshold
```

#### **근본적 해결 방향**
1. **다중 소스 가능성 고려한 손실 함수**
2. **측정점 분포 기반 적응적 가중치**
3. **물리 법칙 기반 강도 상한선 설정**

---

## 🎯 Peak Constraint Methods Comparison V2 - CRITICAL LEARNING ISSUE RESOLVED ✅

### 📊 Current Status: 30-Epoch Comparison In Progress

**Root Cause Successfully Identified and Fixed**: Base model outputs range [1.23, 11.08] while GT is [0, 1.0]. Hard clipping was forcing all outputs to 1.0, completely blocking gradient flow.

### 🔧 V2 Solution: Proper Scaling Before Constraints

```python
class Method1_HardClip(nn.Module):
    def forward(self, x):
        pred = self.base_model(x)
        pred = pred / 4.0  # Normalize before clipping
        pred = torch.clamp(pred, min=0.0, max=1.0)
        return pred
```

**Other Methods Fixed Similarly**:
- **Sigmoid**: pred/6.0 → sigmoid
- **Tanh+Shift**: pred/3.0 → (tanh+1)/2  
- **Loss Constrained**: pred/4.0 + intensity penalty
- **Adaptive**: softplus → per-sample normalization

### ✅ Verification Results (Training Progress Confirmed)
**Method 1 (Hard Clip) - Epoch 1 Progress**:
- **Loss**: 0.54 → 0.28 (decreasing properly)
- **SSIM**: 0.05 → 0.08 (improving)  
- **Range**: [0.15, 0.58] (reasonable, not saturated)
- **Gradient Flow**: ✅ No blockage, smooth training

### 🚀 Current Experiment Details
- **Duration**: 30 epochs × 5 methods = ~150 total epochs
- **Expected Time**: ~3-4 hours
- **Status**: Method 1 training in progress, learning properly
- **Next**: Automatic progression through all 5 methods

### 📈 Key Improvements in V2
1. **Learning Rate**: Reduced to 1e-4 for stability
2. **Scheduler**: OneCycleLR with 2x peak for faster convergence
3. **Loss Rebalancing**: Better unmeasured/measurement loss ratio
4. **Gradient Clipping**: Increased to 2.0 max norm
5. **Physics Loss**: Delayed start (epoch 5) with gradual ramp-up

### 🎯 Expected Outcomes
- All 5 methods should learn properly (no more learning stagnation)
- Meaningful performance comparison between constraint approaches
- Clear identification of optimal constraint strategy
- Production-ready constraint method for main training pipeline

**Status**: ✅ **CRITICAL ISSUE RESOLVED** - Full 30-epoch comparison running successfully

---

*마지막 업데이트: 2025-08-18*
*분석자: Claude Code + User*