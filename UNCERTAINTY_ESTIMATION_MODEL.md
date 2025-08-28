# 불확실성 추정 기능이 포함된 ConvNeXt PGNN 모델

## 📋 모델 개요

본 문서는 방사선 필드 예측과 함께 픽셀별 불확실성 정보를 제공하는 **SimplifiedConvNeXtPGNN** 모델에 대한 상세 설명입니다.

### 🎯 핵심 특징
- **방사선 필드 예측**: 희소한 측정값으로부터 조밀한 방사선 필드 복원
- **불확실성 추정**: Monte Carlo Dropout을 활용한 픽셀별 불확실성 맵 생성
- **물리 유도 학습**: GT-Physics 손실과 라플라시안 평활성 제약
- **효율적 아키텍처**: 99M → 53M 파라미터로 복잡도 절반 감소
- **실시간 추론**: 불확실성 추정 포함하여도 빠른 추론 속도

## 🏗️ 아키텍처 구성

### 1. 전체 구조
```
입력 [B, 6, H, W] 
    ↓
ConvNeXt Encoder (Small)
    ↓
Mask Attention 적용
    ↓
Simplified Decoder + Monte Carlo Dropout
    ↓
2개 브랜치 출력:
├── Main Head → 방사선 필드 예측
└── Confidence Head → 신뢰도 맵
```

### 2. 입력 채널 구성 (6채널)
1. **측정값 (M)**: 센서에서 측정된 방사선 값
2. **측정 마스크 (mask)**: 측정 위치 표시 (1: 측정됨, 0: 측정안됨)
3. **로그 측정값 (logM)**: 로그 스케일 측정값
4. **X 좌표**: 정규화된 X 좌표
5. **Y 좌표**: 정규화된 Y 좌표  
6. **거리 맵**: 가장 가까운 측정점까지의 거리

### 3. 핵심 컴포넌트

#### 3.1 ConvNeXt Encoder
```python
self.encoder = create_model(
    "convnext_small",  # 25M 파라미터
    pretrained=False,
    features_only=True,
    in_chans=6
)
```

#### 3.2 Monte Carlo Dropout 레이어
```python
self.mc_dropout1 = nn.Dropout2d(p=dropout_rate)  # 디코더 레벨 1
self.mc_dropout2 = nn.Dropout2d(p=dropout_rate)  # 디코더 레벨 2  
self.mc_dropout3 = nn.Dropout2d(p=dropout_rate)  # 디코더 레벨 3
```

#### 3.3 출력 헤드
```python
# 주 예측 헤드
self.main_head = nn.Sequential(
    nn.Conv2d(32, 16, kernel_size=3, padding=1),
    nn.ReLU(inplace=True),
    nn.Conv2d(16, 1, kernel_size=1)
)

# 신뢰도 헤드
self.confidence_head = nn.Sequential(
    nn.Conv2d(32, 8, kernel_size=3, padding=1),
    nn.ReLU(inplace=True),
    nn.Conv2d(8, 1, kernel_size=1),
    nn.Sigmoid()  # [0,1] 범위
)
```

## 🎲 불확실성 추정 메커니즘

### 1. Monte Carlo Dropout
- **원리**: 추론 시에도 dropout을 유지하여 모델 불확실성 추정
- **적용 위치**: 디코더의 3개 레벨에 적용
- **장점**: 기존 모델에 최소 변경으로 불확실성 추정 가능

### 2. 불확실성 타입

#### Epistemic Uncertainty (인식론적 불확실성)
- **정의**: 모델이 학습 데이터 부족으로 인해 갖는 불확실성
- **계산**: Monte Carlo 샘플들의 예측값 분산
- **의미**: 더 많은 데이터로 줄일 수 있는 불확실성

#### Aleatoric Uncertainty (우연적 불확실성) 
- **정의**: 데이터 자체의 노이즈나 측정 오차로 인한 불확실성
- **계산**: `1.0 - confidence_head_output`
- **의미**: 데이터를 더 모아도 줄이기 어려운 본질적 불확실성

#### 총 불확실성
```python
total_uncertainty = epistemic_uncertainty + 0.5 * aleatoric_uncertainty
```

### 3. 불확실성 추정 함수
```python
def forward_with_uncertainty(self, x, n_samples=20):
    """
    Args:
        x: 입력 텐서 [B, 6, H, W]
        n_samples: Monte Carlo 샘플 수
        
    Returns:
        prediction: 평균 예측값 [B, 1, H, W]
        uncertainty: 불확실성 맵 [B, 1, H, W] 
        confidence: 신뢰도 맵 [B, 1, H, W]
    """
```

## 🔬 물리 유도 학습

### 1. 핵심 물리 원리

#### 역제곱 법칙
```
λ_ik = m_b + Σ_s [f_s/(d_iks)²] · e^(-β_i·d_iks) · t_ik
```
- **λ_ik**: 센서 i의 k시점 방사선 측정값
- **f_s**: 소스 s의 강도
- **d_iks**: 센서-소스 간 거리
- **β_i**: 공기 중 감쇠 계수

#### 라플라시안 평활성
```
ΔF̂(x,y) = ∂²F̂/∂x² + ∂²F̂/∂y²
```

### 2. 손실 함수 구성

#### 주요 손실 (가중치)
- **Unmeasured MSE (3.0)**: 측정되지 않은 영역 재구성
- **Global MSE (0.5)**: 전체 영역 재구성  
- **Measurement Preservation (10.0)**: 측정값 보존 확인
- **Laplacian Loss (점진적)**: 공간적 평활성
- **GT-Physics Loss (점진적)**: 물리 법칙 준수
- **Uncertainty Loss (0.1)**: 불확실성 품질

#### 점진적 학습 전략
```python
# 라플라시안 손실: 5 에포크부터
if epoch >= 5:
    lambda_pg = min(0.05, (epoch - 5) / 20.0 * 0.05)

# GT-Physics 손실: 15 에포크부터  
if epoch >= 15:
    lambda_gt_physics = min(weight, (epoch - 15) / 20.0 * weight)

# 불확실성 손실: 10 에포크부터
if epoch >= 10:
    lambda_uncertainty = min(0.1, (epoch - 10) / 20.0 * 0.1)
```

## 📊 성능 지표

### 1. 기본 메트릭
- **SSIM**: 구조적 유사성 지수
- **MSE**: 평균 제곱 오차
- **Peak Error**: 최대 오차

### 2. 불확실성 메트릭
- **Calibration Error**: 불확실성 보정 품질
- **Uncertainty-Error Correlation**: 불확실성과 실제 오차 상관관계
- **Coverage**: 불확실성 구간 내 실제값 포함 비율

### 3. 보정 오차 계산
```python
def calibration_error(uncertainties, errors, n_bins=10):
    """
    Expected Calibration Error (ECE) 계산
    불확실성이 실제 오차와 얼마나 잘 일치하는지 측정
    """
```

## 🚀 사용법

### 1. 훈련
```bash
python3 -m src.train.train_conv_next_simple_gt \
    --data_dir data \
    --save_dir checkpoints/uncertainty_exp1 \
    --epochs 80 \
    --batch 16 \
    --lr 2e-4 \
    --use_uncertainty \
    --uncertainty_weight 0.1 \
    --dropout_rate 0.1 \
    --use_gt_physics \
    --gt_physics_weight 0.2
```

### 2. 불확실성 평가
```bash
python3 -m src.eval.eval_uncertainty \
    --ckpt checkpoints/uncertainty_exp1/ckpt_best.pth \
    --data_file data/test.npz \
    --out_dir eval_vis/uncertainty_analysis \
    --n_samples 20 \
    --n_vis_samples 5
```

### 3. 코드에서 불확실성 추정
```python
# 모델 로드
model = SimplifiedConvNeXtPGNN(
    in_channels=6, 
    pred_scale=1.0, 
    dropout_rate=0.1
)

# 불확실성과 함께 예측
prediction, uncertainty, confidence = model.forward_with_uncertainty(
    input_tensor, 
    n_samples=20
)
```

## 🎯 경로 계획 활용

### 1. 위험 회피
- **높은 불확실성 영역**: 예측이 불안정한 위험 지역으로 판단
- **경로 계획**: 불확실성이 낮은 안전한 경로 선택
- **동적 조정**: 실시간 불확실성 정보로 경로 재계획

### 2. 정보 획득 전략
- **탐사 우선순위**: 높은 불확실성 = 정보 가치 높음
- **측정 위치 최적화**: 불확실성 감소 효과가 큰 위치 선정
- **적응적 샘플링**: 현재 불확실성 분포에 기반한 차기 측정 계획

### 3. 의사결정 지원
- **신뢰도 기반 판단**: 낮은 불확실성 영역에서 더 확신 있는 결정
- **위험-이익 균형**: 불확실성을 고려한 최적 경로 선택
- **안전 마진 설정**: 불확실성에 비례한 안전 거리 유지

## 📈 모델 사양

### 파라미터 수
- **총 파라미터**: 52,951,587 (53.0M)
- **학습 가능 파라미터**: 53.0M
- **모델 크기**: 202MB (FP32 기준)

### 메모리 사용량
- **훈련 시**: ~8GB (배치 크기 16 기준)
- **추론 시**: ~2GB (배치 크기 4 기준)
- **불확실성 추정**: 추가 메모리 ~20% (MC 샘플 20개 기준)

### 추론 속도
- **일반 예측**: ~50ms/sample (GPU 기준)
- **불확실성 추정**: ~800ms/sample (MC 샘플 20개 기준)
- **실시간 추론**: MC 샘플 수 조정으로 속도-정확도 균형

## 🔧 설정 옵션

### 모델 설정
```python
SimplifiedConvNeXtPGNN(
    in_channels=6,              # 입력 채널 수
    decoder_channels=[256, 128, 64, 32],  # 디코더 채널
    pred_scale=1.0,             # 예측 스케일 팩터
    dropout_rate=0.1            # Monte Carlo Dropout 비율
)
```

### 훈련 설정
```python
# 불확실성 관련
use_uncertainty=True           # 불확실성 학습 활성화
uncertainty_weight=0.1         # 불확실성 손실 가중치

# GT-Physics 관련  
use_gt_physics=True           # GT-Physics 손실 활성화
gt_physics_weight=0.2         # GT-Physics 손실 가중치
gt_physics_start_epoch=15     # GT-Physics 시작 에포크

# 물리 파라미터
background_level=0.0          # 배경 방사선 레벨
air_attenuation=0.01          # 공기 중 감쇠 계수
```

## 📋 향후 개선 방향

### 1. 불확실성 추정 고도화
- **Deep Ensemble**: 여러 모델 앙상블로 더 정확한 불확실성
- **Evidential Learning**: 확률 분포 직접 학습
- **Bayesian Neural Networks**: 베이지안 방법론 적용

### 2. 물리 모델 확장
- **다중 소스 모델링**: 복수 방사선 소스 동시 처리
- **시간적 역학**: 시간에 따른 방사선 감쇠 모델링
- **3D 공간 확장**: 2D에서 3D 공간으로 확장

### 3. 실시간 최적화
- **모델 경량화**: 더 작은 모델로 동일 성능
- **추론 가속**: GPU 최적화 및 병렬 처리
- **적응적 샘플링**: 동적 MC 샘플 수 조정

## 🎉 결론

본 모델은 방사선 필드 예측과 불확실성 추정을 동시에 수행하는 혁신적인 아키텍처로, 원자력 시설 해체와 같은 고위험 환경에서의 안전한 로봇 경로 계획에 핵심적인 역할을 할 것으로 기대됩니다.

**주요 성과:**
- ✅ 53M 파라미터로 효율적 아키텍처 달성
- ✅ Monte Carlo Dropout으로 실용적 불확실성 추정
- ✅ 물리 법칙과 데이터 기반 학습의 조화
- ✅ 실시간 추론 가능한 성능
- ✅ 경로 계획 직접 활용 가능한 출력

---

*최종 업데이트: 2025년 1월*
*모델 버전: SimplifiedConvNeXtPGNN v2.0 (Uncertainty-enabled)*