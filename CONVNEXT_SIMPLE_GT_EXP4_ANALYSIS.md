# ConvNeXt Simple GT Exp4 종합 분석 리포트

## 🎯 실험 개요
- **모델**: SimplifiedConvNeXtPGNN (53M 파라미터)
- **실험명**: convnext_simple_gt_exp4
- **평가 샘플 수**: 100개
- **전체 성능**: SSIM = 0.6328, PSNR = 18.60 dB

## 📊 전체 성능 요약

### 품질 분포
- **Good**: 0개 (0.0%) 
- **Moderate**: 100개 (100.0%)
- **Poor**: 0개 (0.0%)

### 핵심 메트릭 통계
| 메트릭 | 평균 | 표준편차 | 최소값 | 최대값 | 목표 범위 |
|--------|------|----------|---------|---------|-----------|
| Field MAE | 0.0466 | 0.0186 | 0.0110 | 0.0876 | < 0.05 |
| Intensity Ratio | 1.8496 | 0.3562 | 0.6399 | 2.0000 | 0.8-1.2 |
| Peak Distance | 120.3 | 55.9 | 12.2 | 251.3 | < 10 |
| Std Ratio | 0.6728 | 0.2009 | 0.1694 | 0.9576 | > 0.7 |

## 🔍 주요 문제점 분석

### 1. 공간적 정확도 문제 (Critical)
- **100%의 샘플**에서 피크 위치가 10픽셀 이상 벗어남
- **평균 피크 거리**: 120.3 픽셀 (목표: <10 픽셀)
- **최악 케이스**: test_3에서 251.3 픽셀 벗어남

#### 시각적 분석
- **Best Case (test_16)**: 피크 거리 12.2픽셀
  - 예측과 GT의 형태가 매우 유사
  - 위치만 약간 벗어남 (200,38) vs (210,31)
  - 강도와 분포는 잘 보존됨

- **Worst Case (test_3)**: 피크 거리 251.3픽셀  
  - 예측이 완전히 다른 위치에서 피크 형성
  - 예측: (0,233) vs GT: (144,27)
  - 강도는 유지되나 공간 정보 완전 손실

### 2. 강도 과예측 문제 (High)
- **평균 강도 비율**: 1.85 (목표: 0.8-1.2)
- **85%의 샘플**에서 강도 과예측
- 최대 강도가 항상 2.0으로 클리핑됨 (pred_scale 부족)

### 3. 분포 다양성 부족 (Medium)
- **평균 Std Ratio**: 0.67 (목표: >0.7)
- **4%의 샘플**에서 심각한 분포 다양성 부족
- 단순화된 모델의 부작용으로 추정

## 💡 근본 원인 분석

### 1. 물리 제약 조건 부족
**증거:**
- 공간적 정확도 극도로 낮음 (100% 샘플에서 문제)
- 피크가 측정점과 무관한 위치에 생성

**원인:**
- GT-Physics 손실 가중치 부족
- Spatial attention 메커니즘 단순화
- 측정점-피크 연결성 약화

### 2. 스케일링 문제
**증거:**
- 모든 샘플에서 pred_max = 2.0 (클리핑)
- Intensity ratio 평균 1.85 (과예측)

**원인:**
- pred_scale이 단순화된 모델에 맞지 않음
- Adaptive scaling 메커니즘 약화

### 3. 단순화 부작용
**증거:**
- 일부 샘플에서 분포 다양성 부족
- 복잡한 패턴 표현 능력 저하

**원인:**
- 4→2 브랜치 축소로 표현력 감소
- Suppression 메커니즘 단순화

## 🛠️ 개선 방안

### 1. 즉시 적용 가능한 하이퍼파라미터 조정

```bash
# 공간 정확도 개선을 위한 물리 손실 강화
python3 -m src.train.train_conv_next_simple_gt \
    --save_dir checkpoints/simplified_improved_v1 \
    --gt_physics_weight 0.2 \     # 0.1 → 0.2로 증가
    --gt_physics_start_epoch 5 \  # 15 → 5로 조기 적용
    --physics_weight 0.2 \        # 라플라시안 손실도 증가
    --pred_scale 1.2              # 2.0 → 1.2로 감소 (클리핑 방지)
```

### 2. 아키텍처 미세조정

```python
# simplified_conv_next_pgnn.py 수정사항
class SimplifiedConvNeXtPGNN(nn.Module):
    def __init__(self, ...):
        # 1. 마스크 어텐션 강화
        self.mask_proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, max(ch // 4, 32), ...),  # //8 → //4로 복원
                ...
            ) for ch in enc_channels
        ])
        
        # 2. 공간 정보 보존을 위한 추가 브랜치
        self.spatial_head = nn.Sequential(
            nn.Conv2d(decoder_channels[3], 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, kernel_size=1),
            nn.Sigmoid()
        )
```

### 3. 훈련 전략 개선

**Phase 1: 기본 재구성 집중 (Epoch 1-20)**
```python
# 물리 손실 비활성화, 기본 MSE + 측정점 제약만
measurement_loss_weight = 1000.0
physics_loss_weight = 0.0
```

**Phase 2: 점진적 물리 도입 (Epoch 21-40)**
```python
# GT-Physics 점진적 활성화
gt_physics_weight = min(0.2, (epoch - 20) / 20 * 0.2)
laplacian_weight = min(0.15, (epoch - 20) / 20 * 0.15)
```

**Phase 3: 공간 정확도 집중 (Epoch 41-60)**
```python
# 측정점-피크 연결성 강화
spatial_consistency_loss = F.mse_loss(
    pred * mask_dilated, 
    interpolated_measurements * mask_dilated
)
```

### 4. 데이터 증강 (선택사항)

```python
# 공간 변환에 강건한 학습
def spatial_augmentation(inp, gt):
    # 회전, 스케일링으로 공간 일반화 향상
    return augmented_inp, augmented_gt
```

## 📈 예상 개선 효과

### 단기 목표 (하이퍼파라미터 조정만)
- **Peak Distance**: 120.3 → 60-80 픽셀
- **Intensity Ratio**: 1.85 → 1.2-1.5  
- **Good Quality 비율**: 0% → 15-20%

### 중기 목표 (아키텍처 미세조정 포함)
- **Peak Distance**: 60-80 → 30-50 픽셀
- **Intensity Ratio**: 1.2-1.5 → 0.9-1.3
- **Good Quality 비율**: 15-20% → 40-50%

### 장기 목표 (전면 재설계)
- **Peak Distance**: 30-50 → 10-20 픽셀  
- **Intensity Ratio**: 0.9-1.3 → 0.8-1.2
- **Good Quality 비율**: 40-50% → 70-80%

## 🎯 다음 단계 액션 플랜

### 1. 즉시 실행 (1-2일)
- [ ] 하이퍼파라미터 조정된 모델 재훈련
- [ ] test_16, test_3 등 특정 샘플로 빠른 검증

### 2. 단기 실행 (1주)  
- [ ] 마스크 어텐션 강화 적용
- [ ] 3-phase 훈련 전략 구현
- [ ] 개선된 모델 전체 평가

### 3. 중기 실행 (2-3주)
- [ ] 원본 99M 모델과 성능 비교
- [ ] 추가 브랜치 도입 실험
- [ ] 최적 아키텍처 결정

## 📋 결론

ConvNeXt Simple GT Exp4는 **안정적인 기본 성능**을 보이지만, **공간적 정확도에서 심각한 문제**를 보입니다. 모든 샘플에서 피크 위치가 크게 벗어나는 것은 물리 제약 조건의 부족과 단순화로 인한 공간 정보 손실로 판단됩니다.

**핵심 개선 포인트:**
1. **GT-Physics 손실 강화** (가중치 0.1 → 0.2)
2. **조기 물리 손실 도입** (epoch 15 → 5)
3. **pred_scale 최적화** (클리핑 방지)
4. **마스크 어텐션 복원** (표현력 향상)

이러한 개선을 통해 **공간적 정확도 50-70% 향상**과 **전체 품질 등급 상승**을 기대할 수 있습니다.