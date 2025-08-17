# CLAUDE.md

이 파일은 Claude Code (claude.ai/code)가 이 저장소에서 작업할 때 참고할 가이드를 제공합니다.

## 프로젝트 개요

본 프로젝트는 원자력 시설 해체를 위한 다중 방사능 소스 탐지 및 위험 인식 경로 계획 시스템입니다. 물리 유도 신경망(Physics-Guided Neural Networks, PGNN)을 활용하여 희소한 방사선 측정값으로부터 전체 방사선 필드를 복원하고, 정보 획득, 위험 회피, 이동성을 고려한 최적 경로 계획을 수행합니다.

## 핵심 아키텍처

### 방사선 필드 예측 모델
프로젝트는 다음과 같은 PGNN 기반 아키텍처들을 구현합니다:

- **PGNN-UNet**: 부분 합성곱과 물리 유도 손실을 사용하는 U-Net (`src/model/unet_pgnn.py`, `src/model/unet_deep_pgnn.py`)
- **PGNN-ConvNeXt**: 마스크 어텐션을 활용한 ConvNeXt 백본 (`src/model/conv_next_pgnn.py`) 
- **PGNN-TransUNet**: 비전 트랜스포머를 결합한 Transformer-UNet 하이브리드 (`src/model/trans_unet_pgnn.py`)

모든 모델은 다음 구성 요소를 사용합니다:
- **부분 합성곱(Partial Convolutions)**: 희소 측정값의 마스크 인식 처리
- **라플라시안 물리 손실**: 예측된 방사선 필드의 평활성 제약 조건 강제
- **거리 가중 손실**: 측정 지점 근처의 복원 품질 강조

### 시스템 구성 요소
1. **방사선 측정 및 이동**: 로봇이 특정 위치에서 방사선 값 측정
2. **방사선 필드 추정**: PGNN 모델을 통한 조밀한 방사선 필드 예측
3. **경로 후보 생성**: RRT 기반 경로 샘플링
4. **가중 맵 계산**: 위험도, 정보 획득, 이동성을 결합한 가중치 맵 생성
5. **최적 경로 선택**: 가중 맵 기반의 n-스텝 경로 선택

## 물리 유도 신경망(PGNN) 핵심 원리

### 역제곱 법칙 모델
```
λᵢₖ = mᵦ + Σₛ₌₁ʳ [fₛ/(dᵢₖₛ)²] · e^(-βᵢdᵢₖₛ) · tᵢₖ
```

- **λᵢₖ**: k시점에서 센서 i의 예상 방사선 측정값
- **mᵦ**: 배경 방사선 (자연적으로 존재하는 방사선)
- **fₛ**: 소스 s의 강도 (강한 소스일수록 높은 측정값)
- **dᵢₖₛ**: 센서 i와 소스 s 간의 거리
- **e^(-βᵢdᵢₖₛ)**: 공기 중 흡수에 의한 감쇠
- **tᵢₖ**: 측정 노출 시간

### 라플라시안 평활성 손실
```
ΔF̂(x,y) = ∂²F̂/∂x² + ∂²F̂/∂y²
```
물리적 일관성을 위한 공간적 평활성을 강제합니다.

## 데이터 파이프라인

### 데이터셋 생성
```bash
# 합성 방사선 필드 데이터셋 생성
python3 -m src.dataset.dataset_generator --n_train 9000 --n_val 1000 --n_test 100
```

데이터 파이프라인 구성:
1. **실제 값 생성** (`src/dataset/generate_truth.py`): 가우시안 소스를 사용한 합성 방사선 필드 생성
2. **궤적 샘플링** (`src/dataset/trajectory_sampler.py`): 센서 경로와 희소 측정값 시뮬레이션
3. **데이터셋 조립** (`src/dataset/dataset_generator.py`): 필드와 측정값을 학습 데이터로 결합

입력 특성 (6채널): 측정값, 측정 마스크, 로그 측정값, 정규화된 X/Y 좌표, 거리 맵

## Training Commands

### U-Net Training
```bash
python3 -m src.train.train_unet \
    --data_dir data \
    --save_dir checkpoints/unet_exp1 \
    --epochs 50 \
    --batch 16 \
    --lr 1e-3 \
    --pg_weight 0.1 \
    --pg_warmup 10
```

### ConvNeXt 학습 (개선된 PGNN)
```bash
# 개선된 물리 손실 및 강도 예측을 위한 설정
python3 -m src.train.train_conv_next \
    --data_dir data \
    --save_dir checkpoints/convnext_pgnn_exp6 \
    --epochs 120 \
    --batch 16 \
    --lr 2e-4 \
    --pg_weight 0.15 \
    --physics_weight 0.1 \
    --physics_start_epoch 15 \
    --source_threshold 0.2 \
    --background_level 0.01 \
    --air_attenuation 0.05
```

#### 개선 사항:
- **더 높은 강도 범위**: pred_scale=3.0, Softplus 활성화
- **향상된 물리 손실 균형**: 라플라시안 0.15, 역제곱법칙 0.1  
- **조기 물리 도입**: 15 에포크부터 물리 손실 적용
- **낮은 소스 임계값**: 0.2로 설정하여 약한 소스도 감지
- **적응적 강도 가중치**: 학습 진행에 따라 강도 예측 강조

### TransUNet Training  
```bash
python3 -m src.train.train_trans_unet \
    --data_dir data \
    --save_dir checkpoints/transunet_exp1 \
    --epochs 50 \
    --batch 16 \
    --lr 1e-3
```

## Evaluation Commands

### U-Net Evaluation
```bash
python3 -m src.eval.eval_unet \
    --ckpt checkpoints/unet_exp1/ckpt_best.pth \
    --data_file data/test.npz \
    --out_dir eval_vis/unet_exp1
```

### ConvNeXt Evaluation
```bash
python3 -m src.eval.eval_convnext \
    --ckpt checkpoints/convnext_pgnn_exp1/ckpt_best.pth \
    --data_file data/test.npz \
    --out_dir eval_vis/convnext_exp1
```

### TransUNet Evaluation
```bash
python3 -m src.eval.eval_trans_unet \
    --ckpt checkpoints/transunet_exp1/ckpt_best.pth \
    --data_file data/test.npz \
    --out_dir eval_vis/transunet_exp1
```

## Key Dependencies

- PyTorch 2.7.1+ with CUDA support
- kornia (for SSIM metrics)
- timm (for ConvNeXt backbone)
- einops (for tensor operations in TransUNet)
- matplotlib (for visualization)
- scipy (for distance transforms)
- numpy, tqdm

## Directory Structure

```
src/
├── dataset/           # Data generation pipeline
├── model/            # Neural network architectures
├── train/            # Training scripts for each model
└── eval/             # Evaluation and visualization scripts

checkpoints/          # Saved model checkpoints organized by experiment
data/                # Generated dataset files (train.npz, val.npz, test.npz)
eval_vis/            # Evaluation visualizations and predictions
```

## Physics-Guided Components

The PGNN approach incorporates physics through:
1. **Laplacian Smoothness Loss**: `laplacian_loss()` in model files enforces spatial smoothness
2. **Partial Convolutions**: Handle sparse/masked inputs properly 
3. **Distance-weighted Training**: Prioritizes accuracy near measurement locations
4. **Progressive Physics Loss**: `pg_warmup` parameter gradually increases physics constraint strength

## Experiment Management

Model checkpoints are automatically saved with configuration files. Each experiment directory contains:
- `ckpt_best.pth`: Best validation performance checkpoint
- `ckpt_last.pth`: Most recent checkpoint  
- `config.json`: Training hyperparameters and settings

## Running Environment

- Python 3.10+ recommended
- CUDA-capable GPU strongly recommended for training
- Use `python3` command (not `python`) as the Python interpreter