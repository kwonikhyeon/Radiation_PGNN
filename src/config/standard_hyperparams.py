#!/usr/bin/env python3
"""
표준화된 하이퍼파라미터 설정
모든 실험에서 일관된 하이퍼파라미터 사용을 위한 표준 설정
"""

# ==============================================================================
# STANDARD HYPERPARAMETERS FOR RADIATION PGNN
# ==============================================================================

class StandardHyperParams:
    """표준화된 하이퍼파라미터 클래스"""
    
    # ---------------------------------------------------------------------------
    # TRAINING PARAMETERS
    # ---------------------------------------------------------------------------
    EPOCHS = 80                    # 표준 훈련 에포크 수
    BATCH_SIZE = 16               # 표준 배치 크기 (GPU 메모리 고려)
    LEARNING_RATE = 1e-4          # 표준 학습률 (AdamW 최적화)
    WEIGHT_DECAY = 1e-5           # L2 정규화
    
    # ---------------------------------------------------------------------------
    # MODEL ARCHITECTURE PARAMETERS  
    # ---------------------------------------------------------------------------
    PRED_SCALE = 3.0              # 예측 스케일링 (선명도 vs 안정성 균형)
    DECODER_CHANNELS = [512, 256, 128, 64]  # 디코더 채널 수
    IN_CHANNELS = 6               # 입력 채널 수 (fixed)
    
    # ---------------------------------------------------------------------------
    # LOSS FUNCTION WEIGHTS (표준화됨)
    # ---------------------------------------------------------------------------
    # Physics-Guided Loss
    PG_WEIGHT = 0.10              # 라플라시안 평활성 손실 가중치
    PG_WARMUP = 15                # PG 손실 워밍업 에포크
    
    # Physics Loss (통합 버전)
    PHYSICS_WEIGHT = 0.08         # 물리 손실 가중치 (표준화됨)
    PHYSICS_START_EPOCH = 20      # 물리 손실 시작 에포크
    PHYSICS_WARMUP = 15           # 물리 손실 워밍업 기간
    
    # Reconstruction Loss
    ALPHA_UNMASK = 0.8            # 전체 재구성 손실 가중치
    CENTER_LAMBDA = 1.0           # 중심부(고강도) 영역 가중치
    
    # Physics Parameters (표준화됨)
    SOURCE_THRESHOLD = 0.15       # 소스 감지 임계값
    BACKGROUND_LEVEL = 0.005      # 배경 방사선 수준
    AIR_ATTENUATION = 0.03        # 공기 중 감쇠 계수
    
    # Anti-blur Parameters
    ANTI_BLUR_WEIGHT_MAX = 0.3    # 안개 방지 손실 최대 가중치
    ANTI_BLUR_WARMUP = 40         # 안개 방지 손실 워밍업
    
    # ---------------------------------------------------------------------------
    # SUPPRESSION PARAMETERS (표준화됨)
    # ---------------------------------------------------------------------------
    DISTANCE_DECAY = -1.5         # 거리 기반 억제 감쇠율
    CONFIDENCE_MIN = 0.4          # 신뢰도 최소값
    CONFIDENCE_RANGE = 0.5        # 신뢰도 범위 [0.4, 0.9]
    NEAR_THRESHOLD = 0.08         # 근접 측정점 임계값
    TRANSITION_WIDTH = 0.12       # 전환 구간 폭
    PEAK_THRESHOLD = 0.25         # 피크 확률 임계값
    CONFIDENCE_THRESHOLD = 0.7    # 고신뢰도 임계값
    PRESERVATION_FACTOR = 0.95    # 특징 보존 인수
    
    # ---------------------------------------------------------------------------
    # SCHEDULER PARAMETERS
    # ---------------------------------------------------------------------------
    SCHEDULER_PCT_START = 0.1     # OneCycleLR 워밍업 비율
    SCHEDULER_ANNEAL = 'cos'      # 어닐링 전략
    
    # ---------------------------------------------------------------------------
    # OPTIMIZATION PARAMETERS
    # ---------------------------------------------------------------------------
    GRAD_CLIP_NORM = 2.0          # 그래디언트 클리핑 노름
    
    # ---------------------------------------------------------------------------
    # DATA PARAMETERS
    # ---------------------------------------------------------------------------
    NUM_WORKERS = 4               # 데이터로더 워커 수
    
    @classmethod
    def get_standard_config(cls):
        """표준 설정 딕셔너리 반환"""
        return {
            # Training
            'epochs': cls.EPOCHS,
            'batch_size': cls.BATCH_SIZE,
            'lr': cls.LEARNING_RATE,
            'weight_decay': cls.WEIGHT_DECAY,
            
            # Model
            'pred_scale': cls.PRED_SCALE,
            'in_channels': cls.IN_CHANNELS,
            
            # Loss weights
            'pg_weight': cls.PG_WEIGHT,
            'pg_warmup': cls.PG_WARMUP,
            'physics_weight': cls.PHYSICS_WEIGHT,
            'physics_start_epoch': cls.PHYSICS_START_EPOCH,
            'physics_warmup': cls.PHYSICS_WARMUP,
            'alpha_unmask': cls.ALPHA_UNMASK,
            'center_lambda': cls.CENTER_LAMBDA,
            
            # Physics parameters
            'source_threshold': cls.SOURCE_THRESHOLD,
            'background_level': cls.BACKGROUND_LEVEL,
            'air_attenuation': cls.AIR_ATTENUATION,
            
            # Anti-blur
            'anti_blur_weight_max': cls.ANTI_BLUR_WEIGHT_MAX,
            'anti_blur_warmup': cls.ANTI_BLUR_WARMUP,
            
            # Optimization
            'grad_clip_norm': cls.GRAD_CLIP_NORM,
            'num_workers': cls.NUM_WORKERS,
            
            # Scheduler
            'scheduler_pct_start': cls.SCHEDULER_PCT_START,
            'scheduler_anneal': cls.SCHEDULER_ANNEAL
        }
    
    @classmethod
    def print_config(cls):
        """표준 설정 출력"""
        config = cls.get_standard_config()
        print("🎯 STANDARDIZED HYPERPARAMETERS")
        print("=" * 50)
        for category in ['Training', 'Model', 'Loss weights', 'Physics parameters', 
                        'Anti-blur', 'Optimization']:
            print(f"\n📋 {category}:")
            for key, value in config.items():
                if (category == 'Training' and key in ['epochs', 'batch_size', 'lr', 'weight_decay'] or
                    category == 'Model' and key in ['pred_scale', 'in_channels'] or
                    category == 'Loss weights' and 'weight' in key or 'lambda' in key or 'alpha' in key or
                    category == 'Physics parameters' and any(x in key for x in ['source', 'background', 'air']) or
                    category == 'Anti-blur' and 'anti_blur' in key or
                    category == 'Optimization' and key in ['grad_clip_norm', 'num_workers']):
                    print(f"  {key:20s}: {value}")


# ==============================================================================
# EXPERIMENT VARIANTS
# ==============================================================================

class ExperimentVariants:
    """실험별 변형 설정"""
    
    @staticmethod
    def baseline():
        """기본 실험 설정"""
        config = StandardHyperParams.get_standard_config()
        config['experiment_name'] = 'baseline'
        return config
    
    @staticmethod
    def anti_blur_focused():
        """안개 방지 중심 실험"""
        config = StandardHyperParams.get_standard_config()
        config.update({
            'experiment_name': 'anti_blur_focused',
            'anti_blur_weight_max': 0.4,  # 더 강한 안개 방지
            'anti_blur_warmup': 30,       # 더 빠른 워밍업
            'pred_scale': 3.5,            # 더 높은 스케일
        })
        return config
    
    @staticmethod
    def physics_enhanced():
        """물리 손실 강화 실험"""
        config = StandardHyperParams.get_standard_config()
        config.update({
            'experiment_name': 'physics_enhanced',
            'physics_weight': 0.12,       # 더 강한 물리 손실
            'physics_start_epoch': 15,    # 더 빠른 시작
            'source_threshold': 0.12,     # 더 민감한 소스 감지
        })
        return config
    
    @staticmethod
    def lightweight():
        """경량화 실험"""
        config = StandardHyperParams.get_standard_config()
        config.update({
            'experiment_name': 'lightweight',
            'batch_size': 20,             # 더 큰 배치
            'lr': 1.5e-4,                # 더 높은 학습률
            'epochs': 60,                 # 더 적은 에포크
        })
        return config


# ==============================================================================
# USAGE EXAMPLE
# ==============================================================================

if __name__ == "__main__":
    # 표준 설정 출력
    StandardHyperParams.print_config()
    
    print("\n\n🔬 EXPERIMENT VARIANTS:")
    print("=" * 50)
    
    variants = ['baseline', 'anti_blur_focused', 'physics_enhanced', 'lightweight']
    for variant in variants:
        config = getattr(ExperimentVariants, variant)()
        print(f"\n📊 {variant}:")
        key_changes = {k: v for k, v in config.items() 
                      if k not in StandardHyperParams.get_standard_config() or 
                         StandardHyperParams.get_standard_config()[k] != v}
        for key, value in key_changes.items():
            print(f"  {key:20s}: {value}")