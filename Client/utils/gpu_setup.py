"""
GPU 설정 및 메모리 관리
"""
import os
import warnings
import tensorflow as tf
from tensorflow.keras import mixed_precision

# 경고 필터링
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def setup_gpu(gpu_id: int = 0, enable_mixed_precision: bool = True):
    """
    GPU 설정 및 메모리 최적화
    
    Args:
        gpu_id: 사용할 GPU ID (-1이면 CPU)
        enable_mixed_precision: Mixed Precision 활성화 여부
    """
    # GPU 메모리 할당 전략
    os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
    
    gpus = tf.config.list_physical_devices('GPU')
    
    if gpu_id == -1 or not gpus:
        # CPU만 사용
        tf.config.set_visible_devices([], 'GPU')
        print("\033[1;33m[GPU] CPU만 사용합니다.\033[0m")
    else:
        try:
            # 특정 GPU 선택 및 메모리 증가 허용
            tf.config.set_visible_devices([gpus[gpu_id]], 'GPU')
            tf.config.experimental.set_memory_growth(gpus[gpu_id], True)
            
            visible_devices = tf.config.get_visible_devices('GPU')
            print(f"\033[1;32m[GPU] 사용 중: {visible_devices}\033[0m")
            
        except RuntimeError as e:
            print(f"\033[1;31m[GPU] 설정 실패 (이미 초기화됨): {e}\033[0m")
    
    # Mixed Precision (AMP) 설정
    if enable_mixed_precision:
        mixed_precision.set_global_policy('mixed_float16')
        print("\033[1;32m[AMP] Mixed Precision 활성화 (float16)\033[0m")