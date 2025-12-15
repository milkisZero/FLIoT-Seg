import os
import warnings
import tensorflow as tf
from tensorflow.keras import mixed_precision

# 경고 필터링
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def setup_gpu(
    gpu_id: int = 0,
    enable_mixed_precision: bool = True,
    memory_limit_mb: int = 2000,   # ★ 프로세스당 VRAM 상한 (MB 단위, 기본 12GB)
):
    """
    GPU 설정 및 메모리 최적화
    
    Args:
        gpu_id: 사용할 GPU ID (-1이면 CPU)
        enable_mixed_precision: Mixed Precision 활성화 여부
        memory_limit_mb: 이 프로세스가 최대 사용할 GPU 메모리(MB)
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
            # 물리 GPU 선택
            physical_gpu = gpus[gpu_id]

            # ★ 논리 디바이스로 메모리 상한 설정 (memory_growth 대신 이거 사용)
            tf.config.set_logical_device_configuration(
                physical_gpu,
                [tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit_mb)]
            )
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(f"\033[1;32m[GPU] 사용 중(논리): {logical_gpus}, limit={memory_limit_mb}MB\033[0m")

        except RuntimeError as e:
            print(f"\033[1;31m[GPU] 설정 실패 (이미 초기화됨): {e}\033[0m")
    
    # Mixed Precision (AMP) 설정
    if enable_mixed_precision:
        mixed_precision.set_global_policy('mixed_float16')
        print("\033[1;32m[AMP] Mixed Precision 활성화 (float16)\033[0m")
