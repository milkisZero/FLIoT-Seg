import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from PIL import Image
import os

# SYNTHIA 클래스 이름 매핑
SYNTHIA_CLASSES = {
    0: 'void',
    1: 'sky', 
    2: 'Building',
    3: 'Road',
    4: 'Sidewalk',
    5: 'Fence',
    6: 'Vegetation',
    7: 'Pole',
    8: 'Car',
    9: 'Traffic sign',
    10: 'Pedestrian',
    11: 'Bicycle',
    12: 'Motorcycle',
    13: 'Parking-slot',
    14: 'Road-work',
    15: 'Traffic light',
    16: 'Terrain',
    17: 'Rider',
    18: 'Truck',
    19: 'Bus',
    20: 'Train',
    21: 'Wall',
    22: 'Lanemarking'
}

def analyze_synthia_dataset(X_train, y_train, X_test, y_test, num_samples=5):
    """
    SYNTHIA 데이터셋의 채널 정보와 클래스 분포를 분석
    """
    print("="*60)
    print("SYNTHIA 데이터셋 분석")
    print("="*60)
    
    # 0. 데이터 유효성 검사
    print("\n0. 데이터 유효성 검사:")
    if X_train.size == 0:
        print("❌ X_train이 비어있습니다!")
        return None
    if y_train.size == 0:
        print("❌ y_train이 비어있습니다!")
        return None
    if X_test.size == 0:
        print("❌ X_test가 비어있습니다!")
        return None
    if y_test.size == 0:
        print("❌ y_test가 비어있습니다!")
        return None
    print("✅ 모든 데이터가 정상적으로 로드됨")
    
    # 1. 기본 정보
    print("\n1. 데이터셋 기본 정보:")
    print(f"X_train shape: {X_train.shape}, dtype: {X_train.dtype}")
    print(f"y_train shape: {y_train.shape}, dtype: {y_train.dtype}")
    print(f"X_test shape: {X_test.shape}, dtype: {X_test.dtype}")
    print(f"y_test shape: {y_test.shape}, dtype: {y_test.dtype}")
    
    # 2. RGB 이미지 정보
    print("\n2. RGB 이미지 분석:")
    if X_train.size > 0:
        print(f"X_train 값 범위: {X_train.min()} ~ {X_train.max()}")
    if X_test.size > 0:
        print(f"X_test 값 범위: {X_test.min()} ~ {X_test.max()}")
    
    # 3. 라벨 이미지 정보
    print("\n3. 라벨 이미지 분석:")
    print(f"y_train 값 범위: {y_train.min()} ~ {y_train.max()}")
    print(f"y_test 값 범위: {y_test.min()} ~ {y_test.max()}")
    
    # 4. 전체 클래스 분포 (Train)
    print("\n4. Train 데이터 클래스 분포:")
    train_unique, train_counts = np.unique(y_train, return_counts=True)
    total_train_pixels = y_train.size
    
    for cls, count in zip(train_unique, train_counts):
        percentage = count / total_train_pixels * 100
        class_name = SYNTHIA_CLASSES.get(cls, f"Unknown_{cls}")
        print(f"  Class {cls:2d} ({class_name:12s}): {count:8,} pixels ({percentage:5.2f}%)")
    
    # 5. 전체 클래스 분포 (Test)
    print("\n5. Test 데이터 클래스 분포:")
    test_unique, test_counts = np.unique(y_test, return_counts=True)
    total_test_pixels = y_test.size
    
    for cls, count in zip(test_unique, test_counts):
        percentage = count / total_test_pixels * 100
        class_name = SYNTHIA_CLASSES.get(cls, f"Unknown_{cls}")
        print(f"  Class {cls:2d} ({class_name:12s}): {count:8,} pixels ({percentage:5.2f}%)")
    
    # 6. 샘플 이미지별 클래스 분포
    print(f"\n6. 샘플 이미지별 클래스 분포 (첫 {num_samples}장):")
    for i in range(min(num_samples, len(y_train))):
        sample_unique, sample_counts = np.unique(y_train[i], return_counts=True)
        total_pixels = y_train[i].size
        print(f"\n  Train 이미지 {i}:")
        for cls, count in zip(sample_unique, sample_counts):
            percentage = count / total_pixels * 100
            class_name = SYNTHIA_CLASSES.get(cls, f"Unknown_{cls}")
            if percentage > 1.0:  # 1% 이상인 클래스만 표시
                print(f"    Class {cls:2d} ({class_name:12s}): {percentage:5.2f}%")
    
    # 7. 클래스 불균형 분석
    print("\n7. 클래스 불균형 분석:")
    train_percentages = train_counts / total_train_pixels * 100
    major_classes = train_unique[train_percentages > 10.0]
    minor_classes = train_unique[(train_percentages < 1.0) & (train_percentages > 0)]
    
    print(f"  주요 클래스 (10% 이상): {[SYNTHIA_CLASSES.get(cls, cls) for cls in major_classes]}")
    print(f"  소수 클래스 (1% 미만): {[SYNTHIA_CLASSES.get(cls, cls) for cls in minor_classes]}")
    
    # 8. 중요한 객체 클래스 분석
    important_classes = [8, 10, 11, 12, 17]  # Car, Pedestrian, Bicycle, Motorcycle, Rider
    print("\n8. 중요한 객체 클래스 분석:")
    for cls in important_classes:
        if cls in train_unique:
            idx = np.where(train_unique == cls)[0][0]
            percentage = train_counts[idx] / total_train_pixels * 100
            class_name = SYNTHIA_CLASSES.get(cls, f"Unknown_{cls}")
            print(f"  {class_name:12s} (Class {cls}): {percentage:6.3f}%")
        else:
            class_name = SYNTHIA_CLASSES.get(cls, f"Unknown_{cls}")
            print(f"  {class_name:12s} (Class {cls}): 0.000% (없음)")
    
    return {
        'train_distribution': dict(zip(train_unique, train_counts)),
        'test_distribution': dict(zip(test_unique, test_counts)),
        'major_classes': major_classes,
        'minor_classes': minor_classes
    }

def visualize_sample_images(X_train, y_train, num_samples=3):
    """
    샘플 이미지와 라벨을 시각화
    """
    fig, axes = plt.subplots(2, num_samples, figsize=(15, 8))
    
    for i in range(num_samples):
        # RGB 이미지
        axes[0, i].imshow(X_train[i])
        axes[0, i].set_title(f'RGB Image {i}')
        axes[0, i].axis('off')
        
        # 라벨 이미지
        axes[1, i].imshow(y_train[i], cmap='tab20', vmin=0, vmax=22)
        axes[1, i].set_title(f'Label Image {i}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.show()

def check_individual_label_files(label_dir, num_check=3):
    """
    개별 라벨 파일의 채널 정보를 확인
    """
    print("\n9. 개별 라벨 파일 채널 정보:")
    label_files = sorted([f for f in os.listdir(label_dir) if f.endswith('.png')])[:num_check]
    
    for i, filename in enumerate(label_files):
        filepath = os.path.join(label_dir, filename)
        
        # 원본 파일 정보
        with Image.open(filepath) as img:
            print(f"\n  파일 {i+1}: {filename}")
            print(f"    PIL 모드: {img.mode}")
            print(f"    PIL 크기: {img.size}")
            
            # numpy 배열로 확인
            arr = np.array(img)
            print(f"    NumPy shape: {arr.shape}")
            print(f"    NumPy dtype: {arr.dtype}")
            
            if len(arr.shape) == 3:
                print(f"    채널별 값 범위:")
                for ch in range(arr.shape[2]):
                    print(f"      채널 {ch}: {arr[:,:,ch].min()} ~ {arr[:,:,ch].max()}")
                    
                    # 채널별 고유값 확인 (처음 10개만)
                    unique_vals = np.unique(arr[:,:,ch])[:20]
                    print(f"      고유값 (첫 10개): {unique_vals}")
            else:
                unique_vals = np.unique(arr)[:20]
                print(f"    고유값 (첫 10개): {unique_vals}")

# 사용 예시
if __name__ == "__main__":
    # # 데이터 로드 (기존 함수 사용)
    # X_train, y_train, X_test, y_test = load_synthia_dataset()
    
    # # 분석 실행
    # analysis_results = analyze_synthia_dataset(X_train, y_train, X_test, y_test)
    
    # # 샘플 이미지 시각화
    # visualize_sample_images(X_train, y_train, num_samples=3)
    
    # 개별 파일 확인 (옵션)
    check_individual_label_files("SYNTHIA_Splitted/client1/train/LABELS", num_check=100)
    
    print("\n분석 완료!")