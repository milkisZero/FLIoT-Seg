# non_iid_split.py
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

# SYNTHIA TrainIds 0~18 사용 (총 19개)
NUM_CLASSES = 19

SYNTHIA_CLASSES = {
    0:  'Road',          # 0  road
    1:  'Sidewalk',      # 1  sidewalk
    2:  'Building',      # 2  building
    3:  'Wall',          # 3  wall
    4:  'Fence',         # 4  fence
    5:  'Pole',          # 5  pole
    6:  'Traffic light', # 6  traffic light
    7:  'Traffic sign',  # 7  traffic sign
    8:  'Vegetation',    # 8  vegetation
    9:  'Terrain',       # 9  terrain
    10: 'Sky',           # 10 sky
    11: 'Person',        # 11 person
    12: 'Rider',         # 12 rider
    13: 'Car',           # 13 car
    14: 'Truck',         # 14 truck
    15: 'Bus',           # 15 bus
    16: 'Train',         # 16 train
    17: 'Motorcycle',    # 17 motorcycle
    18: 'Bicycle',       # 18 bicycle
}

def compute_label_histogram(label_path, num_classes=NUM_CLASSES):
    """
    개별 라벨 PNG 파일에서 클래스별 픽셀 비율 반환.
    - labelTrainIds 기준, ignore id(255)는 제거
    - 반환: shape (num_classes,), 합 = 1.0
    """
    try:
        with Image.open(label_path) as img:
            arr = np.array(img)
    except Exception as e:
        print(f"Warning: Failed to process {label_path}: {e}")
        return np.zeros(num_classes, dtype=np.float32)

    # 라벨이 HxW 또는 HxWxC인 경우 대응
    if arr.ndim == 3:
        arr = arr[:, :, 0]

    valid_mask = arr != 255
    arr_valid = arr[valid_mask]

    if arr_valid.size == 0:
        return np.zeros(num_classes, dtype=np.float32)

    # 0~18만 사용 (19~는 잘라냄)
    arr_valid = np.clip(arr_valid, 0, num_classes - 1)

    hist = np.bincount(arr_valid, minlength=num_classes).astype(np.float32)
    if hist.sum() == 0:
        return np.zeros(num_classes, dtype=np.float32)
    hist /= hist.sum()
    return hist


def compute_multi_hot_label(
    label_path: str,
    selected_labels: list,
    threshold: float = 0.01,
    num_classes: int = NUM_CLASSES
) -> list:
    """
    이미지에서 threshold 이상 존재하는 모든 selected_labels 클래스 반환
    
    Args:
        label_path: 라벨 이미지 경로
        selected_labels: 사용할 클래스 리스트
        threshold: 클래스가 존재한다고 판단할 최소 픽셀 비율 (기본 1%)
        num_classes: 전체 클래스 수
    
    Returns:
        이미지에 threshold 이상 존재하는 클래스 리스트
    """
    hist = compute_label_histogram(label_path, num_classes=num_classes)
    if hist is None or hist.sum() == 0:
        return []
    
    # threshold 이상인 selected_labels만 추출
    present_classes = []
    for cls in selected_labels:
        if 0 <= cls < num_classes and hist[cls] >= threshold:
            present_classes.append(cls)
    
    return present_classes


def multi_hot_dirichlet_split(
    matched_pairs: list,
    num_clients: int,
    client_ratios: list,
    selected_labels: list,
    alpha: float,
    threshold: float = 0.01,
    exclude_from_scoring: list = None,
    num_classes: int = NUM_CLASSES,
    random_state: int = 42,
) -> list:
    """
    Multi-hot label 기반 Dirichlet Non-IID 분할
    
    절차:
    1) 각 이미지가 어떤 클래스들을 포함하는지 파악 (multi-hot)
    2) 각 클라이언트의 "클래스 선호도" 결정 (Dirichlet 분포)
    3) 이미지를 선호도 기반으로 확률적 할당 (배경 클래스 제외)
    
    Args:
        matched_pairs: [(rgb_path, label_path), ...] 리스트
        num_clients: 클라이언트 수
        client_ratios: 각 클라이언트 비율 (합=1)
        selected_labels: 사용할 클래스 리스트
        alpha: Dirichlet 분포 concentration parameter
               (작을수록 극단 Non-IID, 클수록 IID에 가까움)
        threshold: multi-hot 판단 기준 픽셀 비율
        exclude_from_scoring: 점수 계산에서 제외할 클래스 (배경 클래스)
                             None이면 기본값 사용 [0,1,2,3,4,5,8,9,10]
        num_classes: 전체 클래스 수
        random_state: 랜덤 시드
    
    Returns:
        client_file_pairs: 클라이언트별 [(rgb_path, label_path), ...] 리스트
    """
    rng = np.random.RandomState(random_state)
    total_files = len(matched_pairs)
    
    # 배경 클래스 기본값 설정
    if exclude_from_scoring is None:
        # Road, Sidewalk, Building, Wall, Fence, Pole, Vegetation, Terrain, Sky
        exclude_from_scoring = [0, 1, 2, 3, 4, 5, 8, 9, 10]
    
    exclude_set = set(exclude_from_scoring)
    
    # 타겟 사이즈 계산
    target_sizes = [int(round(r * total_files)) for r in client_ratios]
    diff = total_files - sum(target_sizes)
    if diff != 0:
        target_sizes[-1] += diff
    
    print(f"[Multi-Hot] Target sizes: {target_sizes}")
    print(f"[Multi-Hot] Alpha: {alpha}, Threshold: {threshold}")
    print(f"[Multi-Hot] Excluded from scoring: {[SYNTHIA_CLASSES.get(c, str(c)) for c in exclude_from_scoring]}")
    
    # 1) 각 이미지의 multi-hot label 계산
    print("[Multi-Hot] Computing multi-hot labels...")
    image_labels = []  # [(img_idx, [class1, class2, ...]), ...]
    for idx, (_, label_path) in enumerate(tqdm(matched_pairs, desc="Multi-hot labels")):
        classes = compute_multi_hot_label(
            label_path, selected_labels, threshold, num_classes
        )
        image_labels.append((idx, classes))
    
    # 통계 출력
    class_occurrence = {cls: 0 for cls in selected_labels}
    for _, classes in image_labels:
        for cls in classes:
            if cls in class_occurrence:
                class_occurrence[cls] += 1
    
    print("[Multi-Hot] Class occurrence (number of images):")
    for cls in selected_labels:
        print(f"  class {cls} ({SYNTHIA_CLASSES.get(cls, 'Unknown')}): {class_occurrence[cls]} images")
    
    # 2) 각 클라이언트의 클래스 선호도 (Dirichlet 분포)
    client_class_preference = np.zeros((num_clients, num_classes), dtype=np.float32)
    
    for c_idx in range(num_clients):
        # Dirichlet 샘플링
        preference = rng.dirichlet(alpha * np.ones(len(selected_labels)))
        for i, cls in enumerate(selected_labels):
            client_class_preference[c_idx, cls] = preference[i]
    
    print("\n[Multi-Hot] Client class preferences (top 3):")
    for c_idx in range(num_clients):
        top_classes = []
        for cls in selected_labels:
            top_classes.append((cls, client_class_preference[c_idx, cls]))
        top_classes.sort(key=lambda x: x[1], reverse=True)
        
        top_3_str = ", ".join([
            f"{SYNTHIA_CLASSES.get(cls, str(cls))}({pref:.3f})" 
            for cls, pref in top_classes[:3]
        ])
        print(f"  Client {c_idx+1}: {top_3_str}")
    
    # 3) 이미지 할당
    client_file_indices = [[] for _ in range(num_clients)]
    current_sizes = [0] * num_clients
    
    # 이미지 순서 섞기
    rng.shuffle(image_labels)
    
    # 통계: 배경만 있는 이미지 개수
    background_only_count = 0
    
    for img_idx, classes in tqdm(image_labels, desc="Assigning images"):
        if not classes:
            # 클래스 없는 이미지는 부족한 클라이언트에
            deficits = [target_sizes[c] - current_sizes[c] for c in range(num_clients)]
            c_idx = int(np.argmax(deficits))
            background_only_count += 1
        else:
            # 🔥 배경 클래스를 제외한 전경 클래스만 점수 계산에 사용
            scoring_classes = [cls for cls in classes if cls not in exclude_set]
            
            if not scoring_classes:
                # 배경 클래스만 있는 이미지 → 균등 분배
                deficits = [target_sizes[c] - current_sizes[c] for c in range(num_clients)]
                c_idx = int(np.argmax(deficits))
                background_only_count += 1
            else:
                # 전경 클래스로만 선호도 점수 계산
                scores = np.zeros(num_clients, dtype=np.float32)
                for cls in scoring_classes:  # 🔥 배경 제외된 클래스만 사용
                    if 0 <= cls < num_classes:
                        scores += client_class_preference[:, cls]
                
                # 남은 용량 고려
                remaining = np.array([
                    max(0, target_sizes[c] - current_sizes[c]) 
                    for c in range(num_clients)
                ], dtype=np.float32)
                
                if remaining.sum() == 0:
                    scores = np.ones(num_clients, dtype=np.float32)
                else:
                    scores = scores * remaining
                
                # 확률적 선택
                if scores.sum() == 0:
                    probs = remaining / remaining.sum() if remaining.sum() > 0 else np.ones(num_clients) / num_clients
                else:
                    probs = scores / scores.sum()
                
                c_idx = rng.choice(num_clients, p=probs)
        
        client_file_indices[c_idx].append(img_idx)
        current_sizes[c_idx] += 1
    
    print(f"\n[Multi-Hot] Final sizes: {current_sizes}")
    print(f"[Multi-Hot] Images with only background classes: {background_only_count} ({background_only_count/total_files*100:.1f}%)")
    
    # 4) 각 클라이언트가 받은 클래스 분포 확인
    print("\n[Multi-Hot] Client class distribution:")
    for c_idx in range(num_clients):
        client_class_count = {cls: 0 for cls in selected_labels}
        for img_idx in client_file_indices[c_idx]:
            _, classes = image_labels[img_idx]
            for cls in classes:
                if cls in client_class_count:
                    client_class_count[cls] += 1
        
        print(f"  Client {c_idx+1}:")
        for cls in selected_labels:
            if client_class_count[cls] > 0:
                ratio = client_class_count[cls] / len(client_file_indices[c_idx]) * 100
                print(f"    {SYNTHIA_CLASSES.get(cls, str(cls))}: {client_class_count[cls]} images ({ratio:.1f}%)")
    
    # 5) 인덱스 → 파일 쌍 변환
    client_file_pairs = [[] for _ in range(num_clients)]
    for c_idx in range(num_clients):
        for idx in client_file_indices[c_idx]:
            client_file_pairs[c_idx].append(matched_pairs[idx])
    
    return client_file_pairs


def compute_class_distribution_in_dir(label_dir, num_classes=NUM_CLASSES, max_files=None):
    """
    label_dir 아래의 LABELS PNG들을 모두(or 일부) 읽어서
    클래스별 픽셀 카운트 배열을 반환.
    """
    if not os.path.isdir(label_dir):
        print(f"[경고] 라벨 디렉토리 없음: {label_dir}")
        return np.zeros(num_classes, dtype=np.int64)

    label_files = sorted([f for f in os.listdir(label_dir) if f.endswith(".png")])
    if max_files is not None:
        label_files = label_files[:max_files]

    total_hist = np.zeros(num_classes, dtype=np.int64)

    for filename in tqdm(label_files, desc=f"라벨 스캔 ({os.path.basename(label_dir)})"):
        path = os.path.join(label_dir, filename)
        with Image.open(path) as img:
            arr = np.array(img)

        if arr.ndim == 3:
            arr = arr[:, :, 0]

        # 255, 0,1,2,3 모두 제거
        valid_mask = (arr != 255)
        for bg in [0, 1, 2, 3, 4, 5, 8, 9, 10]:
            valid_mask &= (arr != bg)

        arr_valid = arr[valid_mask]
        if arr_valid.size == 0:
            continue

        range_mask = (arr_valid >= 0) & (arr_valid < num_classes)
        arr_valid = arr_valid[range_mask]
        if arr_valid.size == 0:
            continue
        
        hist = np.bincount(arr_valid, minlength=num_classes)
        total_hist += hist

    return total_hist  # shape: (num_classes,)


def plot_class_distribution_per_client(base_output_folder, num_clients, split="train", num_classes=NUM_CLASSES):
    """
    특정 클래스(예: 0,1,2,3) 제외하고 시각화
    """
    exclude_classes = [0, 1, 2, 3, 4, 5, 8, 9, 10]
    include_classes = [i for i in range(num_classes) if i not in exclude_classes]

    all_dists = []
    client_names = []

    for c in range(1, num_clients + 1):
        label_dir = os.path.join(base_output_folder, f"client{c}", split, "LABELS")
        dist = compute_class_distribution_in_dir(label_dir, num_classes=num_classes)

        # 제외할 클래스 제거
        filtered_dist = np.array([dist[i] for i in include_classes])

        all_dists.append(filtered_dist)
        client_names.append(f"Client{c}")

    if not all_dists:
        print("[오류] 클래스 분포를 계산할 수 없습니다.")
        return

    all_dists = np.stack(all_dists, axis=0)
    sums = all_dists.sum(axis=1, keepdims=True)
    ratios = all_dists / np.maximum(sums, 1)

    x = np.arange(len(include_classes))
    num_rows = len(client_names)

    fig, axes = plt.subplots(num_rows, 1, figsize=(14, 3 * num_rows), sharex=True)
    if num_rows == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.bar(x, ratios[i])
        ax.set_ylabel(client_names[i])
        ax.set_ylim(0, ratios.max() * 1.1)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(
        [SYNTHIA_CLASSES.get(i, str(i)) for i in include_classes],
        rotation=45,
        ha="right"
    )

    plt.tight_layout()
    plot_file = os.path.join(base_output_folder, f"class_distribution_{split}.png")
    plt.savefig(plot_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"{split} 클래스 분포 플롯 저장: {plot_file}")