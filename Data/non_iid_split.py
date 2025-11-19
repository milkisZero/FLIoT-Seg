# non_iid_split.py
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

# SYNTHIA TrainIds 0~18 사용 (총 19개)
NUM_CLASSES = 19

def compute_label_histogram(label_path, num_classes=NUM_CLASSES):
    """
    개별 라벨 PNG 파일에서 클래스별 픽셀 비율 반환.
    - labelTrainIds 기준, ignore id(255)는 제거
    - 반환: shape (num_classes,), 합 = 1.0
    """
    with Image.open(label_path) as img:
        arr = np.array(img)

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

def get_default_client_class_prefs(num_clients, num_classes=NUM_CLASSES):
    """
    더 극단적인 Non-IID를 위한 클래스 선호 설정
    """
    if num_clients == 2:
        # Client 0: 차량 중심 (Car, Truck, Bicycle, Motorcycle, Traffic sign, Traffic light)
        # Client 1: 보행자 중심 (Pedestrian, Rider, Sidewalk, Parking-slot)
        client_prefs = {
            0: [8, 18, 11, 12, 9, 15],      # 차량 관련
            1: [10, 17, 4, 13],              # 보행자 관련
        }
        return client_prefs
    
    if num_clients == 3:
        client_prefs = {
            0: [8, 18, 12],              # 큰 차량
            1: [10, 17, 11],             # 사람 & 자전거
            2: [9, 15, 7, 4],            # 표지판 & 인프라
        }
        return client_prefs

    # 일반화: 클래스 인덱스를 연속 구간으로 나눔
    # 단, 배경 클래스(0~3)는 제외하고 4~18만 분할
    foreground_classes = np.arange(4, num_classes)
    splits = np.array_split(foreground_classes, num_clients)
    client_prefs = {i: list(splits[i]) for i in range(num_clients)}
    return client_prefs

def assign_non_iid_to_clients(
    matched_pairs,
    client_ratios,
    client_class_prefs,
    num_classes=NUM_CLASSES,
    random_state=42,
):
    """
    픽셀 비율 기반 극단적 Non-IID 분배.

    - 각 이미지에 대해 클래스별 픽셀 비율(hist)을 계산
    - 각 클라이언트는 자신의 선호 클래스(pref)에 해당하는 픽셀 비율 합(pref_ratio)을 점수로 사용
    - pref_ratio가 큰 클라이언트에 이미지를 우선적으로 할당
    - 모든 클라이언트에 대해 pref_ratio == 0인 이미지(선호 클래스가 전혀 없는 경우)는
      마지막에 남는 quota를 맞추기 위해 균등하게 배분

    matched_pairs: [(rgb_path, label_path), ...]
    client_ratios: 각 클라이언트가 가져갈 전체 이미지 비율 리스트 (합=1 권장)
    client_class_prefs: {client_idx: [class_idx1, class_idx2, ...], ...}
    """

    rng = np.random.RandomState(random_state)

    total_files = len(matched_pairs)
    num_clients = len(client_ratios)

    # 각 클라이언트 타깃 개수
    target_sizes = [int(round(r * total_files)) for r in client_ratios]
    diff = total_files - sum(target_sizes)
    if diff != 0:
        target_sizes[-1] += diff

    print("[Non-IID] Target sizes per client:", target_sizes)

    client_assignments = [[] for _ in range(num_clients)]
    current_sizes = [0] * num_clients

    # 1) 각 이미지의 클래스 분포(비율) 미리 계산
    print("[Non-IID] 이미지별 클래스 히스토그램 계산 중...")
    image_hists = []
    for _, label_path in tqdm(matched_pairs):
        hist = compute_label_histogram(label_path, num_classes=num_classes)
        image_hists.append(hist)

    indices = np.arange(total_files)
    rng.shuffle(indices)

    # 클라이언트별 선호 클래스 마스크 미리 계산
    client_pref_masks = []
    for c_idx in range(num_clients):
        mask = np.zeros(num_classes, dtype=bool)
        for cls in client_class_prefs.get(c_idx, []):
            if 0 <= cls < num_classes:
                mask[cls] = True
        client_pref_masks.append(mask)

    for idx in indices:
        rgb_path, label_path = matched_pairs[idx]
        hist = image_hists[idx]  # 비율 합 = 1.0 (또는 0)

        # 각 클라이언트에 대해 선호 클래스 픽셀 비율 합(pref_ratio)을 점수로 사용
        scores = []
        for c_idx in range(num_clients):
            pref_mask = client_pref_masks[c_idx]
            if not np.any(pref_mask):
                # 선호 클래스가 정의되지 않은 클라이언트는 일단 0점
                scores.append(0.0)
                continue

            pref_ratio = hist[pref_mask].sum()  # 이 클라이언트가 좋아하는 클래스 비율의 총합
            scores.append(pref_ratio)

        scores = np.array(scores, dtype=np.float32)
        max_score = scores.max()

        assigned = False

        if max_score <= 0.0:
            # 어떤 클라이언트도 이 이미지에서 선호 클래스를 가지지 않음
            # → 가장 적게 가지고 있는 클라이언트에 넣어 균등 분배
            c_idx = int(np.argmin(current_sizes))
            client_assignments[c_idx].append((rgb_path, label_path))
            current_sizes[c_idx] += 1
            assigned = True
        else:
            # 선호 클래스가 있는 클라이언트가 존재
            # 점수 높은 순서대로, 아직 quota 안 찬 클라이언트에 우선 할당
            sorted_clients = np.argsort(scores)[::-1]  # 내림차순

            for c_idx in sorted_clients:
                if scores[c_idx] <= 0.0:
                    # 그 이하 클라이언트는 선호 클래스 픽셀 비율 0이므로 볼 필요 없음
                    break
                if current_sizes[c_idx] < target_sizes[c_idx]:
                    client_assignments[c_idx].append((rgb_path, label_path))
                    current_sizes[c_idx] += 1
                    assigned = True
                    break

        # 혹시 위 로직에서 quota가 꽉 차서 할당이 안 되었으면
        # 남은 자리 가장 많은 클라이언트에 넣기
        if not assigned:
            c_idx = int(np.argmin(current_sizes))
            client_assignments[c_idx].append((rgb_path, label_path))
            current_sizes[c_idx] += 1

    print("[Non-IID] Final sizes per client:", [len(lst) for lst in client_assignments])

    # 검증: 각 클라이언트가 선호 클래스를 얼마나 많이 받았는지(샘플 50개 기준)
    for c_idx in range(num_clients):
        pref_classes = set(client_class_prefs.get(c_idx, []))
        if len(client_assignments[c_idx]) == 0 or len(pref_classes) == 0:
            continue

        match_count = 0
        sample_size = min(50, len(client_assignments[c_idx]))
        for _, label_path in client_assignments[c_idx][:sample_size]:
            hist = compute_label_histogram(label_path, num_classes=num_classes)
            present = set(np.where(hist > 0.01)[0])
            if len(present & pref_classes) > 0:
                match_count += 1

        print(
            f"  Client {c_idx}: 선호 클래스 {pref_classes}, "
            f"매칭 이미지 비율 {match_count}/{sample_size}"
        )

    return client_assignments

# SYNTHIA TrainIds 0~18 (총 19개)
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
    # 19, 20, 21, 22는 지금은 사용 안 함
}

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
        for bg in [0, 1, 2, 3]:
            valid_mask &= (arr != bg)

        arr_valid = arr[valid_mask]
        if arr_valid.size == 0:
            continue

        # 0~18로 클리핑
        arr_valid = np.clip(arr_valid, 4, num_classes - 1)

        hist = np.bincount(arr_valid, minlength=num_classes)
        total_hist += hist

    return total_hist  # shape: (num_classes,)

def plot_class_distribution_per_client(base_output_folder, num_clients, split="train", num_classes=NUM_CLASSES):
    """
    특정 클래스(예: 0,1,2,3) 제외하고 시각화
    """
    exclude_classes = [0, 1, 2, 3]
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


