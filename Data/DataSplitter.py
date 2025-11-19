import sys
import os
import glob
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import json
import shutil
from typing import List, Tuple, Optional
import shutil
from non_iid_split import (
    NUM_CLASSES,
    compute_class_distribution_in_dir,
    plot_class_distribution_per_client,
    compute_label_histogram,  # 새로 사용
)

load_dotenv()

def partial_shuffle_list(file_list, intensity, random_state=None):
    """파일 리스트를 부분적으로 셔플 (intensity∈[0,1])"""
    n = len(file_list)
    if intensity <= 0:
        return file_list.copy()
    if intensity >= 1:
        rng = np.random.RandomState(random_state) if random_state is not None else np.random
        shuffled = file_list.copy()
        rng.shuffle(shuffled)
        return shuffled

    indices = np.arange(n)
    normalized = indices / (n - 1) if n > 1 else indices
    rng = np.random.RandomState(random_state) if random_state is not None else np.random
    random_vals = rng.rand(n)
    keys = (1 - intensity) * normalized + intensity * random_vals
    order = np.argsort(keys)
    return [file_list[i] for i in order]

def copy_pairs_with_structure(file_pairs: List[Tuple[str, str]], output_dir: str, split_name: str):
    """
    RGB와 labelTrainIds 파일만 복사.
    - output_dir/RGB
    - output_dir/LABELS   (정답은 여기로)
    """
    rgb_dir = os.path.join(output_dir, "RGB")
    labels_dir = os.path.join(output_dir, "LABELS")

    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    for i, (rgb_file, eval_label_file) in enumerate(tqdm(file_pairs, desc=f"복사 중 ({split_name})")):
        base_name = f"{split_name}_{i:06d}.png"
        shutil.copy2(rgb_file, os.path.join(rgb_dir, base_name))
        shutil.copy2(eval_label_file, os.path.join(labels_dir, base_name))

def plot_dataset_distribution(global_count, server_count, client_stats, output_folder):
    categories = []
    counts = []
    colors = []

    # 1) Global / Server
    categories.append("Global Test")
    counts.append(global_count)
    colors.append('#2E86AB')  # 파란색 (Global)

    categories.append("Server Data")
    counts.append(server_count)
    colors.append('#A23B72')  # 보라색 (Server)

    # 2) 각 클라이언트 Train/Test
    client_train_color = '#F18F01'  # 주황색 (Client Train)
    client_test_color = '#C73E1D'   # 빨간색 (Client Test)
    
    for cnum, train_cnt, test_cnt in client_stats:
        categories.append(f"Client{cnum} Train")
        counts.append(train_cnt)
        colors.append(client_train_color)

        categories.append(f"Client{cnum} Test")
        counts.append(test_cnt)
        colors.append(client_test_color)

    if len(categories) == 0:
        return ""

    x = np.arange(len(categories))

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(x, counts, color=colors)

    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha="right")
    ax.set_ylabel("Number of Images")
    ax.set_title("Overall Dataset Split (Global / Server / Clients)")

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2E86AB', label='Global Test'),
        Patch(facecolor='#A23B72', label='Server Data'),
        Patch(facecolor='#F18F01', label='Client Train'),
        Patch(facecolor='#C73E1D', label='Client Test')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    plot_filename = os.path.join(output_folder, "synthia_dataset_distribution.png")
    plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
    plt.close()
    return plot_filename

def _find_labelTrainIds_for_base(labels_root: str, base: str) -> Optional[str]:
    """
    LABELS 하위에서 base에 대응하는 *_labelTrainIds.png를 찾는다 (동일 폴더 평면 구조 가정).
    """
    cand = os.path.join(labels_root, f"{base}_labelTrainIds.png")
    return cand if os.path.isfile(cand) else None

def collect_synthia_pairs_with_trainIds(data_dir: str) -> List[Tuple[str, str]]:
    """
    data_dir(예: SYNTHIA)에서 (RGB, LABELS/*_labelTrainIds.png) 쌍만 수집.
    labelTrainIds가 없으면 해당 RGB는 스킵.
    """
    rgb_root    = os.path.join(data_dir, "RGB")
    labels_root = os.path.join(data_dir, "LABELS")

    if not os.path.isdir(rgb_root) or not os.path.isdir(labels_root):
        print("RGB 또는 LABELS 폴더를 찾을 수 없습니다.")
        return []

    rgb_files = sorted(glob.glob(os.path.join(rgb_root, "*.png")))
    print(f"RGB 이미지 후보: {len(rgb_files)}개 발견")

    matched, missing = [], 0
    for rgb_path in tqdm(rgb_files, desc="RGB-TrainIds 매칭"):
        base = os.path.splitext(os.path.basename(rgb_path))[0]
        eval_label = _find_labelTrainIds_for_base(labels_root, base)
        if eval_label is None:
            missing += 1
            continue
        matched.append((rgb_path, eval_label))

    print(f"labelTrainIds와 매칭된 쌍: {len(matched)}개 (labelTrainIds 없음: {missing}개)")
    return matched

# ---------------- Dirichlet label-non-iid 관련 함수 추가 ---------------- #

def get_dominant_label_for_image(
    label_path: str,
    selected_labels: List[int],
    num_classes: int = NUM_CLASSES
) -> Optional[int]:
    """
    세그멘테이션 라벨에서 '우세 클래스(dominant class)'를 하나 뽑아 이미지 레이블로 사용.
    - selected_labels에 포함된 클래스 중에서 픽셀 비율이 가장 큰 클래스를 반환.
    - selected_labels 픽셀이 전혀 없으면 None 반환.
    """
    hist = compute_class_distribution_for_dominant(label_path, num_classes=num_classes)

    if hist is None or hist.sum() == 0:
        return None

    mask = np.zeros_like(hist, dtype=bool)
    for cls_id in selected_labels:
        if 0 <= cls_id < num_classes:
            mask[cls_id] = True

    if not np.any(mask):
        return None

    masked_hist = hist.copy()
    masked_hist[~mask] = 0.0
    if masked_hist.sum() == 0:
        return None

    dominant_label = int(np.argmax(masked_hist))
    return dominant_label

def compute_class_distribution_for_dominant(label_path: str, num_classes: int = NUM_CLASSES) -> Optional[np.ndarray]:
    hist = compute_label_histogram(label_path, num_classes=num_classes)
    if hist is None:
        return None

    # 배경/공통 클래스 제거: void(0), sky(1), building(2), road(3)
    bg_classes = [0, 1, 2, 3]
    for c in bg_classes:
        hist[c] = 0.0

    # 혹시 전부 0이면 dominant를 못 정하므로 None
    if hist.sum() == 0:
        return None

    hist /= hist.sum()
    return hist


def dirichlet_label_non_iid_split(
    matched_pairs: List[Tuple[str, str]],
    num_clients: int,
    client_ratios: List[float],
    selected_labels: List[int],
    alpha: float,
    num_classes: int = NUM_CLASSES,
    random_state: int = 42,
) -> List[List[Tuple[str, str]]]:
    """
    Dirichlet 기반 label-non-iid 분할 (classification에서 많이 쓰는 방식을 세그멘테이션에 적용).

    절차:
    1) 각 이미지에 대해 dominant label(가장 많이 나온 selected_labels 중 하나)을 정한다.
    2) 클래스별로 이미지 인덱스 리스트를 만든다.
    3) 각 클래스에 대해 Dirichlet(alpha)로 클라이언트 비율을 샘플링하고,
       그 비율에 맞게 이미지 index를 나눈다.
    4) dominant label이 없는 이미지(선택된 라벨이 전혀 없는 경우)는
       client_ratios를 고려하여 부족한 클라이언트에 채워 넣는다.

    반환:
    - client_file_pairs: 길이 num_clients인 리스트
      각 원소는 [(rgb_path, label_path), ...] 형태
    """
    rng = np.random.RandomState(random_state)

    total_files = len(matched_pairs)
    client_file_indices = [[] for _ in range(num_clients)]

    # 1) 이미지별 dominant label 계산
    label_to_indices = {cls: [] for cls in selected_labels}
    unlabeled_indices = []

    print("[Dirichlet] 이미지별 dominant label 계산 중...")
    for idx, (_, label_path) in enumerate(tqdm(matched_pairs)):
        dom_label = get_dominant_label_for_image(
            label_path, selected_labels, num_classes=num_classes
        )
        if dom_label is None or dom_label not in label_to_indices:
            unlabeled_indices.append(idx)
        else:
            label_to_indices[dom_label].append(idx)

    print("[Dirichlet] 클래스별 이미지 수:")
    for cls, idxs in label_to_indices.items():
        print(f"  class {cls}: {len(idxs)}개")

    # 2) 클래스별 Dirichlet 분할
    for cls, idxs in label_to_indices.items():
        if not idxs:
            continue

        idxs = np.array(idxs)
        rng.shuffle(idxs)

        # 각 클래스에 대해 Dirichlet(alpha) 샘플
        proportions = rng.dirichlet(alpha * np.ones(num_clients))
        split_points = (np.cumsum(proportions) * len(idxs)).astype(int)[:-1]
        splits = np.split(idxs, split_points)

        for c_idx in range(num_clients):
            client_file_indices[c_idx].extend(splits[c_idx].tolist())

    # 3) client_ratios 기반 target size 계산
    target_sizes = [int(round(r * total_files)) for r in client_ratios]
    diff = total_files - sum(target_sizes)
    if diff != 0:
        target_sizes[-1] += diff
    print("[Dirichlet] Target sizes per client (approx):", target_sizes)

    # 4) dominant label이 없는 이미지들(unlabeled_indices)을 부족한 클라에 채워 넣기
    current_sizes = [len(idxs) for idxs in client_file_indices]
    print("[Dirichlet] 현재 사이즈(클래스 기반 분배 후):", current_sizes)
    print("[Dirichlet] dominant label 없는 이미지 수:", len(unlabeled_indices))

    for idx in unlabeled_indices:
        deficits = [target_sizes[c] - current_sizes[c] for c in range(num_clients)]
        if any(d > 0 for d in deficits):
            c_idx = int(np.argmax(deficits))
        else:
            c_idx = int(np.argmin(current_sizes))

        client_file_indices[c_idx].append(idx)
        current_sizes[c_idx] += 1

    print("[Dirichlet] 최종 클라이언트별 이미지 수:", current_sizes)

    # 5) index → (rgb_path, label_path)로 변환
    client_file_pairs: List[List[Tuple[str, str]]] = [[] for _ in range(num_clients)]
    for c_idx in range(num_clients):
        for idx in client_file_indices[c_idx]:
            client_file_pairs[c_idx].append(matched_pairs[idx])

    return client_file_pairs

def stratified_train_test_split_per_client(
    file_pairs: List[Tuple[str, str]],
    train_ratio: float,
    selected_labels: List[int],
    num_classes: int = NUM_CLASSES,
    random_state: int = 0,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """
    한 클라이언트의 file_pairs를 dominant label 기준으로 stratified train/test split.
    - 각 이미지에 대해 dominant label을 계산 (selected_labels 중에서 가장 많이 나온 클래스)
    - dominant label(또는 None) 별로 버킷을 나누고, 각 버킷 안에서 train_ratio 비율로 나눔
    - 버킷이 너무 작은 경우:
        - len(bucket) > 0 이면 최소 1개는 train에 들어가도록 처리 (train_ratio > 0 가정)
        - len(bucket) > 1 이고 train_ratio < 1 이면 최소 1개는 test에 남도록 처리
    """
    rng = np.random.RandomState(random_state)

    if not file_pairs:
        return [], []

    # 1) dominant label 기준으로 인덱스 버킷 생성
    label_to_indices: dict = {}
    for idx, (_, label_path) in enumerate(file_pairs):
        dom_label = get_dominant_label_for_image(
            label_path, selected_labels, num_classes=num_classes
        )
        # selected_labels에 없거나 None이면 별도 버킷(-1)으로 처리
        key = dom_label if (dom_label in selected_labels) else -1
        if key not in label_to_indices:
            label_to_indices[key] = []
        label_to_indices[key].append(idx)

    train_indices = []
    test_indices = []

    # 2) 버킷별로 train/test 나누기
    for key, idxs in label_to_indices.items():
        idxs = np.array(idxs)
        rng.shuffle(idxs)

        n = len(idxs)
        k = int(round(n * train_ratio))

        # 너무 극단 값 방지 (원하면 조정)
        if train_ratio > 0 and k == 0 and n > 0:
            k = 1
        if train_ratio < 1 and k == n and n > 1:
            k = n - 1

        train_indices.extend(idxs[:k])
        test_indices.extend(idxs[k:])

    # 3) 전체적으로 한 번 더 섞어줌 (선택)
    rng.shuffle(train_indices)
    rng.shuffle(test_indices)

    train_pairs = [file_pairs[i] for i in train_indices]
    test_pairs = [file_pairs[i] for i in test_indices]

    return train_pairs, test_pairs


# ---------------------------------------------------------------------- #

def main():
    # 출력 폴더
    output_folder = "../Client/SYNTHIA_Splitted"
    
    if os.path.isdir(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder, exist_ok=True)
    
    server_output_folder = "../Server/SYNTHIA_Splitted"
    
    if os.path.isdir(server_output_folder):
        shutil.rmtree(server_output_folder)
    os.makedirs(server_output_folder, exist_ok=True)

    # 데이터 경로 (최상위에 RGB, LABELS 폴더가 있다고 가정)
    data_dir = "SYNTHIA"
    if not os.path.exists(data_dir):
        print(f"SYNTHIA 데이터 디렉토리를 찾을 수 없습니다: {data_dir}")
        print("경로를 확인해주세요.")
        return

    # (RGB, labelTrainIds) 매칭
    matched_pairs = collect_synthia_pairs_with_trainIds(data_dir)
    if len(matched_pairs) == 0:
        print("매칭되는 (RGB, labelTrainIds) 쌍이 없습니다. 경로/파일 구성을 확인해주세요.")
        return
    print(f"총 {len(matched_pairs)}개의 (RGB, labelTrainIds) 쌍을 발견했습니다.")

    # --- [추가] 클래스 선택 로직 (TrainIds 0~18) ---
    unique_labels = list(range(19))
    selected_idx_str = os.getenv("LABEL", "").strip()

    if selected_idx_str.upper() == "ALL":
        selected_indices = list(range(len(unique_labels)))
        selected_labels = unique_labels
    else:
        print(f"Selected index string: {selected_idx_str}")
        if not selected_idx_str:
            selected_idx_str = input(
                "\nEnter the index(es) of the label types to keep (space separated, e.g., 8 10 11): "
            )
        try:
            selected_indices = [int(x.strip()) for x in selected_idx_str.split()]
            selected_labels = [unique_labels[i] for i in selected_indices]
        except Exception:
            print("Error processing the input. Make sure to enter valid indices separated by spaces.")
            return

    print(f"Selected label types: {selected_labels}")

    # 클라이언트 수
    try:
        args = sys.argv[1:]
        if len(args) == 1:
            num_clients = int(args[0])
            print(f"명령행에서 {num_clients}개 클라이언트 사용")
        else:
            num_clients_str = os.getenv("CLIENT", "").strip()
            if not num_clients_str:
                num_clients_str = input("\n클라이언트 수를 입력하세요: ").strip()
            num_clients = int(num_clients_str)
            if num_clients <= 0:
                raise ValueError
    except ValueError:
        print("잘못된 클라이언트 수입니다")
        return
    print(f"클라이언트 수: {num_clients}")

    # 클라이언트 비율
    client_ratios_str = os.getenv("RATIO", "").strip()
    if not client_ratios_str:
        client_ratios_str = input("각 클라이언트의 비율을 입력하세요 (공백 구분, 예: 1 1 1): ").strip()
    try:
        client_ratios = [float(x.strip()) for x in client_ratios_str.split()]
        if len(client_ratios) != num_clients or any(r < 0 for r in client_ratios) or sum(client_ratios) == 0:
            raise ValueError
    except ValueError:
        print("잘못된 비율 입력입니다.")
        return
    total_ratio = sum(client_ratios)
    client_ratios = [r / total_ratio for r in client_ratios]
    print(f"클라이언트 비율 (정규화): {client_ratios}")

    # 훈련/테스트 비율
    train_ratios_str = os.getenv("TR_RATIO", "").strip()
    if not train_ratios_str:
        train_ratios_str = input("각 클라이언트의 훈련 비율을 입력하세요 (공백 구분, 예: 0.8 0.7 0.9): ").strip()
    try:
        train_ratios = [float(x.strip()) for x in train_ratios_str.split()]
        if len(train_ratios) != num_clients:
            raise ValueError
        for tr in train_ratios:
            if not (0 < tr < 1):
                raise ValueError
    except ValueError:
        print("잘못된 훈련 비율입니다. 각 값은 (0,1) 이어야 하며 개수는 클라이언트 수와 같아야 합니다.")
        return
    print(f"훈련 데이터 비율: {train_ratios}")

    # 셔플 강도
    shuffle_intensity_str = os.getenv("SHUFFLE_INTENSITY", "").strip()
    if not shuffle_intensity_str:
        shuffle_intensity_str = input("셔플 강도를 입력하세요 (0: 섞지 않음, 1: 완전 셔플, 예: 0.8): ").strip()
    try:
        shuffle_intensity = float(shuffle_intensity_str)
    except ValueError:
        print("잘못된 셔플 강도입니다.")
        return
    if not (0 <= shuffle_intensity <= 1):
        print("셔플 강도는 0과 1 사이여야 합니다.")
        return
    print(f"셔플 강도: {shuffle_intensity}")

    # 전체 파일 수
    total_files = len(matched_pairs)

    # 1) 전체를 한 번 셔플
    shuffled_all = partial_shuffle_list(matched_pairs, shuffle_intensity, random_state=42)

    # 2) Global Test / Server 비율 입력
    global_ratio_str = os.getenv("GLOBAL_TEST_RATIO", "").strip()
    if not global_ratio_str:
        global_ratio_str = input("Global Test 비율을 입력하세요 (예: 0.15): ").strip()

    server_ratio_str = os.getenv("SERVER_RATIO", "").strip()
    if not server_ratio_str:
        server_ratio_str = input("Server용 데이터 비율을 입력하세요 (예: 0.15): ").strip()

    try:
        global_ratio = float(global_ratio_str)
        server_ratio = float(server_ratio_str)
        if global_ratio < 0 or server_ratio < 0 or global_ratio + server_ratio >= 1.0:
            raise ValueError
    except ValueError:
        print("잘못된 Global/Server 비율입니다. 두 값은 0 이상, 합은 1보다 작아야 합니다.")
        return

    # 3) 개수 계산
    num_global = int(round(total_files * global_ratio))
    num_server = int(round(total_files * server_ratio))
    if num_global + num_server > total_files:
        num_server = total_files - num_global

    global_pairs = shuffled_all[:num_global]
    server_pairs = shuffled_all[num_global:num_global + num_server]
    remaining_pairs = shuffled_all[num_global + num_server:]

    print(f"Global Test: {len(global_pairs)}개")
    print(f"Server data: {len(server_pairs)}개")
    print(f"Client용 남은 데이터: {len(remaining_pairs)}개")

    # 4) Global Test / Server 데이터 복사
    global_test_dir = os.path.join(server_output_folder, "global_test")
    server_dir   = os.path.join(server_output_folder, "serverdata")

    os.makedirs(global_test_dir, exist_ok=True)
    os.makedirs(server_dir,   exist_ok=True)

    copy_pairs_with_structure(global_pairs, global_test_dir, "global")
    copy_pairs_with_structure(server_pairs, server_dir, "serverdata")

    # ---------------- Dirichlet label-non-iid 분할만 사용 ---------------- #
    print("\n[분배] Dirichlet 기반 label Non-IID 분배를 사용합니다.")

    base_pairs = partial_shuffle_list(remaining_pairs, shuffle_intensity, random_state=42)

    dirichlet_alpha_str = os.getenv("DIRICHLET_ALPHA", "").strip()
    if not dirichlet_alpha_str:
        dirichlet_alpha_str = input(
            "Dirichlet alpha 값을 입력하세요 (예: 0.5 또는 0.1, 작을수록 더 극단 Non-IID): "
        ).strip()
    try:
        dirichlet_alpha = float(dirichlet_alpha_str)
        if dirichlet_alpha <= 0:
            raise ValueError
    except ValueError:
        print("잘못된 Dirichlet alpha 값입니다. 양수여야 합니다.")
        return

    print(f"사용할 Dirichlet alpha: {dirichlet_alpha}")

    client_file_pairs = dirichlet_label_non_iid_split(
        matched_pairs=base_pairs,
        num_clients=num_clients,
        client_ratios=client_ratios,
        selected_labels=selected_labels,
        alpha=dirichlet_alpha,
        num_classes=NUM_CLASSES,
        random_state=42,
    )
    # --------------------------------------------------------------------- #

    # 각 클라이언트 처리 및 복사
    splitting_info_lines = []
    splitting_info_lines.append("SYNTHIA Dataset Splitting Information (labelTrainIds only, Dirichlet non-iid)")
    splitting_info_lines.append("=========================================================")
    splitting_info_lines.append(f"Data Directory: {data_dir}")
    splitting_info_lines.append(f"Output Folder: {os.path.abspath(output_folder)}")
    splitting_info_lines.append(f"Total Images: {len(matched_pairs)}")
    splitting_info_lines.append(f"Number of Clients: {num_clients}")
    splitting_info_lines.append(f"Client Ratios: {client_ratios}")
    splitting_info_lines.append(f"Training Ratios: {train_ratios}")
    splitting_info_lines.append(f"Shuffle Intensity: {shuffle_intensity}")
    splitting_info_lines.append(f"Global Test ratio: {global_ratio} -> {len(global_pairs)} images")
    splitting_info_lines.append(f"Server ratio: {server_ratio} -> {len(server_pairs)} images")
    splitting_info_lines.append(f"Client total for splitting: {len(remaining_pairs)} images")
    splitting_info_lines.append(f"Dirichlet alpha: {dirichlet_alpha}")
    splitting_info_lines.append(f"Selected labels: {selected_labels}")
    splitting_info_lines.append("")

    client_stats = []  # (client_num, train_count, test_count)

    for i, file_pairs in enumerate(client_file_pairs):
        client_num = i + 1
        print(f"\nClient {client_num} 처리 중...")
        print(f"  총 {len(file_pairs)}개 이미지")

        if not file_pairs:
            print(f"  클라이언트 {client_num}에 할당된 데이터가 없습니다.")
            continue

        # 추가 셔플
        shuffled_pairs_cli = partial_shuffle_list(file_pairs, shuffle_intensity, random_state=42 + i)

        # Stratified Train/Test split (dominant label 기준)
        train_ratio = train_ratios[i]
        train_pairs, test_pairs = stratified_train_test_split_per_client(
            file_pairs=file_pairs,
            train_ratio=train_ratio,
            selected_labels=selected_labels,
            num_classes=NUM_CLASSES,
            random_state=42 + i,
        )
        print(f"  훈련: {len(train_pairs)}개, 테스트: {len(test_pairs)}개")

        # 폴더 생성 및 복사
        client_dir = os.path.join(output_folder, f"client{client_num}")
        train_dir  = os.path.join(client_dir, "train")
        test_dir   = os.path.join(client_dir, "test")
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir,  exist_ok=True)

        copy_pairs_with_structure(train_pairs, train_dir, "train")
        copy_pairs_with_structure(test_pairs,  test_dir,  "test")

        # 단순 통계(이미지 개수)
        train_count = len(train_pairs)
        test_count  = len(test_pairs)
        client_stats.append((client_num, train_count, test_count))

        splitting_info_lines.append(f"Client {client_num}:")
        splitting_info_lines.append(f"    Total images: {len(file_pairs)}")
        splitting_info_lines.append(f"    Selected Labels: {selected_labels}")
        splitting_info_lines.append(f"    Training images: {train_count}")
        splitting_info_lines.append(f"    Testing images: {test_count}")
        splitting_info_lines.append(f"    Output: client{client_num}/")
        splitting_info_lines.append("")
    
    config_filename = os.path.join("..", "Server", "config.json")
    os.makedirs(os.path.dirname(config_filename), exist_ok=True)
    with open(config_filename, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # 모델 설정 업데이트
    config['model']['num_classes'] = len(selected_labels) 
    config['model']['selected_labels'] = selected_labels
    
    # config.json 저장
    config_filename = os.path.join("..", "Server", "config.json")
    os.makedirs(os.path.dirname(config_filename), exist_ok=True)
    
    with open(config_filename, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
    print(f"\n설정 파일 저장: {config_filename}")

    # 분포 시각화(이미지 개수 막대그래프)
    if client_stats:
        plot_filename = plot_dataset_distribution(
            global_count=len(global_pairs),
            server_count=len(server_pairs),
            client_stats=client_stats,
            output_folder=".",
        )
        if plot_filename:
            splitting_info_lines.append(f"Distribution plot: {os.path.basename(plot_filename)}")

    # 클래스 분포 시각화 (클라이언트별 Train/Test)
    try:
        plot_class_distribution_per_client(
            base_output_folder=output_folder,
            num_clients=num_clients,
            split="train",
            num_classes=NUM_CLASSES
        )
        plot_class_distribution_per_client(
            base_output_folder=output_folder,
            num_clients=num_clients,
            split="test",
            num_classes=NUM_CLASSES
        )
    except Exception as e:
        print(f"[경고] 클래스 분포 플롯 생성 중 에러 발생: {e}")

    # 정보 파일 저장
    info_filename = os.path.join('.', "splitting_info.txt")
    with open(info_filename, "w", encoding="utf-8") as f:
        f.write("\n".join(splitting_info_lines))

    print("\n처리 완료!")
    print(f"출력 폴더: {os.path.abspath(output_folder)}")
    print(f"정보 파일: {info_filename}")

if __name__ == "__main__":
    main()
