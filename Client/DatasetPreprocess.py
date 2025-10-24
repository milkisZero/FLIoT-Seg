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

def plot_client_distributions(client_stats, output_folder):
    """클라이언트별 데이터 양(이미지 개수)만 시각화"""
    num_clients = len(client_stats)
    if num_clients == 0:
        return ""

    clients = [f"Client {cnum}" for cnum, _, _ in client_stats]
    train_sizes = [train for _, train, _ in client_stats]
    test_sizes  = [test  for _, _, test in client_stats]

    x = np.arange(num_clients)
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, train_sizes, width, label="Train")
    ax.bar(x + width/2, test_sizes,  width, label="Test")
    ax.set_xticks(x)
    ax.set_xticklabels(clients, rotation=45)
    ax.set_ylabel("Number of Images")
    ax.set_title("Client Dataset Size Distribution")
    ax.legend()

    plt.tight_layout()
    plot_filename = os.path.join(output_folder, "synthia_client_distribution.png")
    plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
    plt.close()
    return plot_filename

def _find_labelTrainIds_for_base(labels_root: str, base: str) -> Optional[str]:
    """
    LABELS 하위에서 base에 대응하는 *_labelTrainIds.png를 찾는다 (동일 폴더 평면 구조 가정).
    필요하면 recursive 탐색으로 바꿔도 됨.
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

def main():
    # 출력 폴더
    output_folder = "SYNTHIA_Splitted"
    
    if os.path.isdir(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder, exist_ok=True)

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

    # --- [추가] 클래스 선택 로직 (TrainIds 0~22) ---
    unique_labels = list(range(19))  # SYNTHIA TrainIds: 0..22
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
            # 인덱스를 직접 TrainIds로 쓴다고 가정 (0~22)
            # 만약 '인덱스'가 unique_labels의 위치를 의미한다면 아래처럼도 가능:
            # selected_labels = [unique_labels[i] for i in selected_indices]
            selected_labels = [unique_labels[i] for i in selected_indices]
        except Exception as e:
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

    # 클라이언트 분할
    total_files = len(matched_pairs)
    shuffled_pairs = partial_shuffle_list(matched_pairs, shuffle_intensity, random_state=42)

    client_file_pairs = []
    start_idx = 0
    for i, ratio in enumerate(client_ratios):
        if i == len(client_ratios) - 1:
            end_idx = total_files
        else:
            num_files = int(round(ratio * total_files))
            end_idx = min(start_idx + num_files, total_files)
        client_pairs = shuffled_pairs[start_idx:end_idx]
        client_file_pairs.append(client_pairs)
        start_idx = end_idx

    # 각 클라이언트 처리 및 복사
    splitting_info_lines = []
    splitting_info_lines.append("SYNTHIA Dataset Splitting Information (labelTrainIds only)")
    splitting_info_lines.append("=========================================================")
    splitting_info_lines.append(f"Data Directory: {data_dir}")
    splitting_info_lines.append(f"Output Folder: {os.path.abspath(output_folder)}")
    splitting_info_lines.append(f"Total Images: {len(matched_pairs)}")
    splitting_info_lines.append(f"Number of Clients: {num_clients}")
    splitting_info_lines.append(f"Client Ratios: {client_ratios}")
    splitting_info_lines.append(f"Training Ratios: {train_ratios}")
    splitting_info_lines.append(f"Shuffle Intensity: {shuffle_intensity}")
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

        # Train/Test split
        train_ratio = train_ratios[i]
        num_train = int(len(shuffled_pairs_cli) * train_ratio)
        train_pairs = shuffled_pairs_cli[:num_train]
        test_pairs  = shuffled_pairs_cli[num_train:]

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

        # 로그 기록
        splitting_info_lines.append(f"Client {client_num}:")
        splitting_info_lines.append(f"    Total images: {len(file_pairs)}")
        splitting_info_lines.append(f"    Selected Labels: {selected_labels}")
        splitting_info_lines.append(f"    Training images: {train_count}")
        splitting_info_lines.append(f"    Testing images: {test_count}")
        splitting_info_lines.append(f"    Output: client{client_num}/")
        splitting_info_lines.append("")

    config = {
        "selected_labels": selected_labels,
        "num_classes": len(selected_labels) 
    }
    
    config_filename = os.path.join("..", "Server", "config.json")
    os.makedirs(os.path.dirname(config_filename), exist_ok=True)
    with open(config_filename, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
    print(f"\n설정 파일 저장: {config_filename}")

    # 분포 시각화(이미지 개수 막대그래프)
    if client_stats:
        plot_filename = plot_client_distributions(client_stats, output_folder)
        if plot_filename:
            splitting_info_lines.append(f"Distribution plot: {os.path.basename(plot_filename)}")

    # 정보 파일 저장
    info_filename = os.path.join(output_folder, "splitting_info.txt")
    with open(info_filename, "w", encoding="utf-8") as f:
        f.write("\n".join(splitting_info_lines))

    print("\n처리 완료!")
    print(f"출력 폴더: {os.path.abspath(output_folder)}")
    print(f"정보 파일: {info_filename}")

if __name__ == "__main__":
    main()
