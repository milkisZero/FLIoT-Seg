from PIL import Image
import numpy as np
import os
import glob
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial

def _process_single_image(args):
    """단일 이미지 처리 함수 (멀티프로세싱용)"""
    rgb_file, label_file, pil_size, binary, object_classes = args
    
    try:
        # RGB 이미지 로드 & 리사이즈
        img = Image.open(rgb_file).convert("RGB")
        img_resized = img.resize(pil_size, Image.BILINEAR)
        img_array = np.array(img_resized)
        
        # 레이블 로드 & 리사이즈
        lab = Image.open(label_file).convert("L")
        lab_resized = lab.resize(pil_size, Image.NEAREST)
        lab_array = np.array(lab_resized)
        
        # Binary 처리
        if binary and object_classes is not None:
            mask = np.isin(lab_array, object_classes).astype(np.uint8)
            return img_array, mask, None  # 성공
        else:
            return img_array, lab_array, None  # 성공
            
    except Exception as e:
        return None, None, str(e)  # 실패


def load_synthia_dataset(target_size, binary=False, object_classes=None, show_progress=True, num_workers=None):
    # target_size는 (H, W, C) 형식으로 들어옴
    target_h, target_w = target_size[0], target_size[1]
    
    # PIL은 (W, H) 순서 필요
    pil_size = (target_w, target_h)  # (Width, Height)
    
    print(f"Target shape (H×W): {target_h}×{target_w}")
    print(f"PIL resize size (W×H): {pil_size}")
    
    # 워커 수 설정 (기본: CPU 코어 수 - 1, 최소 1)
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)
    print(f"Using {num_workers} workers for parallel processing")
    
    def _load_images(rgb_dir, label_dir, phase="train"):
        X, y = [], []
        rgb_files = sorted(glob.glob(os.path.join(rgb_dir, "*.png")))
        total = len(rgb_files)
        print(f"[{phase}] RGB dir: {rgb_dir}")
        print(f"[{phase}] LABEL dir: {label_dir}")
        print(f"[{phase}] found {total} RGB files")
        
        # 레이블 파일 존재 여부 확인 및 페어 생성
        valid_pairs = []
        for rgb_file in rgb_files:
            base = os.path.basename(rgb_file)
            label_file = os.path.join(label_dir, base)
            if os.path.exists(label_file):
                valid_pairs.append((rgb_file, label_file, pil_size, binary, object_classes))
        
        print(f"[{phase}] valid pairs: {len(valid_pairs)} (skipped {total - len(valid_pairs)} missing files)")
        
        if len(valid_pairs) == 0:
            print(f"[{phase}] No valid image pairs found!")
            return np.array([]), np.array([])
        
        # 멀티프로세싱으로 병렬 처리
        skipped = 0
        with Pool(num_workers) as pool:
            results = list(tqdm(
                pool.imap(_process_single_image, valid_pairs),
                total=len(valid_pairs),
                desc=f"{phase}: loading",
                unit="img",
                disable=not show_progress
            ))
        
        # 결과 수집
        for idx, (img_array, lab_array, error) in enumerate(results):
            if error is not None:
                skipped += 1
                if skipped <= 5:  # 처음 5개 에러만 출력
                    print(f"Error processing {valid_pairs[idx][0]}: {error}")
                continue
            
            if len(X) == 0:
                print(f"[{phase}] First image shape: {img_array.shape}")
                print(f"[{phase}] First label shape: {lab_array.shape}")
            
            X.append(img_array)
            y.append(lab_array)
        
        print(f"[{phase}] skipped (errors): {skipped}")
        print(f"[{phase}] successfully loaded: {len(X)}")
        
        return np.array(X), np.array(y)
    
    print("start datasetLoader")

    X_test, y_test = _load_images(
        os.path.join("SYNTHIA_Splitted", "global_test", "RGB"),
        os.path.join("SYNTHIA_Splitted", "global_test", "LABELS"),
        phase="test"
    )
    print("Loaded Server side data: X =", X_test.shape, "y =", y_test.shape)

    return X_test, y_test