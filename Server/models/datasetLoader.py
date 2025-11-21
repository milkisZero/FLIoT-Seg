from PIL import Image
import numpy as np
import os
import glob
from tqdm import tqdm

def load_synthia_dataset(target_size, binary=False, object_classes=None, show_progress=True):
    # target_size는 (H, W, C) 형식으로 들어옴
    target_h, target_w = target_size[0], target_size[1]
    
    # PIL은 (W, H) 순서 필요
    pil_size = (target_w, target_h)  # (Width, Height)
    
    print(f"Target shape (H×W): {target_h}×{target_w}")
    print(f"PIL resize size (W×H): {pil_size}")
    
    def _load_images(rgb_dir, label_dir, phase="train"):
        X, y = [], []
        rgb_files = sorted(glob.glob(os.path.join(rgb_dir, "*.png")))
        total = len(rgb_files)
        print(f"[{phase}] RGB dir: {rgb_dir}")
        print(f"[{phase}] LABEL dir: {label_dir}")
        print(f"[{phase}] found {total} RGB files")
        
        skipped = 0
        iterator = tqdm(
            rgb_files,
            desc=f"{phase}: loading",
            unit="img",
            total=total,
            disable=not show_progress
        )
        
        for rgb_file in iterator:
            base = os.path.basename(rgb_file)
            label_file = os.path.join(label_dir, base)
            
            if not os.path.exists(label_file):
                skipped += 1
                iterator.set_postfix(skipped=skipped)
                continue
            
            try:
                img = Image.open(rgb_file).convert("RGB")
                img_resized = img.resize(pil_size, Image.BILINEAR)  # (W, H)
                img_array = np.array(img_resized)  # NumPy: (H, W, C)
                
                lab = Image.open(label_file).convert("L")
                lab_resized = lab.resize(pil_size, Image.NEAREST)  # (W, H)
                lab_array = np.array(lab_resized)  # NumPy: (H, W)
                
                if len(X) == 0:
                    print(f"[{phase}] First image shape: {img_array.shape}")
                    print(f"[{phase}] First label shape: {lab_array.shape}")
                
                if binary and object_classes is not None:
                    mask = np.isin(lab_array, object_classes).astype(np.uint8)
                    y.append(mask)
                else:
                    y.append(lab_array)
                    
                X.append(img_array)
                
            except Exception as e:
                skipped += 1
                iterator.set_postfix(skipped=skipped)
                print(f"Error reading {rgb_file}: {e}")
                continue
        
        print(f"[{phase}] skipped (missing or broken): {skipped}")
        return np.array(X), np.array(y)
    
    print("start datasetLoader")

    X_test, y_test = _load_images(
        os.path.join("SYNTHIA_Splitted", "global_test", "RGB"),
        os.path.join("SYNTHIA_Splitted", "global_test", "LABELS"),
        phase="test"
    )
    print("Loaded Server side data: X =", X_test.shape, "y =", y_test.shape)

    return X_test, y_test
