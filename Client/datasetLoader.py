from PIL import Image
import numpy as np
import os
import glob
from tqdm import tqdm

def load_synthia_dataset(binary=False, object_classes=None, show_progress=True):
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
                target_size = (256, 256) 
                
                # RGB 이미지
                img = np.array(
                    Image.open(rgb_file).convert("RGB").resize(target_size, Image.BILINEAR)
                )
                
                # 라벨 이미지 (정수값 유지)
                lab = np.array(
                    Image.open(label_file).resize(target_size, Image.NEAREST)
                )
                
                if binary and object_classes is not None:
                    mask = np.isin(lab, object_classes).astype(np.uint8)
                    y.append(mask)
                else:
                    y.append(lab)

                X.append(img)

            except Exception as e:
                skipped += 1
                iterator.set_postfix(skipped=skipped)
                print(f"⚠️ Error reading {rgb_file}: {e}")
                continue

        print(f"[{phase}] skipped (missing or broken): {skipped}")
        return np.array(X), np.array(y)

    print("start datasetLoader")

    X_train, y_train = _load_images(
        os.path.join("SYNTHIA_Splitted/client", "train", "RGB"),
        os.path.join("SYNTHIA_Splitted/client", "train", "LABELS"),
        phase="train"
    )
    print("Loaded train data: X =", X_train.shape, "y =", y_train.shape)

    X_test, y_test = _load_images(
        os.path.join("SYNTHIA_Splitted/client", "test", "RGB"),
        os.path.join("SYNTHIA_Splitted/client", "test", "LABELS"),
        phase="test"
    )
    print("Loaded test data: X =", X_test.shape, "y =", y_test.shape)

    return X_train, y_train, X_test, y_test
