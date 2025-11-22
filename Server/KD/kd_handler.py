# models/kd_handler.py
import os
import glob
import numpy as np
import tensorflow as tf
from PIL import Image
from tqdm import tqdm


def load_images_from_dirs(
    rgb_dir,
    label_dir,
    target_size,
    binary=False,
    object_classes=None,
    show_progress=True,
    phase="custom",
    filelist=None,   # filenames 순서가 중요하면 지정
):
    """
    - rgb_dir / label_dir 폴더를 받아서 이미지/라벨 로딩
    - target_size: (H, W, C)
    - filelist가 주어지면 그 순서대로만 로딩(teacher logits 순서 맞출 때 사용)

    return:
        X: (N,H,W,3) uint8
        y: (N,H,W)   uint8 또는 int
    """
    target_h, target_w = target_size[0], target_size[1]

    pil_size = (target_w, target_h)  # (W,H)

    print(f"Target shape (H×W): {target_h}×{target_w}")
    print(f"PIL resize size (W×H): {pil_size}")
    print(f"[{phase}] RGB dir: {rgb_dir}")
    print(f"[{phase}] LABEL dir: {label_dir}")

    X, y = [], []

    if filelist is None:
        rgb_files = sorted(glob.glob(os.path.join(rgb_dir, "*.png")))
    else:
        # filenames(list[str])가 들어오면 그 순서대로 경로 구성
        rgb_files = [os.path.join(rgb_dir, fn) for fn in filelist]

    total = len(rgb_files)
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
            img_resized = img.resize(pil_size, Image.BILINEAR)
            img_array = np.array(img_resized)

            lab = Image.open(label_file).convert("L")
            lab_resized = lab.resize(pil_size, Image.NEAREST)
            lab_array = np.array(lab_resized)

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


class KDHandler:
    """
    서버 사이드 지식증류(KD) 전담 핸들러.
    - teacher logits + filenames 로드
    - 서버 distill 데이터 로드(teacher 순서 정렬)
    - KD 1 epoch 수행
    """

    def __init__(self, target_size, ignore_label=255, ce_loss_fn=None):
        """
        target_size: (H,W,3)
        ignore_label: 255 기본
        ce_loss_fn: ignore_label 마스킹된 CE 함수 (BaseGlobalModel의 loss_sparse_ce_ignore_255)
        """
        self.target_size = target_size
        self.ignore_label = ignore_label
        self.ce_loss_fn = ce_loss_fn

        self.kd_X = None            # (N,H,W,3) float32
        self.kd_y = None            # (N,H,W) int32
        self.teacher_logits = None  # (N,H,W,C) float32
        self.teacher_names = None
        self.opt = tf.keras.optimizers.Adam(1e-4)

    def _imagenet_normalize_np(self, X):
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        return (X - mean) / std

    def load_kd_data(
        self,
        server_dir,
        logits_dir,
        show_progress=False,
        use_imagenet_norm=False,
        logits_name="teacher_logits.npy",
        names_name="teacher_filenames.npy",
    ):
        """
        teacher_pipeline에서 만든 logits + filenames를 읽고,
        filenames 순서대로 서버 distill 데이터를 로딩해 1:1 매칭.
        """
        t_logits_path = os.path.join(logits_dir, logits_name)
        t_names_path  = os.path.join(logits_dir, names_name)

        teacher_logits = np.load(t_logits_path)
        teacher_names = np.load(t_names_path)

        teacher_names = [
            n.decode("utf-8") if isinstance(n, bytes) else str(n)
            for n in teacher_names
        ]

        rgb_dir   = os.path.join(server_dir, "RGB")
        label_dir = os.path.join(server_dir, "LABELS")

        X, y = load_images_from_dirs(
            rgb_dir=rgb_dir,
            label_dir=label_dir,
            target_size=self.target_size,
            binary=False,
            object_classes=None,
            show_progress=show_progress,
            phase="kd",
            filelist=teacher_names
        )

        # scale / dtype
        # X = X.astype(np.float32) / 255.0
        X = X.astype(np.float32)
        y = y.astype(np.int32)
        teacher_logits = teacher_logits.astype(np.float32)

        # if use_imagenet_norm:
            # X = self._imagenet_normalize_np(X)

        self.kd_X = X
        self.kd_y = y
        self.teacher_logits = teacher_logits
        self.teacher_names = teacher_names

        print("[KDHandler] loaded:",
              "kd_X", self.kd_X.shape,
              "kd_y", self.kd_y.shape,
              "teacher_logits", self.teacher_logits.shape)

    def kd_loss_with_ce(self, y_true, s_pred, t_logits, tau=4.0, lam=0.5):
        """
        lam * CE + (1-lam) * KL
        CE는 BaseGlobalModel의 loss_sparse_ce_ignore_255를 주입받아 재사용.
        """
        if self.ce_loss_fn is None:
            raise RuntimeError("ce_loss_fn is None. Pass loss_sparse_ce_ignore_255 to KDHandler.")

        ce = self.ce_loss_fn(y_true, s_pred)

        y_true = tf.cast(y_true, tf.int32)
        mask = tf.not_equal(y_true, self.ignore_label)

        valid_count = tf.reduce_sum(tf.cast(mask, tf.int32))

        def _kl_part():
            t_soft = tf.nn.softmax(t_logits / tau, axis=-1)
            s_log  = tf.math.log(tf.maximum(s_pred, 1e-8))
            s_soft = tf.nn.softmax(s_log / tau, axis=-1)

            t_soft_m = tf.boolean_mask(t_soft, mask)
            s_soft_m = tf.boolean_mask(s_soft, mask)

            kl = tf.reduce_mean(tf.keras.losses.KLDivergence()(t_soft_m, s_soft_m))
            return kl * (tau * tau)

        kl = tf.cond(valid_count > 0, _kl_part,
                     lambda: tf.constant(0.0, tf.float32))

        return lam * ce + (1.0 - lam) * kl

    def kd_loss_pure(self, y_true, s_pred, t_logits, tau=4.0):
        """
        Pure KD: KL(teacher || student) * tau^2
        y_true는 mask(IGNORE_LABEL) 계산용으로만 사용
        """
        y_true = tf.cast(y_true, tf.int32)
        mask = tf.not_equal(y_true, self.ignore_label)  # (B,H,W)

        # teacher / student soft target (temperature scaling)
        t_soft = tf.nn.softmax(t_logits / tau, axis=-1)  # (B,H,W,C)
        s_soft = tf.nn.softmax(s_pred / tau, axis=-1)    # s_pred가 logits이면 이게 정석

        # mask로 valid 픽셀만 선택
        t_soft_m = tf.boolean_mask(t_soft, mask)  # (?,C)
        s_soft_m = tf.boolean_mask(s_soft, mask)  # (?,C)

        valid_count = tf.shape(t_soft_m)[0]

        def _kl_part():
            kl = tf.keras.losses.KLDivergence()(t_soft_m, s_soft_m)
            return kl * (tau * tau)

        kd_loss = tf.cond(
            valid_count > 0,
            _kl_part,
            lambda: tf.constant(0.0, tf.float32)
        )
        return kd_loss

    def run_epoch(self, student_model, lr=1e-4, tau=4.0, lam=0.5, batch_size=2):
        """
        student_model(TF 모델)에 대해 KD 1 epoch 수행.
        return: 평균 kd_loss
        """
        if self.kd_X is None or self.teacher_logits is None:
            raise RuntimeError("KD data not loaded. Call load_kd_data() first.")

        n = len(self.kd_X)
        total_loss = 0.0
        num_batches = 0

        for i in range(0, n, batch_size):
            xb = self.kd_X[i:i+batch_size]
            yb = self.kd_y[i:i+batch_size]
            tb = self.teacher_logits[i:i+batch_size]

            xb_t = tf.convert_to_tensor(xb, dtype=tf.float32)
            yb_t = tf.convert_to_tensor(yb, dtype=tf.int32)
            tb_t = tf.convert_to_tensor(tb, dtype=tf.float32)

            with tf.GradientTape() as tape:
                s_pred = student_model(xb_t, training=True)
                loss = self.kd_loss_with_ce(yb_t, s_pred, tb_t, tau=tau, lam=lam)

            grads = tape.gradient(loss, student_model.trainable_variables)
            self.opt.apply_gradients(zip(grads, student_model.trainable_variables))

            total_loss += float(loss.numpy())
            num_batches += 1

        return total_loss / max(1, num_batches)
