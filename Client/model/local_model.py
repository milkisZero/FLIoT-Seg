import numpy as np
from keras.models import model_from_json
import resource 
import time
import tensorflow as tf
from model.datasetLoader import load_synthia_dataset
import os
from PIL import Image

IGNORE_LABEL = 255

# ID(0~22) 순서대로 RGB
SYNTHIA_RGB = [
    (  0,   0,   0),  # 0  road           - black
    (220,  20,  60),  # 1  sidewalk       - strong red
    ( 65, 105, 225),  # 2  building       - royal blue
    (255, 165,   0),  # 3  wall           - orange
    (160,  32, 240),  # 4  fence          - purple
    ( 46, 139,  87),  # 5  pole           - sea green
    (255, 255,   0),  # 6  traffic light  - yellow
    (255, 105, 180),  # 7  traffic sign   - hot pink
    (  0, 128,   0),  # 8  vegetation     - dark green
    (154, 205,  50),  # 9  terrain        - yellow green
    (135, 206, 250),  # 10 sky            - sky blue
    (255,  69,   0),  # 11 person         - orange red
    (255, 215,   0),  # 12 rider          - gold
    (  0, 191, 255),  # 13 car            - deep sky blue
    ( 72,  61, 139),  # 14 truck          - dark slate blue
    (199,  21, 133),  # 15 bus            - medium violet red
    ( 70, 130, 180),  # 16 train          - steel blue
    (  0, 255, 255),  # 17 motorcycle     - cyan
    (127, 255, 212),  # 18 bicycle        - aquamarine
]

def _make_palette(num_classes: int) -> np.ndarray:
    base = np.asarray(SYNTHIA_RGB, dtype=np.uint8)
    return base[:num_classes]


def _overlay_mask_on_image(img_in, mask, palette, alpha=0.5):
    """
    img_in : 원본 이미지
        - 0~255 uint8 이거나
        - 0~255 float32 이거나
        - 0~1 float32 일 수 있음 (옛 코드 호환용)
    mask  : int32, shape (H,W), 예측 클래스 맵
    """
    img = np.asarray(img_in)

    # 1) dtype / 범위 정리
    if img.dtype == np.uint8:
        # 이미 0~255 uint8이면 그대로 사용
        pass
    else:
        img = img.astype(np.float32)
        max_val = img.max() if img.size > 0 else 1.0

        if max_val <= 1.5:
            # [0,1] 범위로 들어온 경우 → 0~255로 스케일
            img = (np.clip(img * 255.0, 0, 255)).astype(np.uint8)
        else:
            # 이미 0~255 근처 float 라고 가정
            img = (np.clip(img, 0, 255)).astype(np.uint8)

    # 2) 마스크 오버레이
    valid = (mask != IGNORE_LABEL)
    idx = np.clip(mask.astype(np.int32), 0, len(palette) - 1)
    color = palette[idx].astype(np.uint8)

    blended = (alpha * color + (1.0 - alpha) * img).astype(np.uint8)

    over = img.copy()
    over[valid] = blended[valid]
    return over


# -----------------------------
# ImageNet 정규화 (모델 입력용)
# -----------------------------
def imagenet_preprocess_tf(x):
    """
    x : uint8 [0,255] 또는 float32 [0,255]
    반환 : ImageNet 정규화된 float32
    """
    # 함수 내부에서 mean/std 정의 → 전역 변수 의존 X
    mean = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
    std  = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)

    x = tf.cast(x, tf.float32) / 255.0
    x = (x - mean) / std
    return x

def loss_sparse_ce_ignore_255(y_true, y_pred):
    # y_true: (B,H,W) 또는 (B,H,W,1)
    # y_pred: (B,H,W,C) [softmax]
    y_true = tf.cast(y_true, tf.int32)

    # ignore=255 마스킹
    mask = tf.not_equal(y_true, IGNORE_LABEL)          # (B,H,W)
    yt = tf.boolean_mask(y_true, mask)                 # (?,)
    yp = tf.boolean_mask(y_pred, mask)                 # (?,C)

    loss = tf.keras.losses.sparse_categorical_crossentropy(yt, yp)
    return tf.reduce_mean(loss)

@tf.function
def masked_pixel_accuracy(y_true, y_pred):
    y_true = tf.cast(y_true, tf.int32)
    y_true = tf.reshape(y_true, tf.shape(y_true)[:3])  # (B,H,W)

    mask = tf.not_equal(y_true, IGNORE_LABEL)
    pred_cls = tf.argmax(y_pred, axis=-1, output_type=tf.int32)  # (B,H,W)

    yt = tf.boolean_mask(y_true, mask)
    yp = tf.boolean_mask(pred_cls, mask)

    correct = tf.reduce_sum(tf.cast(tf.equal(yt, yp), tf.float32))
    total   = tf.cast(tf.size(yt), tf.float32)
    return tf.where(total > 0, correct / total, 0.0)


class LocalModel(object):
    def __init__(self, model_config, num_classes, selected_labels):
        self.model_config = model_config

        # 1) base 모델 로드
        base_model = model_from_json(model_config['model_json'])

        self.num_classes = num_classes
        self.selected_Labels = selected_labels
        
        # 2) 데이터 로딩 (⚠️ 여기서는 uint8 0~255로 반환되게 구현되어 있어야 함)
        datasource = load_synthia_dataset(
            binary=False,
            object_classes=selected_labels,
            target_size=model_config['input_shape']
        )
        self.x_train, self.y_train, self.x_test, self.y_test = datasource
        self.anomaly_threshold = None

        # 3) ImageNet 정규화 레이어를 앞에 붙인 래핑 모델 생성
        inputs = tf.keras.Input(shape=base_model.input_shape[1:])
        x = tf.keras.layers.Lambda(imagenet_preprocess_tf, name="imagenet_preprocess")(inputs)
        outputs = base_model(x)
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)

        self.model.compile(
            loss=loss_sparse_ce_ignore_255,
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            metrics=[masked_pixel_accuracy],
        )
        
    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, new_weights):
        self.model.set_weights(new_weights)

    # return final weights, train loss, train accuracy
    def train_one_round(self):        
        start_time = time.time()

        # self.x_train : uint8 [0,255]
        # → 모델 안에서 Lambda(imagenet_preprocess_tf)가 정규화 수행
        self.loss = self.model.fit(
            self.x_train, self.y_train,
            epochs=self.model_config.get('epoch_per_round', 1),
            batch_size=self.model_config.get('batch_size', 1),
            validation_data=(self.x_test, self.y_test),
            verbose=1
        )
   
        end_time = time.time()
        train_time = end_time - start_time

        usage = resource.getrusage(resource.RUSAGE_SELF)
        peak_memory_mb = usage.ru_maxrss / 1024

        current_loss = self.loss.history['loss'][0]
        print('One round training loss: {:.16f}'.format(current_loss))
        print('이 라운드 학습 시간: {:.4f} 초, 최고 메모리 사용량: {:.2f} MB'.format(train_time, peak_memory_mb))
            
        return self.model.get_weights(), current_loss, train_time, peak_memory_mb

    def evaluate1(self, round_number=None, save_overlays=True, num_samples=8, alpha=0.5):
        """
        세그멘테이션 평가 (ignore=255 무시)
        return:
            miou            : 전체 클래스 mIoU
            fg_miou         : exclude_from_scoring 제외한 foreground mIoU
            pixel_acc       : 픽셀 정확도
            test_loss       : CE(ignore=255) 평균 loss
        """

        n_classes = self.num_classes
        palette = _make_palette(n_classes)

        total_inter = np.zeros(n_classes, dtype=np.float64)
        total_union = np.zeros(n_classes, dtype=np.float64)
        total_correct = 0
        total_labeled = 0

        total_loss = 0.0
        num_batches = 0

        bs_eval = self.model_config['batch_size']
        for i in range(0, len(self.x_test), bs_eval):
            # 입력은 여전히 0~255 → Lambda에서 ImageNet norm
            xb = self.x_test[i:i+bs_eval]
            yb = self.y_test[i:i+bs_eval].astype(np.int32)

            pred = self.model.predict(xb, verbose=0)

            batch_loss = loss_sparse_ce_ignore_255(
                tf.convert_to_tensor(yb),
                tf.convert_to_tensor(pred)
            ).numpy()
            total_loss += batch_loss
            num_batches += 1

            pred_cls = np.argmax(pred, axis=-1).astype(np.int32)

            for y_true, y_pred in zip(yb, pred_cls):
                valid = (y_true != IGNORE_LABEL)
                y_true_v = y_true[valid]
                y_pred_v = y_pred[valid]

                total_correct += np.sum(y_true_v == y_pred_v)
                total_labeled += y_true_v.size

                for c in range(n_classes):
                    yt = (y_true_v == c)
                    yp = (y_pred_v == c)
                    inter = np.logical_and(yt, yp).sum()
                    union = np.logical_or(yt, yp).sum()
                    total_inter[c] += inter
                    total_union[c] += union

        pixel_acc = (total_correct / total_labeled) if total_labeled > 0 else 0.0
        class_iou = total_inter / np.maximum(total_union, 1e-9)

        # 전체 mIoU
        valid_iou = class_iou[np.isfinite(class_iou)]
        miou = float(np.mean(valid_iou)) if valid_iou.size > 0 else 0.0

        # foreground mIoU (exclude_from_scoring 제외)
        exclude_from_scoring = [0, 1, 2, 3, 4, 5, 8, 9, 10]

        fg_mask = np.ones(n_classes, dtype=bool)
        for c in exclude_from_scoring:
            if 0 <= c < n_classes:
                fg_mask[c] = False

        fg_valid_iou = class_iou[fg_mask]
        fg_valid_iou = fg_valid_iou[np.isfinite(fg_valid_iou)]
        fg_miou = float(np.mean(fg_valid_iou)) if fg_valid_iou.size > 0 else 0.0

        test_loss = total_loss / max(num_batches, 1)

        print("Segmentation Evaluation:")
        print(f"  Test Loss: {test_loss:.4f}")
        print(f"  PixelAcc: {pixel_acc:.4f}, mIoU: {miou:.4f}, FG-mIoU: {fg_miou:.4f}")

        if save_overlays and len(self.x_test) > 0:
            base = os.path.join("results", "image")
            tag = f"eval_round_{round_number}" if round_number is not None else "eval_final"
            out_dir = os.path.join(base, "overlays", tag)
            os.makedirs(out_dir, exist_ok=True)

            n = min(num_samples, len(self.x_test))
            idxs = np.linspace(0, len(self.x_test) - 1, num=n, dtype=int)

            xb_vis = self.x_test[idxs]          # 시각화용 원본 (uint8 0~255)
            xb_pred = xb_vis                    # 모델 입력도 그대로 (정규화는 Lambda에서)

            pred = self.model.predict(xb_pred, verbose=0)
            pred_cls = np.argmax(pred, axis=-1).astype(np.int32)

            for j, idx in enumerate(idxs):
                over = _overlay_mask_on_image(xb_vis[j], pred_cls[j], palette, alpha=alpha)
                Image.fromarray(over).save(os.path.join(out_dir, f"test_{idx}.png"))

            print(f"[Overlay] 저장: {out_dir} (샘플 {n}장)")

        return miou, fg_miou, pixel_acc, test_loss
