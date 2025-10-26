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
    (128,  64, 128),   # 0  road
    (244,  35, 232),   # 1  sidewalk
    ( 70,  70,  70),   # 2  building
    (102, 102, 156),   # 3  wall
    (190, 153, 153),   # 4  fence
    (153, 153, 153),   # 5  pole
    (250, 170,  30),   # 6  traffic light
    (220, 220,   0),   # 7  traffic sign
    (107, 142,  35),   # 8  vegetation
    (152, 251, 152),   # 9  terrain
    ( 70, 130, 180),   # 10 sky
    (220,  20,  60),   # 11 person
    (255,   0,   0),   # 12 rider
    (  0,   0, 142),   # 13 car
    (  0,   0,  70),   # 14 truck
    (  0,  60, 100),   # 15 bus
    (  0,  80, 100),   # 16 train
    (  0,   0, 230),   # 17 motorcycle
    (119,  11,  32),   # 18 bicycle
]

def _make_palette(num_classes: int) -> np.ndarray:
    base = np.asarray(SYNTHIA_RGB, dtype=np.uint8)
    return base[:num_classes]

def _overlay_mask_on_image(img01, mask, palette, alpha=0.5):
    """
    img01: float32 [0,1], shape (H,W,3)
    mask : int32,        shape (H,W), 예측 클래스 맵
    """
    # 원본을 uint8로
    img = (np.clip(img01 * 255.0, 0, 255)).astype(np.uint8)  # (H,W,3)

    # 유효 픽셀 (색칠할 위치)
    valid = (mask != IGNORE_LABEL)

    # 팔레트 인덱싱(모듈로 X, 안전하게 클립)
    idx = np.clip(mask.astype(np.int32), 0, len(palette) - 1)
    color = palette[idx].astype(np.uint8)                    # (H,W,3)

    # 반투명 블렌딩 결과
    blended = (alpha * color + (1.0 - alpha) * img).astype(np.uint8)

    # ignore 픽셀은 원본 유지, 나머지만 덮기
    over = img.copy()
    over[valid] = blended[valid]
    return over


def loss_sparse_ce_ignore_255(y_true, y_pred):
    # y_true: (B,H,W) 또는 (B,H,W,1)
    # y_pred: (B,H,W,C) [softmax]
    y_true = tf.cast(y_true, tf.int32)

    # # (B,H,W,1) -> (B,H,W), (B,H,W)는 그대로 유지
    # y_true = tf.reshape(y_true, tf.shape(y_true)[:3])

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
        self.model = model_from_json(model_config['model_json'])
        self.num_classes = num_classes
        self.selected_Labels = selected_labels
        
        datasource = load_synthia_dataset(binary=False, object_classes=selected_labels)
        self.x_train, self.y_train, self.x_test, self.y_test = datasource
        self.anomaly_threshold = None
        
    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, new_weights):
        self.model.set_weights(new_weights)

    # return final weights, train loss, train accuracy
    def train_one_round(self):        
        start_time = time.time()

        self.model.compile(
            loss=loss_sparse_ce_ignore_255,
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            metrics=[masked_pixel_accuracy],
        )

        self.loss = self.model.fit(
            self.x_train, self.y_train,
            epochs=self.model_config.get('epoch_per_round', 1),
            batch_size=self.model_config.get('batch_size', 1),
            validation_data=(self.x_test, self.y_test),
            verbose=1
        )
   
        end_time = time.time()
        train_time = end_time - start_time

        # 리소스를 이용하여 최고 메모리 사용량(킬로바이트 단위)을 측정하고 MB 단위로 변환
        usage = resource.getrusage(resource.RUSAGE_SELF)
        peak_memory_mb = usage.ru_maxrss / 1024

        current_loss = self.loss.history['loss'][0]
        print('One round training loss: {:.16f}'.format(current_loss))
        print('이 라운드 학습 시간: {:.4f} 초, 최고 메모리 사용량: {:.2f} MB'.format(train_time, peak_memory_mb))
            
        return self.model.get_weights(), current_loss, train_time, peak_memory_mb

    def evaluate1(self, round_number=None, save_overlays=True, num_samples=8, alpha=0.5):
        """
        세그멘테이션 평가 (ignore=255 무시): mIoU, PixelAcc 계산
        + (옵션) 예측 오버레이 파일 저장 (evaluate1에서만)
        반환값은 기존 인터페이스 호환을 위해 (miou, pixel_acc, 0.0) 사용
        """
        # 클래스 수/팔레트
        try:
            n_classes = int(self.model.output_shape[-1])
        except Exception:
            n_classes = int(np.max(self.y_train[self.y_train != IGNORE_LABEL])) + 1
        palette = _make_palette(n_classes)

        total_inter = np.zeros(n_classes, dtype=np.float64)
        total_union = np.zeros(n_classes, dtype=np.float64)
        total_correct = 0
        total_labeled = 0

        # 배치 추론 및 누적
        bs_eval = 4
        for i in range(0, len(self.x_test), bs_eval):
            xb = self.x_test[i:i+bs_eval].astype(np.float32)  # (B,256,256,3), [0,1]
            yb = self.y_test[i:i+bs_eval].astype(np.int32)    # (B,256,256)

            pred = self.model.predict(xb, verbose=0)          # (B,256,256,C)
            pred_cls = np.argmax(pred, axis=-1).astype(np.int32)  # (B,256,256)

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
        miou = float(np.mean(class_iou[np.isfinite(class_iou)]))

        print("Segmentation Evaluation:")
        print(f"  PixelAcc: {pixel_acc:.4f}, mIoU: {miou:.4f}")

        # === (여기서만) 오버레이 저장 ===
        if save_overlays and len(self.x_test) > 0:
            base = os.path.join("results", "image")
            tag = f"eval_round_{round_number}" if round_number is not None else "eval_final"
            out_dir = os.path.join(base, "overlays", tag)
            os.makedirs(out_dir, exist_ok=True)

            # 균등 샘플 n개 추출
            n = min(num_samples, len(self.x_test))
            idxs = np.linspace(0, len(self.x_test) - 1, num=n, dtype=int)
            xb = self.x_test[idxs].astype(np.float32)
            pred = self.model.predict(xb, verbose=0)
            pred_cls = np.argmax(pred, axis=-1).astype(np.int32)

            for j, idx in enumerate(idxs):
                over = _overlay_mask_on_image(xb[j], pred_cls[j], palette, alpha=alpha)
                Image.fromarray(over).save(os.path.join(out_dir, f"test_{idx}.png"))

            print(f"[Overlay] 저장: {out_dir} (샘플 {n}장)")

        # 인터페이스 호환 (f1, precision, recall 자리)
        return miou, pixel_acc, 0.0
