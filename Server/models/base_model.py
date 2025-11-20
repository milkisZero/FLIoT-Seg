import numpy as np
import time
import json
from utils.pickle_utils import obj_to_pickle_string, pickle_string_to_obj
import numpy as np
from keras.models import model_from_json
import resource 
import time
import tensorflow as tf
from models.datasetLoader import load_synthia_dataset
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

class BaseGlobalModel(object):
    """docstring for GlobalModel"""
    def __init__(self, selected_labels, target_size):
        self.model = self.build_model()
        self.current_weights = self.model.get_weights()
        
        # for convergence check
        self.prev_train_loss = None

        self.train_losses = []
        self.valid_losses = []
        self.train_accuracies = []
        self.valid_accuracies = []
        self.training_start_time = int(round(time.time())) 
    
        self.x_test = None
        self.y_test = None
        self.selected_labels=selected_labels
        self.target_size=target_size
    
    def load_dataset(self):
        self.x_test, self.y_test = load_synthia_dataset(
            target_size=self.target_size,
            binary=False,
            object_classes= self.selected_labels
        )
        
    @staticmethod
    def fedavg(client_weights, client_sizes, current_weights):
        """
        Federated Averaging (FedAvg)
        w_global = Σ(n_k / n_total) * w_k
        """
        new_weights = [np.zeros_like(w, dtype=np.float32) for w in current_weights]
        total_size = np.sum(client_sizes)
        
        for c in range(len(client_weights)):
            # pickle string 역직렬화
            if isinstance(client_weights[c], str):
                client_weight_obj = pickle_string_to_obj(client_weights[c])
            else:
                client_weight_obj = client_weights[c]
            
            # 가중치 비율 계산
            weight_factor = client_sizes[c] / total_size
            
            # 각 레이어별 가중 평균
            for i in range(len(new_weights)):
                w_client = client_weight_obj[i]
                
                # numpy array 변환
                if not isinstance(w_client, np.ndarray):
                    w_client = np.array(w_client, dtype=np.float32)
                elif w_client.dtype != np.float32:
                    w_client = w_client.astype(np.float32)
                
                # 가중 평균 누적
                new_weights[i] += w_client * weight_factor
        return new_weights
        
    def update_weights(self, client_weights, client_sizes):
        self.current_weights = self.fedavg(
            client_weights,
            client_sizes,
            self.current_weights
        )
        self.model.set_weights(self.current_weights)

    def aggregate_loss_accuracy(self, client_losses, client_sizes):
        total_size = np.sum(client_sizes)
        print("-----client sizes-------", client_sizes)
        print('\033[1;35;0m client_losses \033[0m', client_losses)
        aggr_loss = np.sum(client_losses[i] / total_size * client_sizes[i]
                           for i in range(len(client_sizes)))
        return aggr_loss

    def aggregate_train_loss_accuracy(self, client_losses, client_sizes, cur_round):
        cur_time = int(round(time.time())) - self.training_start_time
        aggr_loss = self.aggregate_loss_accuracy(client_losses, client_sizes)
        self.train_losses.append([cur_round, cur_time, aggr_loss])
        with open('stats.txt', 'w') as outfile:
            json.dump(self.get_stats(), outfile)
        return aggr_loss

    def get_stats(self):
        return {
            "train_loss": self.train_losses,
            "valid_loss": self.valid_losses,
            "train_accuracy": self.train_accuracies,
            "valid_accuracy": self.valid_accuracies
        }
        
    def evaluate_global(self, round_number=None, save_overlays=True, num_samples=8, alpha=0.5):
        """
        세그멘테이션 평가 (ignore=255 무시)
        """
        # 클래스 수/팔레트
        n_classes = len(self.selected_labels)
        palette = _make_palette(n_classes)

        total_inter = np.zeros(n_classes, dtype=np.float64)
        total_union = np.zeros(n_classes, dtype=np.float64)
        total_correct = 0
        total_labeled = 0

        total_loss = 0.0
        num_batches = 0

        bs_eval = 4
        for i in range(0, len(self.x_test), bs_eval):
            xb = self.x_test[i:i+bs_eval].astype(np.float32)
            yb = self.y_test[i:i+bs_eval].astype(np.int32)

            pred = self.model.predict(xb, verbose=0)

            # ----- 여기서 batch loss 계산 -----
            # yb: (B,H,W), pred: (B,H,W,C)
            batch_loss = loss_sparse_ce_ignore_255(
                tf.convert_to_tensor(yb),
                tf.convert_to_tensor(pred)
            ).numpy()
            total_loss += batch_loss
            num_batches += 1
            # ---------------------------------

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
        miou = float(np.mean(class_iou[np.isfinite(class_iou)]))

        test_loss = total_loss / max(num_batches, 1)

        print("Segmentation Evaluation:")
        print(f"  Test Loss: {test_loss:.4f}")
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
        return miou, pixel_acc, test_loss