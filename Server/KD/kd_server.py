# kd_server.py
import os
import json
import numpy as np
import tensorflow as tf

from models.base_model import IGNORE_LABEL, loss_sparse_ce_ignore_255
from models.datasetLoader import load_images_from_dirs


def load_config(config_path="./config.json"):
    with open(config_path, "r") as f:
        return json.load(f)


def load_teacher_logits(logits_dir):
    logits = np.load(os.path.join(logits_dir, "teacher_logits.npy"))
    names  = np.load(os.path.join(logits_dir, "teacher_filenames.npy"))
    names  = [n.decode("utf-8") if isinstance(n, bytes) else str(n) for n in names]
    return logits, names


def imagenet_normalize(x):
    """
    x: float32 [0,1]
    PyTorch teacher에서 쓴 mean/std와 동일하게 맞춤
    """
    mean = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
    std  = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)
    return (x - mean) / std


def make_kd_dataset(
    server_dir,
    logits_dir,
    target_size,
    batch_size=2,
    show_progress=False,
    use_imagenet_norm=True,  # teacher logits 전처리와 맞출지 여부
):
    t_logits, names = load_teacher_logits(logits_dir)

    rgb_dir = os.path.join(server_dir, "RGB")
    label_dir = os.path.join(server_dir, "LABELS")

    X, y = load_images_from_dirs(
        rgb_dir=rgb_dir,
        label_dir=label_dir,
        target_size=target_size,
        binary=False,
        object_classes=None,
        show_progress=show_progress,
        phase="kd",
        filelist=names
    )

    X = X.astype(np.float32) / 255.0
    y = y.astype(np.int32)
    t_logits = t_logits.astype(np.float32)

    ds = tf.data.Dataset.from_tensor_slices((X, y, t_logits))

    def _prep(xb, yb, tb):
        xb = tf.cast(xb, tf.float32)
        if use_imagenet_norm:
            xb = imagenet_normalize(xb)
        yb = tf.cast(yb, tf.int32)
        tb = tf.cast(tb, tf.float32)
        return xb, yb, tb

    ds = ds.map(_prep, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def kd_loss_with_ce(y_true, s_pred, t_logits, tau=4.0, lam=0.5):
    ce = loss_sparse_ce_ignore_255(y_true, s_pred)

    y_true = tf.cast(y_true, tf.int32)
    mask = tf.not_equal(y_true, IGNORE_LABEL)

    # 유효 픽셀이 0개면 ce만 반환(kl=0)
    valid_count = tf.reduce_sum(tf.cast(mask, tf.int32))

    def _kl_part():
        t_soft = tf.nn.softmax(t_logits / tau, axis=-1)

        s_log  = tf.math.log(tf.maximum(s_pred, 1e-8))
        s_soft = tf.nn.softmax(s_log / tau, axis=-1)

        t_soft_m = tf.boolean_mask(t_soft, mask)
        s_soft_m = tf.boolean_mask(s_soft, mask)

        kl = tf.reduce_mean(tf.keras.losses.KLDivergence()(t_soft_m, s_soft_m))
        return kl * (tau * tau)

    kl = tf.cond(valid_count > 0, _kl_part, lambda: tf.constant(0.0, tf.float32))

    return lam * ce + (1.0 - lam) * kl


@tf.function
def kd_train_step(student_model, optimizer, xb, yb, tlogit, tau=4.0, lam=0.5):
    with tf.GradientTape() as tape:
        s_pred = student_model(xb, training=True)
        loss = kd_loss_with_ce(yb, s_pred, tlogit, tau=tau, lam=lam)

    grads = tape.gradient(loss, student_model.trainable_variables)
    optimizer.apply_gradients(zip(grads, student_model.trainable_variables))
    return loss


def run_server_kd_epoch(student_model, kd_ds, lr=1e-4, tau=4.0, lam=0.5):
    opt = tf.keras.optimizers.Adam(lr)
    losses = []
    for xb, yb, tlogit in kd_ds:
        l = kd_train_step(student_model, opt, xb, yb, tlogit, tau=tau, lam=lam)
        losses.append(float(l))
    return sum(losses) / max(1, len(losses))


if __name__ == "__main__":
    cfg = load_config("./config.json")
    model_cfg = cfg["model"]

    target_size = tuple(model_cfg["input_shape"])
    batch_size = model_cfg.get("batch_size", 2)

    kd_ds = make_kd_dataset(
        server_dir="../SYNTHIA_Splitted/serverdata",
        logits_dir="./teacher_logits",
        target_size=target_size,
        batch_size=batch_size,
        use_imagenet_norm=True  # teacher 전처리와 맞춤
    )
