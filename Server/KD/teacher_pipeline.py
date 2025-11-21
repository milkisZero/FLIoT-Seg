# teacher_pipeline.py
import os
import sys
import glob
import json
import time
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW
import torchvision.transforms.functional as TF
from torchvision.models.segmentation import deeplabv3_resnet101
from torchvision.models.segmentation.deeplabv3 import DeepLabHead


IGNORE_LABEL = 255
FG_EXCLUDE = [0, 1, 2, 3, 4, 5, 8, 9, 10]


def load_config(config_path="../config.json"):
    with open(config_path, "r") as f:
        cfg = json.load(f)
    
    assert "model" in cfg, "config.json에 'model' 섹션이 없습니다"
    assert "num_classes" in cfg["model"], "num_classes가 정의되지 않았습니다"
    assert "input_shape" in cfg["model"], "input_shape가 정의되지 않았습니다"
    
    return cfg


# -----------------------------
# 0) DeepLabV3+ 모델 직접 정의 (VainF 레포 대체)
# -----------------------------
def deeplabv3plus_resnet101(num_classes=21, output_stride=16, pretrained_backbone=True):
    """
    torchvision의 DeepLabV3를 사용하여 간단히 구현
    완전히 동일하지는 않지만 Knowledge Distillation 목적상 충분함
    """
    model = deeplabv3_resnet101(pretrained=False, num_classes=num_classes)
    
    # output_stride 조정이 필요하면 여기서 수정
    # 기본적으로 torchvision DeepLabV3는 output_stride=8 사용
    
    return model


# -----------------------------
# 1) 서버 distill 데이터 Dataset
# -----------------------------
class ServerDistillDataset(Dataset):
    def __init__(self, server_dir, input_hw, selected_labels=None):
        self.img_dir = os.path.join(server_dir, "RGB")
        self.mask_dir = os.path.join(server_dir, "LABELS")
        self.input_hw = input_hw
        self.selected_labels = selected_labels

        img_paths = sorted(glob.glob(os.path.join(self.img_dir, "*")))
        self.pairs = []
        for ip in img_paths:
            stem = os.path.splitext(os.path.basename(ip))[0]
            mp = os.path.join(self.mask_dir, stem + ".png")
            if os.path.exists(mp):
                self.pairs.append((ip, mp))

        if len(self.pairs) == 0:
            raise RuntimeError(
                f"No image-mask pairs found.\n"
                f"Image dir: {self.img_dir}\n"
                f"Mask dir: {self.mask_dir}"
            )

        print(f"Found {len(self.pairs)} image-mask pairs")

        self.mean = (0.485, 0.456, 0.406)
        self.std  = (0.229, 0.224, 0.225)

    def __len__(self):
        return len(self.pairs)

    def _remap_mask(self, mask_np):
        if self.selected_labels is None:
            return mask_np
        out = np.ones_like(mask_np, dtype=np.int32) * IGNORE_LABEL
        for new_id, old_id in enumerate(self.selected_labels):
            out[mask_np == old_id] = new_id
        return out

    def __getitem__(self, idx):
        ip, mp = self.pairs[idx]

        img = Image.open(ip).convert("RGB")
        mask = Image.open(mp).convert("L")

        H, W = self.input_hw
        img = img.resize((W, H), Image.BILINEAR)
        mask = mask.resize((W, H), Image.NEAREST)

        img_t = TF.to_tensor(img)
        img_t = TF.normalize(img_t, self.mean, self.std)

        mask_np = np.array(mask, dtype=np.int32)
        mask_np = self._remap_mask(mask_np)
        mask_t = torch.from_numpy(mask_np).long()

        return img_t, mask_t, os.path.basename(ip)


# -----------------------------
# 2) teacher 생성 + Cityscapes pretrained 로드
# -----------------------------
def build_teacher(num_classes, output_stride=16):
    """Teacher 모델 생성"""
    model = deeplabv3plus_resnet101(num_classes=num_classes)
    print(f"Built DeepLabV3+ ResNet101 (num_classes={num_classes})")
    return model


def load_cityscapes_pretrained(model, ckpt_path):
    """Cityscapes pretrained 가중치 로드"""
    if not os.path.exists(ckpt_path):
        print(f"⚠ Checkpoint not found: {ckpt_path}")
        print("  Continuing without pretrained weights...")
        return
    
    print(f"Loading pretrained weights from: {ckpt_path}")
    
    # PyTorch 2.6+ 호환성: weights_only=False 추가
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # 다양한 저장 포맷 처리
    state = None
    for key in ["model_state", "state_dict", "model", "net"]:
        if isinstance(ckpt, dict) and key in ckpt:
            state = ckpt[key]
            print(f"Found weights under key: '{key}'")
            break
    
    if state is None:
        state = ckpt
        print("Loading weights directly")

    model_state = model.state_dict()
    filtered = {}
    
    for k, v in state.items():
        # 키 이름 매칭
        target_key = k
        
        # VainF 형식 → torchvision 형식 변환
        if k.startswith("backbone."):
            target_key = k.replace("backbone.", "backbone.")
        elif k.startswith("classifier."):
            target_key = k.replace("classifier.", "classifier.")
            
        if target_key in model_state:
            if hasattr(v, "shape") and model_state[target_key].shape == v.shape:
                filtered[target_key] = v

    if len(filtered) > 0:
        model_state.update(filtered)
        model.load_state_dict(model_state, strict=False)
        print(f"✓ Loaded {len(filtered)}/{len(model_state)} weights")
    else:
        print("⚠ No matching weights found, using random initialization")


# -----------------------------
# 2.5) FG-mIoU 계산 유틸
# -----------------------------
def _confusion_matrix_from_logits(logits, masks, num_classes):
    with torch.no_grad():
        # torchvision DeepLabV3는 {'out': logits} 형태로 반환
        if isinstance(logits, dict):
            logits = logits['out']
            
        pred = torch.argmax(logits, dim=1)

        valid = (masks != IGNORE_LABEL)
        y_true = masks[valid].view(-1)
        y_pred = pred[valid].view(-1)

        if y_true.numel() == 0:
            return torch.zeros((num_classes, num_classes),
                               dtype=torch.int64, device=logits.device)

        k = num_classes * y_true + y_pred
        conf = torch.bincount(k, minlength=num_classes * num_classes)
        conf = conf.reshape(num_classes, num_classes).to(torch.int64)
        return conf


def _fg_miou_from_conf(conf, exclude_classes):
    conf = conf.to(torch.float64)
    inter = torch.diag(conf)
    union = conf.sum(0) + conf.sum(1) - inter
    iou = inter / torch.clamp(union, min=1.0)

    fg_mask = torch.ones(conf.shape[0], dtype=torch.bool, device=conf.device)
    for c in exclude_classes:
        if 0 <= c < conf.shape[0]:
            fg_mask[c] = False

    fg_iou = iou[fg_mask]
    fg_iou = fg_iou[torch.isfinite(fg_iou)]
    if fg_iou.numel() == 0:
        return 0.0
    return float(fg_iou.mean().item())


# -----------------------------
# 3) Fine-tune (나머지 코드는 동일, modeling 파라미터만 제거)
# -----------------------------
def finetune_teacher(
    server_dir,
    pretrained_ckpt,
    save_dir,
    num_classes,
    input_hw,
    batch_size,
    selected_labels=None,
    epochs_head=5,
    epochs_full=20,
    lr_head=1e-4,
    lr_full=5e-6,
    weight_decay=1e-4,
    patience_limit=5,
    device="cuda",
    output_stride=16
):
    """Teacher 모델 Fine-tuning"""
    print("\n" + "="*60)
    print("Starting Teacher Fine-tuning")
    print("="*60)
    
    os.makedirs(save_dir, exist_ok=True)

    ds = ServerDistillDataset(server_dir, input_hw=input_hw, selected_labels=selected_labels)
    n_val = max(1, int(len(ds) * 0.1))
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val])
    
    print(f"Train samples: {n_train}, Val samples: {n_val}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, drop_last=True, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=True)

    teacher = build_teacher(num_classes, output_stride=output_stride)
    load_cityscapes_pretrained(teacher, pretrained_ckpt)
    
    if device == "cuda" and not torch.cuda.is_available():
        print("⚠ CUDA not available, using CPU")
        device = "cpu"
    teacher.to(device)
    print(f"Using device: {device}")

    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_LABEL)

    best_fg = -1.0
    best_path = os.path.join(save_dir, f"teacher_best_{input_hw[0]}x{input_hw[1]}.pth")

    train_log = []

    def run_epoch(loader, train=True):
        teacher.train(train)
        total_loss = 0.0
        conf_total = None
        
        desc = "Training" if train else "Validation"
        num_batches = len(loader)

        for batch_idx, (imgs, masks, _) in enumerate(loader):
            imgs, masks = imgs.to(device), masks.to(device)
            output = teacher(imgs)
            
            # torchvision DeepLabV3는 {'out': logits} 반환
            if isinstance(output, dict):
                logits = output['out']
            else:
                logits = output
                
            loss = criterion(logits, masks)

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            else:
                conf = _confusion_matrix_from_logits(logits, masks, num_classes)
                conf_total = conf if conf_total is None else (conf_total + conf)

            total_loss += loss.item()
            
            if (batch_idx + 1) % 10 == 0:
                print(f"{desc} [{batch_idx+1}/{num_batches}] loss: {loss.item():.4f}")

        avg_loss = total_loss / max(1, len(loader))

        if train:
            return avg_loss, None
        fg_miou = _fg_miou_from_conf(conf_total, FG_EXCLUDE)
        return avg_loss, fg_miou

    # Stage 1: head warmup
    print("\n" + "-"*60)
    print("Stage 1: Head Warmup (Backbone Frozen)")
    print("-"*60)
    
    for name, p in teacher.named_parameters():
        if "backbone" in name:
            p.requires_grad = False

    optimizer = AdamW(filter(lambda p: p.requires_grad, teacher.parameters()),
                      lr=lr_head, weight_decay=weight_decay)

    patience = 0
    for ep in range(epochs_head):
        print(f"\nEpoch {ep+1}/{epochs_head}")
        tr_loss, _ = run_epoch(train_loader, train=True)
        vl_loss, fg_miou = run_epoch(val_loader, train=False)

        improved = fg_miou > best_fg
        if improved:
            best_fg = fg_miou
            patience = 0
            torch.save({"model_state": teacher.state_dict()}, best_path)
            print(f"✓ New best model saved! (FG-mIoU: {fg_miou:.4f})")
        else:
            patience += 1

        log_row = {
            "stage": "head",
            "epoch": ep + 1,
            "train_loss": float(tr_loss),
            "val_loss": float(vl_loss),
            "val_fg_miou": float(fg_miou),
            "best_fg_miou": float(best_fg),
            "improved": bool(improved),
            "patience": patience,
            "lr": float(lr_head),
            "time": time.time()
        }
        train_log.append(log_row)

        print(f"[Head FT] train_loss: {tr_loss:.4f} | val_loss: {vl_loss:.4f} | "
              f"val_fg_miou: {fg_miou:.4f} | best: {best_fg:.4f} | patience: {patience}/{patience_limit}")

        if patience >= patience_limit:
            print("⚠ Early stopping (no improvement)")
            break

    # Stage 2: full finetune
    print("\n" + "-"*60)
    print("Stage 2: Full Fine-tuning")
    print("-"*60)
    
    for p in teacher.parameters():
        p.requires_grad = True

    optimizer = AdamW(teacher.parameters(), lr=lr_full, weight_decay=weight_decay)

    patience = 0
    for ep in range(epochs_full):
        print(f"\nEpoch {ep+1}/{epochs_full}")
        tr_loss, _ = run_epoch(train_loader, train=True)
        vl_loss, fg_miou = run_epoch(val_loader, train=False)

        improved = fg_miou > best_fg
        if improved:
            best_fg = fg_miou
            patience = 0
            torch.save({"model_state": teacher.state_dict()}, best_path)
            print(f"✓ New best model saved! (FG-mIoU: {fg_miou:.4f})")
        else:
            patience += 1

        log_row = {
            "stage": "full",
            "epoch": ep + 1,
            "train_loss": float(tr_loss),
            "val_loss": float(vl_loss),
            "val_fg_miou": float(fg_miou),
            "best_fg_miou": float(best_fg),
            "improved": bool(improved),
            "patience": patience,
            "lr": float(lr_full),
            "time": time.time()
        }
        train_log.append(log_row)

        print(f"[Full FT] train_loss: {tr_loss:.4f} | val_loss: {vl_loss:.4f} | "
              f"val_fg_miou: {fg_miou:.4f} | best: {best_fg:.4f} | patience: {patience}/{patience_limit}")

        if patience >= patience_limit:
            print("⚠ Early stopping (no improvement)")
            break

    final_path = os.path.join(save_dir, f"teacher_final_{input_hw[0]}x{input_hw[1]}.pth")
    torch.save({"model_state": teacher.state_dict()}, final_path)

    log_path = os.path.join(save_dir, "teacher_finetune_log.json")
    with open(log_path, "w") as f:
        json.dump(train_log, f, indent=2)

    print("\n" + "="*60)
    print("Fine-tuning Complete!")
    print(f"Best FG-mIoU: {best_fg:.4f}")
    print(f"✓ Saved best: {best_path}")
    print(f"✓ Saved final: {final_path}")
    print(f"✓ Saved log: {log_path}")

    return best_path, final_path, log_path


# -----------------------------
# 4) teacher logits 저장
# -----------------------------
def dump_teacher_logits(
    teacher_ckpt,
    server_dir,
    out_dir,
    num_classes,
    input_hw,
    batch_size,
    selected_labels=None,
    device="cuda",
    output_stride=16
):
    """Teacher logits를 numpy로 저장"""
    print("\n" + "="*60)
    print("Dumping Teacher Logits")
    print("="*60)
    
    os.makedirs(out_dir, exist_ok=True)

    ds = ServerDistillDataset(server_dir, input_hw=input_hw, selected_labels=selected_labels)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, 
                       num_workers=0, pin_memory=True)

    teacher = build_teacher(num_classes, output_stride=output_stride)
    ckpt = torch.load(teacher_ckpt, map_location="cpu")
    teacher.load_state_dict(ckpt["model_state"], strict=True)

    if device == "cuda" and not torch.cuda.is_available():
        print("⚠ CUDA not available, using CPU")
        device = "cpu"
    
    teacher.to(device)
    teacher.eval()
    print(f"Using device: {device}")

    all_logits, all_names = [], []
    num_batches = len(loader)

    with torch.no_grad():
        for batch_idx, (imgs, _, names) in enumerate(loader):
            imgs = imgs.to(device)
            output = teacher(imgs)
            
            # torchvision DeepLabV3는 {'out': logits} 반환
            if isinstance(output, dict):
                logits = output['out']
            else:
                logits = output
                
            logits = logits.permute(0,2,3,1)  # (B,H,W,C)
            all_logits.append(logits.cpu().numpy().astype(np.float16))
            all_names.extend(list(names))
            
            if (batch_idx + 1) % 10 == 0:
                print(f"Processed [{batch_idx+1}/{num_batches}] batches")

    all_logits = np.concatenate(all_logits, axis=0)

    logits_path = os.path.join(out_dir, "teacher_logits.npy")
    names_path = os.path.join(out_dir, "teacher_filenames.npy")
    
    np.save(logits_path, all_logits)
    np.save(names_path, np.array(all_names))

    print(f"✓ Saved logits: {logits_path} (shape: {all_logits.shape})")
    print(f"✓ Saved filenames: {names_path} ({len(all_names)} files)")


# -----------------------------
# 5) main
# -----------------------------
if __name__ == "__main__":
    print("="*60)
    print("Teacher Pipeline - Knowledge Distillation Preparation")
    print("="*60)
    
    cfg = load_config("../config.json")

    model_cfg = cfg["model"]
    num_classes = model_cfg["num_classes"]
    selected_labels = model_cfg.get("selected_labels", None)

    H, W, _ = model_cfg["input_shape"]
    input_hw = (H, W)
    # batch_size = model_cfg.get("batch_size", 2)
    batch_size = 8

    print(f"\nConfiguration:")
    print(f"  num_classes: {num_classes}")
    print(f"  input_shape: {input_hw}")
    print(f"  batch_size: {batch_size}")

    server_dir = "../SYNTHIA_Splitted/serverdata"
    city_ckpt  = "./best_deeplabv3plus_resnet101_cityscapes_os16.pth.tar"
    save_dir   = "./teacher_ft"
    out_dir    = "./teacher_logits"

    if not os.path.exists(server_dir):
        raise FileNotFoundError(f"Server data directory not found: {server_dir}")

    best_ckpt, final_ckpt, log_path = finetune_teacher(
        server_dir=server_dir,
        pretrained_ckpt=city_ckpt,
        save_dir=save_dir,
        num_classes=num_classes,
        input_hw=input_hw,
        batch_size=batch_size,
        selected_labels=selected_labels,
        epochs_head=5,
        epochs_full=20,
        patience_limit=5,
        device="cuda"
    )

    dump_teacher_logits(
        teacher_ckpt=best_ckpt,
        server_dir=server_dir,
        out_dir=out_dir,
        num_classes=num_classes,
        input_hw=input_hw,
        batch_size=batch_size,
        selected_labels=selected_labels,
        device="cuda"
    )
    
    print("\n" + "="*60)
    print("All Done!")
    print("="*60)