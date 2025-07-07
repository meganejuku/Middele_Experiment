import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
import matplotlib.pyplot as plt
import time 
import matplotlib.patches as patches
from PIL import Image  
import tifffile
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import torchvision
from torchvision.models import resnet50
from torchvision.models._utils import IntermediateLayerGetter
from torch.amp import autocast, GradScaler
from torchvision.ops import box_iou
from torchvision.models.detection.rpn import AnchorGenerator

# -------------------------------- Data --------------------------------

def parse_voc_xml(xml_path):
    if not Path(xml_path).is_file():
        return {
            "boxes":  torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.zeros((0,),    dtype=torch.int64)
        }

    root = ET.parse(xml_path).getroot()
    boxes = []
    labels = []
    for obj in root.findall("object"):
        if obj.find("name").text != "battery":
            continue
        bb = obj.find("bndbox")
        xmin = float(bb.find("xmin").text)
        ymin = float(bb.find("ymin").text)
        xmax = float(bb.find("xmax").text)
        ymax = float(bb.find("ymax").text)
        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(1)
    return {
        "boxes": torch.tensor(boxes, dtype=torch.float32),
        "labels": torch.tensor(labels, dtype=torch.int64)
    }


class HSIVOCDataset(torch.utils.data.Dataset):
    def __init__(self, root, split="val", band_mask=None):
        base = Path(root) / split

        # 画像ディレクトリ自動検出
        self.img_dir = None
        for d in ("TIFFImages", "images"):
            p = base / d
            if p.is_dir():
                self.img_dir = p
                break
        if self.img_dir is None:
            raise FileNotFoundError(f"No TIFFImages/ or images/ in {base}")

        self.ann_dir = base / "Annotations"

        # ── ① 拡張子ごとに一覧を作成 ──
        self.files = sorted(p for p in self.img_dir.iterdir()
                            if p.suffix.lower() in {".tiff", ".tif", ".jpg", ".jpeg", ".png"})
        self.band_mask = band_mask

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        img_id = path.stem

        # ── ② 拡張子で分岐してロード ──
        if path.suffix.lower() in (".tiff", ".tif"):
            img = tifffile.imread(path).astype("float32")      # (H,W,C)
            if self.band_mask is not None:
                img = img[..., self.band_mask]
            img = torch.from_numpy(img).permute(2, 0, 1) / 65535.0
            img = (img - 0.5) / 0.5
        else:  # RGB
            img = Image.open(path).convert("RGB")
            img = torch.as_tensor(np.array(img), dtype=torch.float32).permute(2,0,1) / 255.0
            img = (img - 0.5) / 0.5

        ann_path = self.ann_dir / f"{img_id}.xml"
        target   = parse_voc_xml(ann_path)

        target["image_id"] = torch.tensor([idx])
        return img, target, img_id

# ------------------------------- Model --------------------------------

import torch
import torch.nn as nn
import torchvision
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FasterRCNN, TwoMLPHead

# ── TwoMLPHead に Dropout を追加 ──────────────────────────────
class TwoMLPHeadWithDropout(TwoMLPHead):
    def __init__(self, in_channels: int, rep_size: int = 1024, p: float = 0.2):
        super().__init__(in_channels, rep_size)
        # 再定義してドロップアウトを挿入
        self.fc6 = nn.Sequential(
            nn.Linear(in_channels, rep_size), nn.ReLU(inplace=True), nn.Dropout(p)
        )
        self.fc7 = nn.Sequential(
            nn.Linear(rep_size, rep_size), nn.ReLU(inplace=True), nn.Dropout(p)
        )

# ── モデル生成関数 ──────────────────────────────────────────
def get_model(num_classes: int = 2, in_channels: int = 51, p_dropout: float = 0.2):
    # 1) バックボーン
    if in_channels == 3:
        base = torchvision.models.resnet50(weights="IMAGENET1K_V2")
    else:
        base = torchvision.models.resnet50(weights="IMAGENET1K_V2")
        base.conv1 = nn.Conv2d(in_channels, 64, 7, 2, 3, bias=False)
        nn.init.kaiming_normal_(base.conv1.weight, mode="fan_out", nonlinearity="relu")

    backbone = IntermediateLayerGetter(base, {"layer4": "0"})
    backbone.out_channels = 2048
    image_mean = [0.0] * in_channels
    image_std = [1.0] * in_channels

    # 2) ROI プール後に入るヘッドを差し替え
    representation_size = 1024
    box_head = TwoMLPHeadWithDropout(
        backbone.out_channels * 7 * 7,  # 7×7 プール後
        representation_size,
        p=p_dropout,
    )

    # 3) RPN のアンカー
    anchor_gen = AnchorGenerator(
        sizes=((32, 64, 128, 256),), aspect_ratios=((0.5, 1.0, 2.0),)
    )

    # 4) Faster-RCNN 本体
    model = FasterRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_gen,
        box_head=box_head,
        image_mean=image_mean,
        image_std=image_std,
    )
    return model


# ----------------------------- Helpers --------------------------------

def collate_fn(batch):
    imgs, targets, _ = zip(*batch)
    return list(imgs), list(targets)

def valid_one_epoch(model, loader, device):
    """
    学習を行わずに validation loss を計算する。
    ── torchvision detection 系モデルは
       ・model.training == True のとき ⇒ loss 辞書を返す
       ・model.training == False のとき ⇒ 予測を返す
       ため、ここでは **model.train()** に切り替えて
       `torch.no_grad()` で損失だけを取る。
    """
    model.train()
    loss_sum = 0.0
    with torch.no_grad():
        for imgs, targets in loader:
            imgs = [im.to(device) for im in imgs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(imgs, targets)          # ← 損失を取得
            loss = sum(loss_dict.values()).item()
            loss_sum += loss

    model.eval()  # 元に戻す
    return loss_sum / len(loader)

def train_one_epoch(model, optimizer, loader, device, scaler, accum_steps=1):
    model.train()
    epoch_loss, running_loss = 0.0, 0.0

    optimizer.zero_grad()
    for step, (imgs, targets) in enumerate(loader, 1):
        imgs = [im.to(device) for im in imgs]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with autocast(device_type=device.type):
            loss_dict = model(imgs, targets)
            loss = sum(loss_dict.values()) / accum_steps   # ★ 分割して累積
        scaler.scale(loss).backward()

        if step % accum_steps == 0 or step == len(loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        running_loss += loss.item()
        epoch_loss   += loss.item()

    return epoch_loss / max(1, len(loader))



def evaluate(model, loader, device,
             iou_thresh=0.5, score_thresh=0.3):
    model.eval()
    tp = fp = fn = 0
    with torch.no_grad():
        for imgs, targets in loader:
            imgs = [im.to(device) for im in imgs]
            preds = model(imgs)
            for pred, tgt in zip(preds, targets):
                gt = tgt["boxes"]
                keep = pred["scores"] >= score_thresh
                pr = pred["boxes"][keep].cpu()
                if len(pr) == 0:
                    fn += len(gt)
                    continue
                ious = box_iou(pr, gt)
                matched = set()
                for row in ious:
                    m, ix = row.max(0)
                    if m >= iou_thresh and ix.item() not in matched:
                        tp += 1
                        matched.add(ix.item())
                    else:
                        fp += 1
                fn += len(gt) - len(matched)
    prec = tp / (tp + fp) if tp + fp > 0 else 0.0
    rec  = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    return prec, rec, f1

# -------------------------------- CLI ---------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--save_path", required=True)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    total_t0 = time.perf_counter()
    train_losses, val_losses = [], []
    train_ds = HSIVOCDataset(args.data_root, "train")
    val_ds   = HSIVOCDataset(args.data_root, "val")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                shuffle=True, collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                                shuffle=False, collate_fn=collate_fn)
    in_ch = train_ds[0][0].shape[0]
    model = get_model(2, in_ch).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=12, gamma=0.1)
    accum_steps = 1
    scaler = GradScaler()
    for epoch in range(1, args.epochs + 1):
        torch.cuda.synchronize(device)
        ep_t0 = time.perf_counter()
        train_loss = train_one_epoch(model, optimizer, train_loader,
                                    device, scaler, accum_steps)
        val_loss   = valid_one_epoch(model,   val_loader,   device)
        prec, rec, f1 = evaluate(model, val_loader, device)
        torch.cuda.synchronize(device)
        ep_sec = time.perf_counter() - ep_t0
        print(f"Epoch {epoch}: "
              f"train Loss={train_loss:.4f}  Val Loss={val_loss:.4f}  "
              f"P={prec:.3f}  R={rec:.3f}  F1={f1:.3f}  "
              f"| Time={ep_sec:6.1f}s")
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        scheduler.step()
    # save
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    torch.cuda.synchronize(device)
    total_sec = time.perf_counter() - total_t0
    print(f"Training finished in {total_sec/60:.1f} min "
          f"({total_sec:.1f} s)")
    print(f"Model saved to {save_path}")

    # loss curve
    loss_dir = Path("runs/loss")
    loss_dir.mkdir(parents=True, exist_ok=True)
    fig_path = loss_dir / f"{save_path.stem}_loss.png"

    epochs = range(1, args.epochs + 1)
    plt.figure()
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses,   label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training / Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Loss curve saved to {fig_path}")

if __name__ == '__main__':
    main()