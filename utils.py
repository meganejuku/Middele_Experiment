import xml.etree.ElementTree as ET
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import tifffile
import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchvision.models import resnet50
from torchvision.models._utils import IntermediateLayerGetter
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
        exts = ("*.tiff", "*.tif", "*.jpg", "*.jpeg", "*.png")
        self.files = []
        for pat in exts:
            self.files.extend(sorted(self.img_dir.glob(pat)))

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

def get_model(num_classes=2, in_channels=51):
    if in_channels == 3:
        # ── RGB: ImageNet の ResNet50 をそのまま ──
        backbone = resnet50(weights="IMAGENET1K_V2")
        backbone = IntermediateLayerGetter(backbone, {"layer4": "0"})
        backbone.out_channels = 2048
        image_mean = [0.0] * 3
        image_std  = [1.0] * 3
    else:
        # ── HSI: Conv1 を差し替え ──
        base = resnet50(weights="IMAGENET1K_V2")
        base.conv1 = nn.Conv2d(in_channels, 64, 7, 2, 3, bias=False)
        nn.init.kaiming_normal_(base.conv1.weight, mode="fan_out", nonlinearity="relu")
        backbone = IntermediateLayerGetter(base, {"layer4": "0"})
        backbone.out_channels = 2048
        image_mean = [0.0] * in_channels
        image_std  = [1.0] * in_channels

    anchor_gen = AnchorGenerator(
        sizes=((32, 64, 128, 256),),
        aspect_ratios=((0.5, 1.0, 2.0),)
    )

    model = torchvision.models.detection.FasterRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_gen,
        image_mean=image_mean,
        image_std=image_std
    )
    return model

# ----------------------------- Helpers --------------------------------

def collate_fn(batch):
    imgs, targets, _ = zip(*batch)
    return list(imgs), list(targets)

# --------------------------- Visualization -----------------------------

def visualize(img_tensor, boxes, scores, save_path):
    arr = img_tensor[:3].permute(1, 2, 0).numpy()
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)
    fig, ax = plt.subplots(1)
    ax.imshow(arr)
    for box, sc in zip(boxes, scores):
        x1, y1, x2, y2 = box
        ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                      fill=False, edgecolor="lime", linewidth=2))
        ax.text(x1, y1, f"{sc:.2f}", color="yellow", fontsize=8,
                bbox=dict(facecolor="black", alpha=0.3))
    ax.axis("off")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
