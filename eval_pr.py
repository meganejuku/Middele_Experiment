#!/usr/bin/env python3
"""
Precision–Recall & Confusion‑Matrix evaluation for object‑detection models
===========================================================================
Evaluates an object‑detection model (e.g. HSI‑adapted Faster R‑CNN) on a
Pascal‑VOC‑style dataset and **visualises three key metrics at once**:

1. **COCO‑style mAP** via `torchmetrics`
2. **Confusion matrix** that correctly counts **TP / FP / FN**
3. **Precision–Recall curve** with the **best‑F1 point** highlighted

Background (TN) pixels are effectively infinite and therefore omitted from the
confusion matrix.  The matrix is 2 × 2 with the TN cell fixed to 0.

Usage example
-------------
```bash
python eval_pr.py \
    --data_root   dataset/TIFF_battery \
    --weights     runs/hsi_rcnn.pth \
    --split       val \
    --output_dir  runs/eval_val \
    --score_thresh 0.05 \
    --iou_thresh   0.5
```

Replace the *project‑specific* imports (`HSIVOCDataset`, `get_model`) with your
own, if they live elsewhere.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import precision_recall_curve
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.ops import box_iou

# ─────── Project‑specific modules ──────────────────────────────────────────
from hsi_faster_rcnn import HSIVOCDataset  # dataset class
from hsi_faster_rcnn import get_model      # model factory
# ───────────────────────────────────────────────────────────────────────────

##############################################################################
# Helper functions                                                            #
##############################################################################

def collate_fn(batch):
    """Keep only *images* and *targets* from each dataset item."""
    images  = [b[0] for b in batch]
    targets = [b[1] for b in batch]
    return images, targets


@torch.no_grad()
def collect_results(model, loader, device):
    """Inference loop – returns lists compatible with torchmetrics."""
    model.eval()
    preds, tgts = [], []
    for imgs, tgs in loader:
        imgs_gpu = [im.to(device) for im in imgs]
        tgs_gpu  = [{k: v.to(device) for k, v in t.items()} for t in tgs]
        outs = model(imgs_gpu)

        for o in outs:
            preds.append({
                "boxes" : o["boxes" ].cpu(),
                "scores": o["scores"].cpu(),
                "labels": o["labels"].cpu(),
            })
        for t in tgs_gpu:
            tgts.append({
                "boxes" : t["boxes"].cpu(),
                "labels": t["labels"].cpu(),
            })
    return preds, tgts


def confusion_counts(preds, targets, *, score_th: float, iou_th: float):
    """Return **tp, fp, fn** across the dataset."""
    cnt = Counter(tp=0, fp=0, fn=0)
    for pred, tgt in zip(preds, targets):
        keep    = pred["scores"] >= score_th
        p_boxes = pred["boxes"][keep]
        g_boxes = tgt["boxes"]
        matched = torch.zeros(len(g_boxes), dtype=torch.bool)

        if len(p_boxes) and len(g_boxes):
            iou = box_iou(p_boxes, g_boxes)
            for i in range(len(p_boxes)):
                best_iou, idx = iou[i].max(0)
                if best_iou >= iou_th and not matched[idx]:
                    cnt["tp"] += 1
                    matched[idx] = True
                else:
                    cnt["fp"] += 1
        else:
            cnt["fp"] += len(p_boxes)

        cnt["fn"] += (~matched).sum().item()
    return cnt["tp"], cnt["fp"], cnt["fn"]

##############################################################################
# Main                                                                        #
##############################################################################

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",    required=True)
    parser.add_argument("--weights",      required=True)
    parser.add_argument("--split",        default="val", choices=["train", "val", "test"])
    parser.add_argument("--output_dir",   default="runs/eval")
    parser.add_argument("--score_thresh", type=float, default=0.5)
    parser.add_argument("--iou_thresh",   type=float, default=0.5)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset & loader
    ds = HSIVOCDataset(args.data_root, split=args.split)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=4,
                        collate_fn=collate_fn)

    in_ch = ds[0][0].shape[0]
    model = get_model(num_classes=2, in_channels=in_ch).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))

    # Inference
    preds, targets = collect_results(model, loader, device)

    # mAP
    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")
    metric.update(preds, targets)
    res = metric.compute()
    print(f"[{args.split}] mAP@[0.50:0.95]={res['map']:.3f} | mAP@0.50={res['map_50']:.3f}")

    # Confusion matrix
    tp, fp, fn = confusion_counts(preds, targets,
                                  score_th=args.score_thresh,
                                  iou_th=args.iou_thresh)
    cm = np.array([[0, fp], [fn, tp]])
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Background", "Battery"],
                yticklabels=["Background", "Battery"])
    plt.xlabel("Predicted class")
    plt.ylabel("Actual class")
    plt.title("Battery vs. Background – Confusion Matrix")
    cm_path = out_dir / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix saved → {cm_path}")

    # PR curve
    y_scores, y_true = [], []
    for pred, tgt in zip(preds, targets):
        keep    = pred["scores"] >= args.score_thresh
        p_boxes = pred["boxes"][keep]
        p_scores = pred["scores"][keep]
        g_boxes = tgt["boxes"]
        matched = torch.zeros(len(g_boxes), dtype=torch.bool)

        if len(p_boxes) and len(g_boxes):
            iou = box_iou(p_boxes, g_boxes)
            for i in range(len(p_boxes)):
                best_iou, idx = iou[i].max(0)
                is_tp = best_iou >= args.iou_thresh and not matched[idx]
                y_true.append(int(is_tp))
                matched[idx] = matched[idx] or is_tp
                y_scores.append(p_scores[i].item())
        else:
            y_true.extend([0] * len(p_boxes))
            y_scores.extend(p_scores.tolist())

    if y_scores:
        prec, rec, thr = precision_recall_curve(y_true, y_scores)
        f1 = 2 * prec * rec / (prec + rec + 1e-6)
        best_idx = int(np.nanargmax(f1))
        best_f1  = f1[best_idx]
        best_thr = thr[best_idx] if best_idx < len(thr) else 1.0

        plt.figure()
        plt.plot(rec, prec, label=f"best F1={best_f1:.3f} @ thr={best_thr:.2f}")
        plt.xlabel("Recall (Battery)")
        plt.ylabel("Precision (Battery)")
        plt.title(f"PR Curve – {args.split}")
        plt.grid(True)
        plt.legend()
        pr_path = out_dir / "pr_curve.png"
        plt.savefig(pr_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"PR curve saved → {pr_path}")
    else:
        print("No predictions above score_thresh – PR curve skipped.")


if __name__ == "__main__":
    main()
