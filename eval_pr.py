#!/usr/bin/env python3
"""
Precision–Recall & Confusion‑Matrix evaluation for object‑detection models
===========================================================================
Adds configurable font sizes for PR curve and confusion matrix.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import precision_recall_curve
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.ops import box_iou
import pandas as pd
import seaborn as sns

# ─────── Project‑specific modules ──────────────────────────────────────────
from hsi_faster_rcnn import HSIVOCDataset  # dataset class
from hsi_faster_rcnn import get_model      # model factory
# ───────────────────────────────────────────────────────────────────────────

def collate_fn(batch):
    images  = [b[0] for b in batch]
    targets = [b[1] for b in batch]
    return images, targets

@torch.no_grad()
def collect_results(model, loader, device):
    model.eval()
    preds, tgts = [], []
    for imgs, tgs in loader:
        imgs_gpu = [im.to(device) for im in imgs]
        tgs_gpu  = [{k: v.to(device) for k, v in t.items()} for t in tgs]
        outs = model(imgs_gpu)
        for o in outs:
            preds.append({"boxes": o["boxes"].cpu(),
                          "scores": o["scores"].cpu(),
                          "labels": o["labels"].cpu()})
        for t in tgs_gpu:
            tgts.append({"boxes": t["boxes"].cpu(),
                         "labels": t["labels"].cpu()})
    return preds, tgts


def confusion_counts(preds, targets, *, score_th: float, iou_th: float):
    cnt = Counter(tp=0, fp=0, fn=0)
    for pred, tgt in zip(preds, targets):
        keep = pred["scores"] >= score_th
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--split", default="val", choices=["train","val","test"])
    parser.add_argument("--output_dir", default="runs/eval")
    parser.add_argument("--score_thresh", type=float, default=0.5)
    parser.add_argument("--iou_thresh", type=float, default=0.5)
    parser.add_argument("--font_size", type=float, default=12,
                        help="Base font size for labels, titles, ticks, and annotations")
    args = parser.parse_args()

    # Set global font size
    plt.rcParams.update({'font.size': args.font_size})

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
    labels = ["Battery", "Background"]
    cm_counts = np.array([[tp, fn], [fp, 0]])
    cm = cm_counts.astype(float) / cm_counts.sum()

    df_cm = pd.DataFrame(cm, index=labels, columns=labels)
    annot = (((df_cm * 1000).apply(np.floor)).round(1) / 1000).astype(str)
    annot.loc["Background","Background"] = ""

    plt.figure(figsize=(4,3))
    sns.heatmap(df_cm, annot=annot, fmt="", cmap="Blues",
                xticklabels=labels, yticklabels=labels,
                annot_kws={"fontsize": args.font_size})
    plt.xlabel("Predicted class", fontsize=args.font_size)
    plt.ylabel("Actual class", fontsize=args.font_size)
    plt.title(f"Confusion Matrix (score_threshold={args.score_thresh:.2f})", fontsize=args.font_size)
    cm_path = out_dir / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix saved → {cm_path}")

    # PR curve
    y_scores, y_true = [], []
    for pred, tgt in zip(preds, targets):
        p_boxes = pred["boxes"]
        p_scores = pred["scores"]
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
            y_true.extend([0]*len(p_boxes))
            y_scores.extend(p_scores.tolist())

    if y_scores:
        prec, rec, thr = precision_recall_curve(y_true, y_scores)
        f1 = 2*prec*rec/(prec+rec+1e-6)
        best_idx = int(np.nanargmax(f1))
        best_f1 = f1[best_idx]
        best_thr = thr[best_idx] if best_idx < len(thr) else 1.0

        plt.figure()
        plt.plot(rec, prec, label=f"best F1={best_f1:.3f} at score_threshold={best_thr:.2f}")
        plt.xlabel("Recall", fontsize=args.font_size)
        plt.ylabel("Precision", fontsize=args.font_size)
        plt.title("PR Curve", fontsize=args.font_size)
        plt.grid(True)
        plt.legend(loc="lower left", fontsize=args.font_size)
        plt.xlim(-0.05, 1.05)
        plt.ylim(0.58, 1.02)
        pr_path = out_dir / "pr_curve.png"
        plt.savefig(pr_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"PR curve saved → {pr_path}")
    else:
        print("No predictions above score_thresh – PR curve skipped.")

if __name__ == "__main__":
    main()
