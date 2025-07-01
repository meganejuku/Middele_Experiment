#!/usr/bin/env python3
"""
Precision–Recall & F1 evaluation
--------------------------------
  python eval_pr.py \
      --data_root dataset/TIFF_battery \         # フォーマット自動判定 (HSI or RGB)
      --weights runs/hsi_rcnn.pth          # 学習済み重み
      --split val                          # train / val / test
      --output_plot runs/pr_val.png        # PR 曲線保存先
      --score_thresh 0.0                   # スコア下限 (曲線用なので 0.0 推奨)
Options:
  --iou_thresh   IoU ≥ ? で TP とみなす (default 0.5)
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import precision_recall_curve, confusion_matrix
import seaborn as sns
from torch.utils.data import DataLoader
from torchvision.ops import box_iou
from torchmetrics.detection import MeanAveragePrecision

from hsi_faster_rcnn import HSIVOCDataset, get_model    # 同ディレクトリにある学習スクリプト

# ───── 評価 ──────────────────────────────────────────────────────────────
def collect_results(model, loader, device):
    all_preds, all_targets = [], []
    model.eval()
    with torch.no_grad():
        for img, tgt, _ in loader.dataset:           # dataset から直接イテレート
            pred = model([img.to(device)])[0]
            # torchmetrics用にcpuに送る
            for k in pred:
                pred[k] = pred[k].cpu()
            all_preds.append(pred)
            all_targets.append(tgt)
    return all_preds, all_targets

# ───── メイン ──────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--weights",   required=True)
    ap.add_argument("--split",     default="val", choices=["train","val","test"])
    ap.add_argument("--output_dir", default="runs/pr_val")
    ap.add_argument("--score_thresh", type=float, default=0.0)
    ap.add_argument("--iou_thresh",   type=float, default=0.5)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = HSIVOCDataset(args.data_root, split=args.split)
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    in_ch  = ds[0][0].shape[0]

    model = get_model(2, in_ch).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))

    preds, targets = collect_results(model, loader, device)

    # --- mAP ---------------------------
    metric = MeanAveragePrecision(box_format='xyxy', iou_type='bbox')
    metric.update(preds, targets)
    results = metric.compute()
    print(f"[{args.split}] mAP50-95: {results['map']:.3f}")
    print(f"[{args.split}] mAP50:    {results['map_50']:.3f}")

    # --- Confusion Matrix ---------------------------
    all_scores = np.concatenate([p['scores'] for p in preds])
    all_labels = []
    for p, t in zip(preds, targets):
        if len(t['boxes']) == 0:
            all_labels.extend([0] * len(p['boxes']))
            continue
        iou = box_iou(p['boxes'], t['boxes'])
        matched = (iou.max(dim=1).values >= args.iou_thresh)
        all_labels.extend(matched.int().tolist())

    y_true = np.array(all_labels)

    # score_thresh を超えるボックスのみで Confusion Matrix を作成
    score_mask = all_scores >= args.score_thresh
    cm_y_true = y_true[score_mask]
    cm_y_pred = np.ones_like(cm_y_true)

    cm = confusion_matrix(cm_y_true, cm_y_pred, labels=[0, 1])
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    cm_path = Path(args.output_dir) / "confusion_matrix.png"
    cm_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    print(f"Confusion matrix saved → {cm_path}")
    plt.close()

    # --- PR Curve ---------------------------
    p, r, th = precision_recall_curve(y_true, all_scores)
    f1 = 2 * (p * r) / (p + r + 1e-6)
    best = np.nanargmax(f1)
    best_f1, best_thr = f1[best], th[best] if best < len(th) else 1.0
    print(f"[{args.split}] Best F1={best_f1:.3f} at score≥{best_thr:.2f}")

    plt.figure()
    plt.plot(r, p, label=f"F1={best_f1:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"PR Curve ({args.split})")
    plt.grid()
    plt.legend()
    pr_path = Path(args.output_dir) / "pr_curve.png"
    plt.savefig(pr_path, dpi=150, bbox_inches="tight")
    print(f"PR curve saved → {pr_path}")
    plt.close()

if __name__ == "__main__":
    from pathlib import Path
    main()