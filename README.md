# HSI Faster R-CNN

ハイパースペクトル画像（HSI）に対応した Faster R-CNN を用いて、モバイルバッテリー（1 クラス）を検出する PyTorch プロジェクトです。学習・推論・評価がそれぞれ独立したスクリプトとして整理されています。

## 主なスクリプト
- `hsi_faster_rcnn.py`  
  EfficientNet-B3 をバックボーンにした Faster R-CNN の学習スクリプト。混合精度と学習曲線の保存に対応しています。
- `inference.py`  
  学習済みウェイトを用いて画像群へ推論を行い、検出結果を可視化します。
- `eval_pr.py`  
  mAP@0.50:0.95、mAP@0.50、混同行列、PR カーブを計算・出力します。
- `utils.py`  
  前処理や共通処理をまとめたユーティリティ（必要に応じて import されています）。

## 動作環境
- Python 3.12 以上
- PyTorch 2.7 / TorchVision 0.22 以降（CUDA 対応 GPU を推奨）
- メモリ 16 GB 以上、GPU メモリ 8 GB 程度を推奨

依存パッケージは `pyproject.toml` に記載しています。GPU を利用する場合は、事前に CUDA 対応の PyTorch を公式手順でインストールしてください。

## セットアップ
1. 仮想環境の作成
   ```bash
   python -m venv .venv
   source .venv/bin/activate      # Windows の場合: .venv\Scripts\activate
   python -m pip install --upgrade pip
   ```
2. 依存関係のインストール
   ```bash
   pip install -e .
   ```
   もしくは `uv` を使用している場合は `uv sync` でも構いません。

## データセットの準備
本リポジトリには学習用データは含まれていません。PASCAL VOC 形式のアノテーション（`.xml`）と HSI（`.tiff` など）を以下の構成で配置してください。

```
/path/to/dataset/
  ├─ train/
  │   ├─ Annotations/*.xml
  │   └─ TIFFImages/*.tiff  # または images/*.tif, *.png など
  ├─ val/
  │   ├─ Annotations/*.xml
  │   └─ TIFFImages/*.tiff
  └─ test/
      ├─ Annotations/*.xml
      └─ TIFFImages/*.tiff
```

- バンド数が 3 以外の HSI を想定していますが、RGB 画像も扱えます。
- アノテーションのクラス名は `battery` を想定しています。必要に応じてスクリプト内のフィルタを変更してください。
- `sample/` に小さなサンプルがあれば参考用に確認できます。

## 学習
```bash
python hsi_faster_rcnn.py \
  --data_root /path/to/dataset \
  --epochs 20 \
  --batch_size 2 \
  --lr 5e-3 \
  --save_path models/hsi_rcnn.pth
```

- `--data_root` は前述のデータセットルートを指定します。
- 学習中のロスと指標は標準出力へ記録され、学習後に `runs/loss/` 以下へ学習曲線が保存されます。
- `models/` ディレクトリは自動で作成され、学習済みウェイトが保存されます。

## 推論
```bash
python inference.py \
  --data_root /path/to/dataset \
  --split test \
  --weights models/hsi_rcnn.pth \
  --score_thresh 0.5 \
  --output_dir runs/predict
```

- 指定した `split` 内の全画像に対して推論を行い、`output_dir` に PNG 形式で可視化結果を保存します。
- `--score_thresh` で検出スコアの下限値を調整できます。

## 評価
```bash
python eval_pr.py \
  --data_root /path/to/dataset \
  --split val \
  --weights models/hsi_rcnn.pth \
  --score_thresh 0.5 \
  --iou_thresh 0.5 \
  --output_dir runs/eval
```

- コンソールに mAP を出力し、`output_dir` に PR カーブと混同行列を保存します。
- `--font_size` を指定すると、出力図の注釈サイズをまとめて変更できます。

## よくある質問
- **CUDA が検出されない**: PyTorch をインストールする際に CUDA 対応ビルドを選択してください。
- **データを読み込めない**: `TIFFImages` ではなく `images` といった別ディレクトリ名を使う場合は、スクリプトが自動検出します。想定外の構造になっていないかご確認ください。
- **クラス数を変更したい**: `hsi_faster_rcnn.get_model` の `num_classes`、およびアノテーションのクラス名フィルタを合わせて変更してください。

## リポジトリ構成（抜粋）
- `hsi_faster_rcnn.py` … 学習用スクリプト
- `inference.py` … 推論スクリプト
- `eval_pr.py` … 評価スクリプト
- `utils.py` … 前処理・共通関数
- `dataset/` … データセット（空または未同梱）
- `runs/` … 学習・推論・評価結果の出力先（実行時に生成）
- `models/` … 学習済みモデルの保存先（実行時に生成）

## ライセンス
未設定です。必要に応じてライセンスファイルを追加してください。

**注意：このREADMEはCodex CLIによって生成されたものです。一応確認はしていますが、実際の動作とは異なる可能性があります。**