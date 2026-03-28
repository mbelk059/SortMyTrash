# ♻️ SortMyTrash

Image classifier for household waste: given a photo, predict one of seven material classes (plastic, metal, paper, cardboard, glass, organic, trash). Training uses transfer learning (ResNet or EfficientNet). `gradcam.py` overlays a heatmap so you can see which regions influenced the prediction.

## Project Scope

- Train a custom image classifier on waste data (TrashNet + Kaggle Garbage)
- Evaluate with valid metrics: accuracy, precision, recall, F1, confusion matrix
- Produce Grad-CAM visual explanations for model decisions

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Data

Expected layout under `data/`:

```
data/train/<class>/*.jpg
data/val/<class>/*.jpg
data/test/<class>/*.jpg
```

Class folder names: `plastic`, `metal`, `paper`, `cardboard`, `glass`, `organic`, `trash`.

If you start from one folder per class (e.g. after downloading datasets), build splits with:

```bash
python src/data_prep.py --src_dir raw_dataset --dest_dir data --train_ratio 0.70 --val_ratio 0.15 --test_ratio 0.15 --clear_dest
```

Count images per split (useful before training):

```bash
python src/dataset_stats.py --data_dir data --output_json outputs/dataset_counts.json
```

TrashNet does not ship an `organic` class. If you keep `organic` in the list above, add your own images under `organic/` in the raw folder before running `data_prep.py`.

**Using `trashnet-master.zip`:** unzip it. Class images may live under `trashnet-master/data/dataset/` or under a nested **resized** path such as `trashnet-master/data/dataset-reszied/dataset-resized/` (folder names vary). The importer searches under `trashnet-master/data` by default and recognizes common typos (e.g. `cardbord` → cardboard).

```bash
python src/import_trashnet.py
python src/data_prep.py --src_dir raw_dataset --dest_dir data --train_ratio 0.70 --val_ratio 0.15 --test_ratio 0.15 --clear_dest
```

To search only a specific folder: `python src/import_trashnet.py --trashnet_root trashnet-master/data/dataset-reszied/dataset-resized`

`import_trashnet.py` fills `raw_dataset/` with the six TrashNet classes. Add photos under `raw_dataset/organic/` if you want the seventh class represented.

## Train

Example (what was used for the numbers in `FINAL_RESULTS.md` — CPU-friendly):

```bash
python src/train.py --data_dir data --epochs 3 --batch_size 16 --lr 1e-4 --backbone resnet18 --pretrained --seed 42
```

Longer run if you have time or a GPU:

```bash
python src/train.py --data_dir data --epochs 20 --batch_size 32 --lr 1e-4 --backbone resnet18 --pretrained --seed 42
```

Frozen backbone (train only the linear head / use as a simpler comparison model):

```bash
python src/train.py --data_dir data --epochs 10 --batch_size 32 --lr 1e-3 --backbone resnet18 --pretrained --freeze_backbone --seed 42 --output_dir outputs_baseline
```

EfficientNet:

```bash
python src/train.py --data_dir data --epochs 20 --backbone efficientnet_b0 --pretrained
```

Checkpoints and `training_history.json` go under `outputs/` (or `outputs_baseline/` if you set `--output_dir`).

## Evaluate

On the held-out test split:

```bash
python src/evaluate.py --checkpoint outputs/model_best.pth --data_dir data --split test --prefix test
```

Writes `test_metrics.json`, `test_classification_report.txt`, `test_confusion_matrix.png`, and `test_confusion_matrix.csv` under `outputs/`.

## Grad-CAM

One image:

```bash
python src/gradcam.py --checkpoint outputs/model_best.pth --image_path path\to\image.jpg --output outputs/gradcam.png
```

Several images (glob):

```bash
python src/gradcam_batch.py --checkpoint outputs/model_best.pth --glob_pattern "path\to\images\**\*.jpg" --output_dir outputs/gradcam_batch
```

## Source files

| File | Role |
|------|------|
| `src/train.py` | Training loop, saves best checkpoint |
| `src/evaluate.py` | Accuracy, balanced accuracy, macro/weighted F1, confusion matrix |
| `src/gradcam.py` | Grad-CAM visualization |
| `src/gradcam_batch.py` | Batch Grad-CAM |
| `src/data_prep.py` | Train/val/test split from class folders |
| `src/import_trashnet.py` | Copy TrashNet folders into `raw_dataset/` |
| `src/dataset_stats.py` | Per-split counts |
| `src/model.py` | Backbone + classifier head |
| `src/dataset.py` | Dataset and augmentations |
