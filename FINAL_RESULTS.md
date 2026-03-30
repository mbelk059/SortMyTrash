# Results

What I ran: TrashNet resized images → `import_trashnet.py` → `data_prep.py` → train with ResNet-18 pretrained on ImageNet. Only 3 epochs because training on CPU was slow (~3.5 min per epoch). Seed 42, batch 16, lr 1e-4.

Best validation accuracy hit **90.8%** during training; the saved weights are in `outputs/model_best.pth`.

## Test set

379 test images. Overall **accuracy ~91%**. I’m also reporting balanced accuracy and F1, macro F1 looks worse (~77%) mostly because the model still has an “organic” output neuron but never had any organic photos, so that class is empty in the data. Weighted F1 (~91%) matches how well it does on the classes that actually appear.

| Metric | Value |
|--------|--------|
| Accuracy | ~0.91 |
| Balanced accuracy | ~0.89 |
| F1 macro | ~0.77 |
| F1 weighted | ~0.91 |

Per class on the test split:

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| plastic | 0.94 | 0.90 | 0.92 | 73 |
| metal | 0.81 | 0.90 | 0.85 | 61 |
| paper | 0.93 | 0.93 | 0.93 | 89 |
| cardboard | 0.97 | 0.95 | 0.96 | 61 |
| glass | 0.89 | 0.91 | 0.90 | 75 |
| organic | — | — | — | 0 |
| trash | 0.94 | 0.75 | 0.83 | 20 |

Trash has the fewest test images so not surprising it’s a bit weaker. Confusion matrix and full numbers saved under `outputs/` (`test_metrics.json`, etc.).

`evaluate.py` and `gradcam.py` also print an **illustrative** blue/green/black bin hint via `src/bin_hint.py` (not legal disposal advice).

For Grad-CAM I used one image from `data/test/plastic/` and saved it as `outputs/gradcam_plastic_test.png` (or `outputs/gradcam.png` on a later run).

## Baseline run (frozen backbone)

Baseline command:
`python -u src/train.py --data_dir data --epochs 3 --batch_size 16 --lr 1e-4 --backbone resnet18 --pretrained --freeze_backbone --seed 42 --output_dir outputs_baseline`

This baseline trains only the classifier head (frozen ResNet-18 backbone) for 3 epochs on CPU.

Validation progression from `outputs_baseline/training_history.json`:

- Epoch 1: train_loss=1.8966, val_loss=1.6254, val_acc=0.3456
- Epoch 2: train_loss=1.7360, val_loss=1.4934, val_acc=0.4248
- Epoch 3: train_loss=1.6168, val_loss=1.3939, val_acc=0.5145

Best checkpoint: `outputs_baseline/model_best.pth` (best val_acc **0.5145**)

Baseline test command:
`python src/evaluate.py --checkpoint outputs_baseline/model_best.pth --data_dir data --split test --prefix test_baseline --output_dir outputs_baseline`

Baseline test metrics (`outputs_baseline/test_baseline_metrics.json`):

| Metric | Value |
|--------|-------|
| Accuracy | 0.4934 |
| Balanced accuracy | 0.4358 |
| F1 macro | 0.3663 |
| F1 weighted | 0.4835 |

Per-class baseline test results:

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| plastic | 0.4253 | 0.5068 | 0.4625 | 73 |
| metal | 0.4342 | 0.5410 | 0.4818 | 61 |
| paper | 0.5385 | 0.4719 | 0.5030 | 89 |
| cardboard | 0.7381 | 0.5082 | 0.6019 | 61 |
| glass | 0.4583 | 0.5867 | 0.5146 | 75 |
| organic | — | — | — | 0 |
| trash | — | — | — | 20 |

Baseline artifacts are in `outputs_baseline/`:
`test_baseline_metrics.json`, `test_baseline_classification_report.txt`, `test_baseline_confusion_matrix.csv`, and `test_baseline_confusion_matrix.png`.

## Main vs Baseline summary

| Metric | Main (fine-tuned) | Baseline (frozen) |
|--------|:-----------------:|:-----------------:|
| Accuracy | 0.910 | 0.493 |
| Balanced accuracy | 0.891 | 0.436 |
| F1 macro | 0.772 | 0.366 |
| F1 weighted | 0.911 | 0.483 |
