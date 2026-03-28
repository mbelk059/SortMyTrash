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

For Grad-CAM I used one image from `data/test/plastic/` and saved it as `outputs/gradcam_plastic_test.png`.

