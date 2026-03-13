# SortMyTrash Final Results

## Training Results
- Training dataset: TrashNet resized dataset
- Model: ResNet18 backbone with final linear classifier (7 classes)
- Augmentation: random flip, rotation, color jitter
- 1 epoch run results: train_acc ~0.664, val_acc ~0.823

## Evaluation Metrics (val set)
- Accuracy: 0.8228
- Per-class precision/recall/F1 (from classification report):
  - plastic: 0.8261 / 0.7835 / 0.8042
  - metal: 0.7200 / 0.8780 / 0.7912
  - paper: 0.9091 / 0.8403 / 0.8734
  - cardboard: 0.8916 / 0.9136 / 0.9024
  - glass: 0.7636 / 0.8317 / 0.7962
  - organic: 0 / 0 / 0 (zero support in dataset)
  - trash: 0.9231 / 0.4286 / 0.5854

## Explainability
- Grad-CAM visual explanation output generated: outputs/gradcam_plastic.png

## Notes
- Organic class is absent in TrashNet sample images; for full course, include an organic image subset.
- Next improvements: train more epochs, tune learning rate, and evaluate on held-out real bin images.
