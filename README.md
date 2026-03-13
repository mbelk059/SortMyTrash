# SortMyTrash

This project builds a custom waste-material classifier and explainable reasoning for recycling guidance.

## Project Scope

- Train a custom image classifier on waste data (TrashNet + Kaggle Garbage)
- Evaluate with valid metrics: accuracy, precision, recall, F1, confusion matrix
- Produce Grad-CAM visual explanations for model decisions
- (Stretch goal) Build an optional demo interface.

## Setup

1. Create a Python virtual environment
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. Prepare data directory structure:
   ```
   data/
     train/
       plastic/
       metal/
       paper/
       cardboard/
       glass/
       organic/
       trash/
     val/
       ...
   ```
3. Run training:
   ```bash
   python src/train.py --data_dir data --epochs 20 --batch_size 32 --lr 1e-4 --backbone resnet18
   ```
4. Evaluate:
   ```bash
   python src/evaluate.py --checkpoint outputs/model_best.pth --data_dir data
   ```
5. Grad-CAM explain one image:
   ```bash
   python src/gradcam.py --checkpoint outputs/model_best.pth --image_path example.jpg --output outputs/gradcam.png
   ```

### Optional: Prepare data from a raw folder
If you have a raw folder with class subfolders, run:
```bash
python src/data_prep.py --src_dir raw_dataset --dest_dir data --split 0.8
```

## Structure

- `src/train.py` - training pipeline
- `src/evaluate.py` - metrics and confusion matrix evaluation
- `src/gradcam.py` - visual explanation heatmap
- `src/model.py` - model creation
- `src/dataset.py` - dataset loader and transforms
