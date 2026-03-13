import argparse
import os
import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

from dataset import WasteDataset, get_transforms, DEFAULT_CLASSES
from model import WasteClassifier, load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate waste classifier")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--backbone", type=str, default="resnet18")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="outputs")
    return parser.parse_args()


def load_model(args):
    model = WasteClassifier(num_classes=len(DEFAULT_CLASSES), backbone_name=args.backbone, pretrained=False)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, device


def run_eval(args):
    os.makedirs(args.output_dir, exist_ok=True)
    model, device = load_model(args)
    dataset = WasteDataset(args.data_dir, split=args.split, classes=DEFAULT_CLASSES, transform=get_transforms(args.image_size, is_train=False))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    print("Accuracy:", accuracy_score(all_labels, all_preds))
    print("Classification report:")
    print(classification_report(all_labels, all_preds, target_names=DEFAULT_CLASSES, digits=4))

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=DEFAULT_CLASSES, yticklabels=DEFAULT_CLASSES, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    cm_path = os.path.join(args.output_dir, "confusion_matrix.png")
    plt.savefig(cm_path)
    print("Saved confusion matrix to", cm_path)


if __name__ == "__main__":
    run_eval()
