import argparse
import json
import os
import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)
import seaborn as sns
import matplotlib.pyplot as plt

from dataset import WasteDataset, get_transforms, DEFAULT_CLASSES
from model import WasteClassifier


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate waste classifier")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--split", type=str, default="test", help="Usually test for final report")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--backbone", type=str, default="resnet18")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--prefix", type=str, default="eval", help="Prefix for saved metric files")
    return parser.parse_args()


def load_model(args):
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    saved = ckpt.get("args") or {}
    backbone = saved.get("backbone", args.backbone)
    model = WasteClassifier(num_classes=len(DEFAULT_CLASSES), backbone_name=backbone, pretrained=False)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, device, backbone


def run_eval(args):
    os.makedirs(args.output_dir, exist_ok=True)
    model, device, backbone_used = load_model(args)
    dataset = WasteDataset(
        args.data_dir, split=args.split, classes=DEFAULT_CLASSES, transform=get_transforms(args.image_size, is_train=False)
    )
    class_names = dataset.classes
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
    labels_idx = list(range(len(class_names)))

    acc = float(accuracy_score(all_labels, all_preds)) if len(all_labels) else 0.0
    bacc = float(balanced_accuracy_score(all_labels, all_preds)) if len(all_labels) else 0.0
    f1_macro = float(f1_score(all_labels, all_preds, average="macro", labels=labels_idx, zero_division=0))
    f1_weighted = float(f1_score(all_labels, all_preds, average="weighted", labels=labels_idx, zero_division=0))

    print("Accuracy:", acc)
    print("Balanced accuracy:", bacc)
    print("F1 (macro):", f1_macro)
    print("F1 (weighted):", f1_weighted)
    report_txt = classification_report(
        all_labels, all_preds, labels=labels_idx, target_names=class_names, digits=4, zero_division=0
    )
    print("Classification report:")
    print(report_txt)

    cm = confusion_matrix(all_labels, all_preds, labels=labels_idx)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix ({args.split})")
    plt.tight_layout()
    cm_png = os.path.join(args.output_dir, f"{args.prefix}_confusion_matrix.png")
    plt.savefig(cm_png)
    print("Saved confusion matrix to", cm_png)

    np.savetxt(os.path.join(args.output_dir, f"{args.prefix}_confusion_matrix.csv"), cm, delimiter=",", fmt="%d")

    report_path = os.path.join(args.output_dir, f"{args.prefix}_classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_txt)
    print("Wrote", report_path)

    prec, rec, f1, sup = precision_recall_fscore_support(
        all_labels, all_preds, labels=labels_idx, zero_division=0
    )
    per_class = []
    for i, name in enumerate(class_names):
        per_class.append(
            {"class": name, "precision": float(prec[i]), "recall": float(rec[i]), "f1": float(f1[i]), "support": int(sup[i])}
        )

    metrics = {
        "split": args.split,
        "backbone": backbone_used,
        "n_samples": int(len(all_labels)),
        "accuracy": acc,
        "balanced_accuracy": bacc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "per_class": per_class,
    }
    metrics_path = os.path.join(args.output_dir, f"{args.prefix}_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print("Wrote", metrics_path)


if __name__ == "__main__":
    args = parse_args()
    run_eval(args)
