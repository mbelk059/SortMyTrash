import os
import argparse
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import accuracy_score

from dataset import WasteDataset, get_transforms, DEFAULT_CLASSES
from model import WasteClassifier


def parse_args():
    parser = argparse.ArgumentParser(description="Train waste classifier")
    parser.add_argument("--data_dir", type=str, default="data", help="Data directory with train/val folders")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--backbone", type=str, default="resnet18")
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def prepare_loaders(args):
    train_ds = WasteDataset(args.data_dir, split="train", classes=DEFAULT_CLASSES, transform=get_transforms(args.image_size, is_train=True))
    val_ds = WasteDataset(args.data_dir, split="val", classes=DEFAULT_CLASSES, transform=get_transforms(args.image_size, is_train=False))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, val_loader


def evaluate_batch(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    accuracy = accuracy_score(all_labels, all_preds)
    return accuracy


def train():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    train_loader, val_loader = prepare_loaders(args)
    model = WasteClassifier(num_classes=len(DEFAULT_CLASSES), backbone_name=args.backbone, dropout=args.dropout, pretrained=args.pretrained)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    best_val = 0.0
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        all_preds = []
        all_labels = []
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            pbar.set_postfix({"loss": loss.item()})

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = accuracy_score(all_labels, all_preds)

        val_acc = evaluate_batch(model, val_loader, device)
        scheduler.step(1.0 - val_acc)

        if val_acc > best_val:
            best_val = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "classes": DEFAULT_CLASSES,
            }, os.path.join(args.output_dir, "model_best.pth"))

        history["train_loss"].append(epoch_loss)
        history["val_loss"].append(None)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch}: train_loss={epoch_loss:.4f}, train_acc={epoch_acc:.4f}, val_acc={val_acc:.4f}")

    torch.save({
        "model_state_dict": model.state_dict(),
        "classes": DEFAULT_CLASSES,
        "args": vars(args),
    }, os.path.join(args.output_dir, "model_last.pth"))
    print(f"Training complete. Best val acc={best_val:.4f}. Saved to outputs/model_best.pth")


if __name__ == "__main__":
    train()
