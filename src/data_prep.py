import argparse
import os
import random
import shutil

DEFAULT_CLASSES = ["plastic", "metal", "paper", "cardboard", "glass", "organic", "trash"]


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare train/val dataset folders")
    parser.add_argument("--src_dir", type=str, required=True, help="Source root directory containing class subfolders")
    parser.add_argument("--dest_dir", type=str, default="data", help="Destination base directory")
    parser.add_argument("--split", type=float, default=0.8, help="Train split ratio, rest for val")
    parser.add_argument("--classes", type=str, nargs="*", default=DEFAULT_CLASSES, help="List of class names")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def copy_images_for_class(src_class_dir, train_class_dir, val_class_dir, split, seed):
    if not os.path.exists(src_class_dir):
        return 0,0
    os.makedirs(train_class_dir, exist_ok=True)
    os.makedirs(val_class_dir, exist_ok=True)
    files = [f for f in os.listdir(src_class_dir) if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"))]
    random.Random(seed).shuffle(files)
    split_idx = int(len(files) * split)
    train_files = files[:split_idx]
    val_files = files[split_idx:]
    for f in train_files:
        shutil.copy2(os.path.join(src_class_dir, f), os.path.join(train_class_dir, f))
    for f in val_files:
        shutil.copy2(os.path.join(src_class_dir, f), os.path.join(val_class_dir, f))
    return len(train_files), len(val_files)


def main():
    args = parse_args()
    random.seed(args.seed)
    for c in args.classes:
        src_class_dir = os.path.join(args.src_dir, c)
        train_class_dir = os.path.join(args.dest_dir, "train", c)
        val_class_dir = os.path.join(args.dest_dir, "val", c)
        train_count, val_count = copy_images_for_class(src_class_dir, train_class_dir, val_class_dir, args.split, args.seed)
        print(f"{c}: train={train_count}, val={val_count}")
    print("Dataset preparation done.")

if __name__ == "__main__":
    main()
