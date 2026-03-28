import argparse
import os
import random
import shutil

DEFAULT_CLASSES = ["plastic", "metal", "paper", "cardboard", "glass", "organic", "trash"]
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare train/val/test folders from class subfolders (ratios must sum to 1.0)"
    )
    parser.add_argument("--src_dir", type=str, required=True, help="Source root with one folder per class")
    parser.add_argument("--dest_dir", type=str, default="data", help="Destination base (train/val/test)")
    parser.add_argument("--train_ratio", type=float, default=0.70, help="Fraction for train")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="Fraction for val")
    parser.add_argument("--test_ratio", type=float, default=0.15, help="Fraction for test")
    parser.add_argument("--classes", type=str, nargs="*", default=DEFAULT_CLASSES)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--clear_dest",
        action="store_true",
        help="Remove existing train/val/test under dest_dir before copying",
    )
    return parser.parse_args()


def list_images(src_class_dir):
    return [f for f in os.listdir(src_class_dir) if f.lower().endswith(IMAGE_EXTS)]


def split_three_way(files, train_ratio, val_ratio, test_ratio, seed):
    r = train_ratio + val_ratio + test_ratio
    if abs(r - 1.0) > 1e-6:
        raise ValueError(f"train_ratio + val_ratio + test_ratio must equal 1.0, got {r}")
    rng = random.Random(seed)
    shuffled = files[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    n_train = min(n_train, n)
    n_val = min(n_val, max(0, n - n_train))
    train_f = shuffled[:n_train]
    val_f = shuffled[n_train : n_train + n_val]
    test_f = shuffled[n_train + n_val :]
    return train_f, val_f, test_f


def copy_list(files, src_class_dir, dest_class_dir):
    os.makedirs(dest_class_dir, exist_ok=True)
    for f in files:
        shutil.copy2(os.path.join(src_class_dir, f), os.path.join(dest_class_dir, f))
    return len(files)


def main():
    args = parse_args()
    random.seed(args.seed)
    if not os.path.isdir(args.src_dir):
        raise SystemExit(
            'src_dir is missing or not a folder: ' + os.path.abspath(args.src_dir) + chr(10) +
            'Add class subfolders (plastic, metal, ...) or run import_trashnet.py first.'
        )
    if args.clear_dest:
        for sp in ("train", "val", "test"):
            p = os.path.join(args.dest_dir, sp)
            if os.path.isdir(p):
                shutil.rmtree(p)

    total_copied = 0
    for c in args.classes:
        src_class_dir = os.path.join(args.src_dir, c)
        if not os.path.exists(src_class_dir):
            print(f"{c}: skip (no source folder)")
            continue
        files = list_images(src_class_dir)
        per_class_seed = args.seed + hash(c) % 997
        train_f, val_f, test_f = split_three_way(
            files, args.train_ratio, args.val_ratio, args.test_ratio, per_class_seed
        )
        tr = copy_list(train_f, src_class_dir, os.path.join(args.dest_dir, "train", c))
        va = copy_list(val_f, src_class_dir, os.path.join(args.dest_dir, "val", c))
        te = copy_list(test_f, src_class_dir, os.path.join(args.dest_dir, "test", c))
        total_copied += tr + va + te
        print(f"{c}: train={tr}, val={va}, test={te}")
    if total_copied == 0:
        print(
            "\nNo images copied. src_dir needs class subfolders with images (e.g. raw_dataset/plastic/)."
            "\nTrashNet: unzip trashnet-master.zip, then: python src/import_trashnet.py"
        )
        raise SystemExit(1)
    print("Dataset preparation done.")


if __name__ == "__main__":
    main()
