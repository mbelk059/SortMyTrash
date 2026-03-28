"""
Copy TrashNet class folders into raw_dataset/ so data_prep.py can split them.

Supports nested layouts, e.g. trashnet-master/data/dataset-reszied/dataset-resized/
and misspelled class folders (cardbord -> cardboard).
"""
import argparse
import os
import shutil
from typing import Dict, Optional, Set

TRASHNET_CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

FOLDER_ALIASES = {
    "cardboard": "cardboard",
    "cardbord": "cardboard",
    "glass": "glass",
    "metal": "metal",
    "paper": "paper",
    "plastic": "plastic",
    "plastics": "plastic",
    "trash": "trash",
}


def folder_to_canonical(folder_name: str) -> Optional[str]:
    n = folder_name.lower().strip()
    return FOLDER_ALIASES.get(n)


def is_trashnet_leaf_dir(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    found = set()
    for name in os.listdir(path):
        sub = os.path.join(path, name)
        if not os.path.isdir(sub):
            continue
        c = folder_to_canonical(name)
        if c:
            found.add(c)
    return found == set(TRASHNET_CLASSES)


def find_dataset_root(start: str) -> Optional[str]:
    start = os.path.abspath(start)
    if not os.path.isdir(start):
        return None
    if is_trashnet_leaf_dir(start):
        return start
    for root, dirs, _ in os.walk(start):
        for d in dirs:
            candidate = os.path.join(root, d)
            if is_trashnet_leaf_dir(candidate):
                return candidate
    return None


def copy_class_images(src_dir: str, dest_dir: str) -> int:
    os.makedirs(dest_dir, exist_ok=True)
    n = 0
    for f in os.listdir(src_dir):
        if f.lower().endswith(IMAGE_EXTS):
            shutil.copy2(os.path.join(src_dir, f), os.path.join(dest_dir, f))
            n += 1
    return n


def main():
    p = argparse.ArgumentParser(description="Build raw_dataset/ from TrashNet folders")
    p.add_argument(
        "--trashnet_root",
        type=str,
        default="trashnet-master/data",
        help="Search here for a folder containing the six class subfolders",
    )
    p.add_argument("--dest", type=str, default="raw_dataset", help="Output folder")
    args = p.parse_args()

    root = find_dataset_root(args.trashnet_root)
    if root is None:
        print(
            "Could not find all six class folders under:\n  "
            + os.path.abspath(args.trashnet_root)
            + "\n\nTry: python src/import_trashnet.py --trashnet_root trashnet-master/data"
        )
        raise SystemExit(1)

    print(f"Using TrashNet images from: {root}")
    name_by_canon = {}
    for name in os.listdir(root):
        sub = os.path.join(root, name)
        if not os.path.isdir(sub):
            continue
        c = folder_to_canonical(name)
        if c:
            name_by_canon[c] = name

    total = 0
    for c in TRASHNET_CLASSES:
        disk_name = name_by_canon[c]
        src = os.path.join(root, disk_name)
        dest = os.path.join(args.dest, c)
        n = copy_class_images(src, dest)
        total += n
        print(f"  {c} (from {disk_name}/): copied {n} images -> {dest}/")

    os.makedirs(os.path.join(args.dest, "organic"), exist_ok=True)
    print(
        f"\nDone. {total} images under {args.dest}/.\n"
        "Add photos under raw_dataset/organic/ if you need the organic class."
    )


if __name__ == "__main__":
    main()
