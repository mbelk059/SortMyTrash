"""Print per-split, per-class image counts for data/ layout (train/val/test)."""
import argparse
import json
import os

from dataset import DEFAULT_CLASSES


def count_split(root, split, classes):
    split_dir = os.path.join(root, split)
    out = {}
    total = 0
    for c in classes:
        d = os.path.join(split_dir, c)
        if not os.path.isdir(d):
            out[c] = 0
            continue
        n = sum(
            1
            for f in os.listdir(d)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"))
        )
        out[c] = n
        total += n
    return out, total


def main():
    parser = argparse.ArgumentParser(description="Dataset statistics")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--classes", type=str, nargs="*", default=DEFAULT_CLASSES)
    parser.add_argument("--output_json", type=str, default=None, help="If set, write counts to this JSON file")
    args = parser.parse_args()

    splits = ["train", "val", "test"]
    report = {"data_dir": args.data_dir, "splits": {}}
    grand = 0
    for sp in splits:
        counts, t = count_split(args.data_dir, sp, args.classes)
        report["splits"][sp] = counts
        grand += t
        print(f"{sp}: total={t}")
        for c in args.classes:
            print(f"  {c}: {counts[c]}")

    print(f"grand_total={grand}")
    if args.output_json:
        parent = os.path.dirname(args.output_json)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print("Wrote", args.output_json)


if __name__ == "__main__":
    main()

