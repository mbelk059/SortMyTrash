"""Generate Grad-CAM images for many files (e.g. for the report)."""
import argparse
import glob
import os

from gradcam import run_gradcam


def parse_args():
    p = argparse.ArgumentParser(description="Batch Grad-CAM")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--glob_pattern", type=str, default=None, help="e.g. report_figures/**/*.jpg")
    p.add_argument("--image_paths", type=str, nargs="*", default=None)
    p.add_argument("--output_dir", type=str, default="outputs/gradcam_batch")
    p.add_argument("--backbone", type=str, default="resnet18")
    p.add_argument("--image_size", type=int, default=224)
    return p.parse_args()


def main():
    args = parse_args()
    paths = []
    if args.image_paths:
        paths.extend(args.image_paths)
    if args.glob_pattern:
        paths.extend(sorted(glob.glob(args.glob_pattern, recursive=True)))
    paths = [p for p in paths if os.path.isfile(p)]
    if not paths:
        print("No images found.")
        return
    os.makedirs(args.output_dir, exist_ok=True)
    ns = argparse.Namespace(
        checkpoint=args.checkpoint,
        image_path="",
        output="",
        backbone=args.backbone,
        image_size=args.image_size,
    )
    for img_path in paths:
        stem = os.path.splitext(os.path.basename(img_path))[0]
        ns.image_path = img_path
        ns.output = os.path.join(args.output_dir, f"gradcam_{stem}.png")
        run_gradcam(ns)
    print(f"Wrote {len(paths)} overlays under {args.output_dir}")


if __name__ == "__main__":
    main()
