from __future__ import annotations

from pathlib import Path
import argparse
import subprocess
import sys


def run_step(step_name: str, cmd: list[str]) -> None:
    print(f"\n[RUN] {step_name}")
    print(" ".join(cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Step failed: {step_name} (exit code {result.returncode})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full crop disease pipeline in one command.")
    parser.add_argument(
        "--raw",
        type=Path,
        default=Path("Data/crop_disease/raw"),
        help="Raw dataset folder (class-wise subfolders)",
    )
    parser.add_argument(
        "--processed",
        type=Path,
        default=Path("Data/crop_disease/processed"),
        help="Processed output folder",
    )
    parser.add_argument(
        "--segmented",
        type=Path,
        default=Path("Data/crop_disease/segmented"),
        help="Segmented output folder",
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("crop-disease-detection/dataset/labels/features.csv"),
        help="Feature CSV output path",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("crop-disease-detection/runs/classify/svm_model.joblib"),
        help="Trained model output path",
    )
    parser.add_argument("--size", type=int, default=224, help="Resize dimension")
    parser.add_argument("--clusters", type=int, default=3, help="K-Means cluster count")
    args = parser.parse_args()

    if not args.raw.exists():
        raise FileNotFoundError(f"Raw dataset folder not found: {args.raw}")

    py = sys.executable

    run_step(
        "Resize images",
        [
            py,
            "crop-disease-detection/preprocessing/resize.py",
            "--input",
            str(args.raw),
            "--output",
            str(args.processed),
            "--size",
            str(args.size),
        ],
    )

    run_step(
        "Segment images (K-Means + Otsu)",
        [
            py,
            "crop-disease-detection/segmentation/kmeans_otsu.py",
            "--input",
            str(args.processed),
            "--output",
            str(args.segmented),
            "--clusters",
            str(args.clusters),
        ],
    )

    run_step(
        "Extract features",
        [
            py,
            "crop-disease-detection/feature_extraction/color_texture_features.py",
            "--input",
            str(args.segmented),
            "--output",
            str(args.features),
        ],
    )

    run_step(
        "Train classifier",
        [
            py,
            "crop-disease-detection/classification/train_classifier.py",
            "--features",
            str(args.features),
            "--model-out",
            str(args.model),
        ],
    )

    print("\n[SUCCESS] Full pipeline completed.")
    print(f"Model saved at: {args.model}")


if __name__ == "__main__":
    main()
