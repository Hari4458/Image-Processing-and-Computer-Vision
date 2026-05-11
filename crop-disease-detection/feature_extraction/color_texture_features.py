from pathlib import Path
import argparse
import cv2
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops


VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def default_input_dir() -> Path:
    workspace_root = Path(__file__).resolve().parents[2]
    segmented = workspace_root / "Data" / "crop_disease" / "segmented"
    processed = workspace_root / "Data" / "crop_disease" / "processed"
    return segmented if segmented.exists() else processed


def default_output_csv() -> Path:
    workspace_root = Path(__file__).resolve().parents[1]
    return workspace_root / "dataset" / "labels" / "features.csv"


def color_moments(image: np.ndarray) -> list[float]:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    feats = []
    for c in cv2.split(hsv):
        feats.extend([float(np.mean(c)), float(np.std(c))])
    return feats


def glcm_features(image: np.ndarray) -> list[float]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA)

    glcm = graycomatrix(
        gray,
        distances=[1],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=256,
        symmetric=True,
        normed=True,
    )

    props = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]
    values = [float(np.mean(graycoprops(glcm, p))) for p in props]
    return values


def extract_features(image: np.ndarray) -> list[float]:
    return color_moments(image) + glcm_features(image)


def build_feature_table(input_dir: Path) -> pd.DataFrame:
    rows = []
    for class_dir in sorted([p for p in input_dir.iterdir() if p.is_dir()]):
        for image_path in class_dir.glob("*"):
            if image_path.suffix.lower() not in VALID_EXT:
                continue

            image = cv2.imread(str(image_path))
            if image is None:
                continue

            feats = extract_features(image)
            rows.append({
                "image": image_path.name,
                "label": class_dir.name,
                **{f"f{i}": v for i, v in enumerate(feats, start=1)},
            })

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract color + texture features.")
    parser.add_argument("--input", type=Path, default=default_input_dir(), help="Input directory")
    parser.add_argument("--output", type=Path, default=default_output_csv(), help="Output CSV path")
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input directory not found: {args.input}")

    df = build_feature_table(args.input)
    if df.empty:
        raise RuntimeError("No images found for feature extraction.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Saved features to: {args.output}")
    print(f"Total samples: {len(df)}")


if __name__ == "__main__":
    main()
