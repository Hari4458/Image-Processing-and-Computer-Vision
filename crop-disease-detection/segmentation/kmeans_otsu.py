from pathlib import Path
import argparse
import cv2
import numpy as np


def default_input_dir() -> Path:
    workspace_root = Path(__file__).resolve().parents[2]
    enhanced = workspace_root / "Data" / "crop_disease" / "processed_enhanced"
    fallback = workspace_root / "Data" / "crop_disease" / "processed"
    return enhanced if enhanced.exists() else fallback


def default_output_dir() -> Path:
    workspace_root = Path(__file__).resolve().parents[2]
    return workspace_root / "Data" / "crop_disease" / "segmented"


def kmeans_cluster_mask(image: np.ndarray, clusters: int = 3) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    pixels = lab.reshape((-1, 3)).astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.2)
    _, labels, centers = cv2.kmeans(
        pixels,
        clusters,
        None,
        criteria,
        5,
        cv2.KMEANS_PP_CENTERS,
    )

    centers = centers.astype(np.uint8)
    # Pick cluster with strongest green-like channel in BGR space after conversion.
    centers_bgr = cv2.cvtColor(centers.reshape((1, clusters, 3)), cv2.COLOR_LAB2BGR).reshape((clusters, 3))
    leaf_cluster = int(np.argmax(centers_bgr[:, 1]))

    mask = (labels.reshape(image.shape[:2]) == leaf_cluster).astype(np.uint8) * 255
    return mask


def otsu_mask(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask


def segment_image(image: np.ndarray, clusters: int = 3) -> tuple[np.ndarray, np.ndarray]:
    kmask = kmeans_cluster_mask(image, clusters=clusters)
    omask = otsu_mask(image)
    combined = cv2.bitwise_and(kmask, omask)

    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)

    segmented = cv2.bitwise_and(image, image, mask=cleaned)
    return segmented, cleaned


def run_segmentation(input_dir: Path, output_dir: Path, clusters: int) -> None:
    mask_dir = output_dir / "masks"
    output_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for class_dir in sorted([p for p in input_dir.iterdir() if p.is_dir()]):
        out_class = output_dir / class_dir.name
        out_mask_class = mask_dir / class_dir.name
        out_class.mkdir(parents=True, exist_ok=True)
        out_mask_class.mkdir(parents=True, exist_ok=True)

        for image_path in class_dir.glob("*"):
            if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue

            image = cv2.imread(str(image_path))
            if image is None:
                continue

            segmented, mask = segment_image(image, clusters=clusters)
            cv2.imwrite(str(out_class / image_path.name), segmented)
            cv2.imwrite(str(out_mask_class / image_path.name), mask)
            count += 1

    print(f"Segmented {count} images into: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="K-Means + Otsu image segmentation.")
    parser.add_argument("--input", type=Path, default=default_input_dir(), help="Input directory")
    parser.add_argument("--output", type=Path, default=default_output_dir(), help="Output directory")
    parser.add_argument("--clusters", type=int, default=3, help="Number of K-Means clusters")
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input directory not found: {args.input}")

    run_segmentation(args.input, args.output, args.clusters)


if __name__ == "__main__":
    main()
