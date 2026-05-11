from pathlib import Path
import argparse
import cv2
import joblib
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops


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
    return [float(np.mean(graycoprops(glcm, p))) for p in props]


def extract_features(image: np.ndarray) -> list[float]:
    return color_moments(image) + glcm_features(image)


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
    centers_bgr = cv2.cvtColor(centers.reshape((1, clusters, 3)), cv2.COLOR_LAB2BGR).reshape((clusters, 3))
    leaf_cluster = int(np.argmax(centers_bgr[:, 1]))

    return (labels.reshape(image.shape[:2]) == leaf_cluster).astype(np.uint8) * 255


def otsu_mask(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask


def segment_for_inference(image: np.ndarray) -> np.ndarray:
    kmask = kmeans_cluster_mask(image, clusters=3)
    omask = otsu_mask(image)
    combined = cv2.bitwise_and(kmask, omask)

    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)

    return cv2.bitwise_and(image, image, mask=cleaned)


def default_model_path() -> Path:
    project_root = Path(__file__).resolve().parents[1]
    return project_root / "runs" / "classify" / "svm_model.joblib"


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict disease class for one image.")
    parser.add_argument("--image", type=Path, required=True, help="Input image path")
    parser.add_argument("--model", type=Path, default=default_model_path(), help="Model path")
    args = parser.parse_args()

    if not args.image.exists():
        raise FileNotFoundError(f"Image not found: {args.image}")
    if not args.model.exists():
        raise FileNotFoundError(f"Model not found: {args.model}")

    image = cv2.imread(str(args.image))
    if image is None:
        raise RuntimeError(f"Could not read image: {args.image}")

    # Match training-time preprocessing: predict on segmented image, not raw image.
    image = segment_for_inference(image)

    model_bundle = joblib.load(args.model)
    pipeline = model_bundle["pipeline"]
    feature_cols = model_bundle.get("feature_cols")

    feats = np.array(extract_features(image), dtype=np.float32).reshape(1, -1)
    if feature_cols:
        feat_input = pd.DataFrame(feats, columns=feature_cols)
    else:
        feat_input = feats

    pred = pipeline.predict(feat_input)[0]
    probs = pipeline.predict_proba(feat_input)[0]
    best = float(np.max(probs))

    print("\n" + "="*70)
    print("  CROP DISEASE DETECTION AND CLASSIFICATION FOR FARMERS")
    print("="*70)
    print(f"\nImage: {args.image.name}")
    print(f"\n✓ Detection Result:")
    print(f"  Disease Detected:  {pred}")
    print(f"  Confidence Level:  {best:.2%}")
    if best < 0.3:
        print(f"  Status:            LOW confidence - consult expert for verification")
    elif best < 0.6:
        print(f"  Status:            MODERATE confidence - review recommended")
    else:
        print(f"  Status:            HIGH confidence - likely accurate")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
