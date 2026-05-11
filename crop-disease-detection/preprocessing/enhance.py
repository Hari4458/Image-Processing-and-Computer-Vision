from pathlib import Path
import argparse
import cv2


def default_input_dir() -> Path:
    workspace_root = Path(__file__).resolve().parents[2]
    return workspace_root / "Data" / "crop_disease" / "processed_denoised"


def default_output_dir() -> Path:
    workspace_root = Path(__file__).resolve().parents[2]
    return workspace_root / "Data" / "crop_disease" / "processed_enhanced"


def enhance_dataset(input_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    count = 0

    for class_dir in sorted([p for p in input_dir.iterdir() if p.is_dir()]):
        target_dir = output_dir / class_dir.name
        target_dir.mkdir(parents=True, exist_ok=True)

        for image_path in class_dir.glob("*"):
            if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue

            image = cv2.imread(str(image_path))
            if image is None:
                continue

            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l = clahe.apply(l)
            enhanced = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
            cv2.imwrite(str(target_dir / image_path.name), enhanced)
            count += 1

    print(f"Enhanced {count} images into: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Enhance dataset images with CLAHE.")
    parser.add_argument("--input", type=Path, default=default_input_dir(), help="Input directory")
    parser.add_argument("--output", type=Path, default=default_output_dir(), help="Output directory")
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input directory not found: {args.input}")

    enhance_dataset(args.input, args.output)


if __name__ == "__main__":
    main()
