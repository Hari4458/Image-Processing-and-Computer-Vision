from pathlib import Path
import argparse
import cv2


def default_input_dir() -> Path:
    workspace_root = Path(__file__).resolve().parents[2]
    return workspace_root / "Data" / "crop_disease" / "raw"


def default_output_dir() -> Path:
    workspace_root = Path(__file__).resolve().parents[2]
    return workspace_root / "Data" / "crop_disease" / "processed"


def resize_dataset(input_dir: Path, output_dir: Path, size: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    image_count = 0

    for class_dir in sorted([p for p in input_dir.iterdir() if p.is_dir()]):
        target_class_dir = output_dir / class_dir.name
        target_class_dir.mkdir(parents=True, exist_ok=True)

        for image_path in class_dir.glob("*"):
            if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue

            image = cv2.imread(str(image_path))
            if image is None:
                continue

            resized = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(target_class_dir / image_path.name), resized)
            image_count += 1

    print(f"Resized {image_count} images into: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Resize PlantVillage dataset images.")
    parser.add_argument("--input", type=Path, default=default_input_dir(), help="Input dataset directory")
    parser.add_argument("--output", type=Path, default=default_output_dir(), help="Output directory")
    parser.add_argument("--size", type=int, default=224, help="Target image size")
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input directory not found: {args.input}")

    resize_dataset(args.input, args.output, args.size)


if __name__ == "__main__":
    main()
