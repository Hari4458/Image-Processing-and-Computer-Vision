from pathlib import Path
import argparse
import cv2


def default_input_dir() -> Path:
    workspace_root = Path(__file__).resolve().parents[2]
    return workspace_root / "Data" / "crop_disease" / "processed"


def default_output_dir() -> Path:
    workspace_root = Path(__file__).resolve().parents[2]
    return workspace_root / "Data" / "crop_disease" / "processed_denoised"


def denoise_dataset(input_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
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

            denoised = cv2.fastNlMeansDenoisingColored(image, None, 7, 7, 7, 21)
            cv2.imwrite(str(target_dir / image_path.name), denoised)
            count += 1

    print(f"Denoised {count} images into: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Denoise dataset images.")
    parser.add_argument("--input", type=Path, default=default_input_dir(), help="Input directory")
    parser.add_argument("--output", type=Path, default=default_output_dir(), help="Output directory")
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input directory not found: {args.input}")

    denoise_dataset(args.input, args.output)


if __name__ == "__main__":
    main()
