"""Convert images into FER-2013 rows and append to a CSV."""
import argparse
from pathlib import Path

from fer import append_images_to_fer2013_csv, image_to_fer2013_row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert images to FER-2013-style CSV rows.")
    parser.add_argument("--image", required=True, help="Single image to preview as a FER-2013 row.")
    parser.add_argument("--images", nargs="*", help="Additional images to append to the output CSV.")
    parser.add_argument("--output", default="synthetic_fer.csv", help="Destination CSV file for appended rows.")
    parser.add_argument("--emotion", type=int, default=0, help="Emotion label index (0-6) for the rows.")
    parser.add_argument("--usage", default="PrivateTest", help="Usage split label for the rows (e.g., PrivateTest).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preview_image = Path(args.image)
    if not preview_image.is_file():
        raise SystemExit(f"Preview image not found: {preview_image}")

    if args.images:
        missing = [str(path) for path in args.images if not Path(path).is_file()]
        if missing:
            raise SystemExit(f"Additional image(s) not found: {', '.join(missing)}")

    row = image_to_fer2013_row(str(preview_image), emotion=args.emotion, usage=args.usage)
    print("Single-row preview:")
    print(row)
    if args.images:
        updated = append_images_to_fer2013_csv(args.images, args.output, emotion=args.emotion, usage=args.usage)
        print(f"\nAppended {len(args.images)} image(s) to {args.output} (total rows: {len(updated)})")


if __name__ == "__main__":
    main()
