from pathlib import Path
from PIL import Image, ImageDraw
import random


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "datasets" / "VEDAI-yolo"

CLASS_NAMES = [
    "car",
    "truck",
    "tractor",
    "camping-car",
    "van",
    "other",
    "pickup",
    "boat",
    "plane",
]

OUTPUT = ROOT / "runs" / "vedai_label_check"
OUTPUT.mkdir(parents=True, exist_ok=True)


def draw_sample(split="train", n=8):
    image_dir = DATASET / "images" / split
    label_dir = DATASET / "labels" / split

    images = list(image_dir.glob("*.png"))

    random.seed(42)
    selected = random.sample(images, min(n, len(images)))

    for image_path in selected:
        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)

        w, h = image.size

        label_path = label_dir / f"{image_path.stem}.txt"

        if label_path.exists():
            for line in label_path.read_text().splitlines():
                if not line.strip():
                    continue

                cls_id, xc, yc, bw, bh = map(float, line.split())

                cls_id = int(cls_id)

                x1 = (xc - bw / 2) * w
                y1 = (yc - bh / 2) * h
                x2 = (xc + bw / 2) * w
                y2 = (yc + bh / 2) * h

                draw.rectangle(
                    [x1, y1, x2, y2],
                    outline="red",
                    width=3,
                )

                draw.text(
                    (x1, max(0, y1 - 15)),
                    CLASS_NAMES[cls_id],
                    fill="red",
                )

        output_path = OUTPUT / f"{split}_{image_path.name}"
        image.save(output_path)

        print("Saved:", output_path)


if __name__ == "__main__":
    draw_sample("train", 8)
    draw_sample("val", 4)
    draw_sample("test", 4)