from pathlib import Path
from collections import Counter
from PIL import Image
import argparse
import json
import shutil


# ============================================================
# VEDAI 1024 -> YOLO converter
#
# Uses:
#   - visible-light *_co.png images only
#   - Fold 01 test as final TEST set
#   - Fold 02 test as VALIDATION set
#   - remaining Fold 01 training images as TRAIN set
#
# Converts VEDAI oriented quadrilateral annotations to
# standard axis-aligned YOLO bounding boxes.
# ============================================================


ROOT = Path(__file__).resolve().parents[1]

RAW_ROOT = (
    ROOT
    / "datasets"
    / "VEDAI_raw"
    / "extracted"
)

IMAGE_DIR = RAW_ROOT / "Vehicules1024"
ANNOTATION_DIR = RAW_ROOT / "Annotations1024"

OUTPUT_ROOT = ROOT / "datasets" / "VEDAI-yolo"
CONFIG_DIR = ROOT / "configs"
CONFIG_FILE = CONFIG_DIR / "vedai.yaml"


# ============================================================
# Standard nine-class VEDAI mapping
# ============================================================

RAW_TO_YOLO = {
    1: 0,    # car
    2: 1,    # truck
    4: 2,    # tractor
    5: 3,    # camping car
    9: 4,    # van
    10: 5,   # other / vehicle
    11: 6,   # pickup
    23: 7,   # boat / ship
    31: 8,   # plane
}

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

# Rare raw IDs outside the standard 9-class protocol.
IGNORED_RAW_IDS = {7, 8}


# ============================================================
# Utility functions
# ============================================================

def read_split(path):
    """Read VEDAI fold file into a set of image IDs."""

    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")

    return {
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip()
    }


def validate_sources():
    """Verify that required VEDAI files exist."""

    print("\n=== Checking VEDAI source files ===")

    required = [
        IMAGE_DIR,
        ANNOTATION_DIR,
        ANNOTATION_DIR / "fold01.txt",
        ANNOTATION_DIR / "fold01test.txt",
        ANNOTATION_DIR / "fold02test.txt",
    ]

    missing = [p for p in required if not p.exists()]

    if missing:
        raise FileNotFoundError(
            "Missing required file(s):\n"
            + "\n".join(str(p) for p in missing)
        )

    co_count = len(list(IMAGE_DIR.glob("*_co.png")))
    ir_count = len(list(IMAGE_DIR.glob("*_ir.png")))

    print(f"[OK] Visible-light images : {co_count}")
    print(f"[OK] Infrared images      : {ir_count}")
    print(f"[OK] Annotation directory : {ANNOTATION_DIR}")


def prepare_output(overwrite=False):
    """Create clean YOLO output directory."""

    if OUTPUT_ROOT.exists():
        if not overwrite:
            raise FileExistsError(
                f"\nOutput directory already exists:\n{OUTPUT_ROOT}\n\n"
                "Run again with --overwrite if you want to recreate it."
            )

        print(f"Removing existing output: {OUTPUT_ROOT}")
        shutil.rmtree(OUTPUT_ROOT)

    for split in ["train", "val", "test"]:
        (OUTPUT_ROOT / "images" / split).mkdir(
            parents=True,
            exist_ok=True
        )

        (OUTPUT_ROOT / "labels" / split).mkdir(
            parents=True,
            exist_ok=True
        )

    CONFIG_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Split construction
# ============================================================

def build_splits():
    """
    Construct disjoint train/val/test sets.

    Official Fold 01:
        fold01.txt      = training pool
        fold01test.txt  = held-out Fold 01 test set

    We use Fold 02 test samples as validation because they
    belong to the Fold 01 training pool.

    Final:
        TRAIN = fold01 train - fold02 test
        VAL   = fold02 test
        TEST  = fold01 test
    """

    fold01_train = read_split(
        ANNOTATION_DIR / "fold01.txt"
    )

    fold01_test = read_split(
        ANNOTATION_DIR / "fold01test.txt"
    )

    fold02_test = read_split(
        ANNOTATION_DIR / "fold02test.txt"
    )

    # Fold02 test should normally be contained in Fold01 train.
    validation_ids = fold02_test & fold01_train

    train_ids = fold01_train - validation_ids
    test_ids = fold01_test

    print("\n=== Dataset split ===")
    print(f"Fold01 training pool : {len(fold01_train)}")
    print(f"Validation           : {len(validation_ids)}")
    print(f"Final training       : {len(train_ids)}")
    print(f"Test                 : {len(test_ids)}")

    # Ensure complete separation.
    assert train_ids.isdisjoint(validation_ids)
    assert train_ids.isdisjoint(test_ids)
    assert validation_ids.isdisjoint(test_ids)

    print("[OK] Train/validation/test sets are disjoint.")

    if len(validation_ids) != len(fold02_test):
        print(
            "[WARNING] Some Fold02 test IDs were not found "
            "inside Fold01 training."
        )

    return {
        "train": sorted(train_ids),
        "val": sorted(validation_ids),
        "test": sorted(test_ids),
    }


# ============================================================
# Annotation conversion
# ============================================================

def convert_annotation(annotation_path, image_width, image_height):
    """
    Convert one VEDAI annotation file to YOLO format.

    Individual VEDAI annotation structure:

        x_center
        y_center
        orientation
        class_id
        flag1
        flag2
        x1 x2 x3 x4
        y1 y2 y3 y4

    The final eight values describe the four oriented
    bounding-box corners.

    We convert the oriented polygon to its enclosing
    axis-aligned bounding box.
    """

    labels = []
    class_counter = Counter()
    ignored_counter = Counter()
    malformed = 0

    if not annotation_path.exists():
        # Valid negative image.
        return labels, class_counter, ignored_counter, malformed

    lines = annotation_path.read_text(
        errors="ignore"
    ).splitlines()

    for line in lines:
        line = line.strip()

        if not line:
            continue

        parts = line.split()

        if len(parts) < 14:
            malformed += 1
            continue

        try:
            raw_class = int(float(parts[3]))

            # Four X polygon coordinates.
            xs = [
                float(parts[6]),
                float(parts[7]),
                float(parts[8]),
                float(parts[9]),
            ]

            # Four Y polygon coordinates.
            ys = [
                float(parts[10]),
                float(parts[11]),
                float(parts[12]),
                float(parts[13]),
            ]

        except (ValueError, IndexError):
            malformed += 1
            continue

        # Ignore rare IDs outside standard benchmark.
        if raw_class in IGNORED_RAW_IDS:
            ignored_counter[raw_class] += 1
            continue

        if raw_class not in RAW_TO_YOLO:
            ignored_counter[raw_class] += 1
            continue

        # Axis-aligned box surrounding the quadrilateral.
        xmin = max(0.0, min(xs))
        xmax = min(float(image_width), max(xs))

        ymin = max(0.0, min(ys))
        ymax = min(float(image_height), max(ys))

        box_width = xmax - xmin
        box_height = ymax - ymin

        if box_width <= 0 or box_height <= 0:
            malformed += 1
            continue

        # Convert to normalized YOLO coordinates.
        x_center = ((xmin + xmax) / 2.0) / image_width
        y_center = ((ymin + ymax) / 2.0) / image_height
        width = box_width / image_width
        height = box_height / image_height

        # Safety clipping.
        x_center = min(max(x_center, 0.0), 1.0)
        y_center = min(max(y_center, 0.0), 1.0)
        width = min(max(width, 0.0), 1.0)
        height = min(max(height, 0.0), 1.0)

        yolo_class = RAW_TO_YOLO[raw_class]

        labels.append(
            f"{yolo_class} "
            f"{x_center:.6f} "
            f"{y_center:.6f} "
            f"{width:.6f} "
            f"{height:.6f}"
        )

        class_counter[yolo_class] += 1

    return labels, class_counter, ignored_counter, malformed


# ============================================================
# Split conversion
# ============================================================

def convert_split(split_name, image_ids):
    """Copy images and generate YOLO labels."""

    output_images = OUTPUT_ROOT / "images" / split_name
    output_labels = OUTPUT_ROOT / "labels" / split_name

    stats = {
        "images": 0,
        "objects": 0,
        "negative_images": 0,
        "missing_annotations": 0,
        "ignored_objects": 0,
        "malformed_objects": 0,
    }

    class_counts = Counter()
    ignored_counts = Counter()

    print(f"\n=== Converting {split_name} ===")

    for index, image_id in enumerate(image_ids, start=1):

        source_image = IMAGE_DIR / f"{image_id}_co.png"
        source_annotation = ANNOTATION_DIR / f"{image_id}.txt"

        if not source_image.exists():
            raise FileNotFoundError(
                f"Visible-light image not found: {source_image}"
            )

        destination_image = (
            output_images / f"{image_id}_co.png"
        )

        destination_label = (
            output_labels / f"{image_id}_co.txt"
        )

        # Get actual dimensions rather than assuming 1024.
        with Image.open(source_image) as image:
            width, height = image.size

        labels, counts, ignored, malformed = convert_annotation(
            source_annotation,
            width,
            height,
        )

        if not source_annotation.exists():
            stats["missing_annotations"] += 1

        if not labels:
            stats["negative_images"] += 1

        # Copy RGB/color image.
        shutil.copy2(
            source_image,
            destination_image
        )

        # Write YOLO label file.
        destination_label.write_text(
            "\n".join(labels),
            encoding="utf-8",
        )

        stats["images"] += 1
        stats["objects"] += len(labels)
        stats["malformed_objects"] += malformed
        stats["ignored_objects"] += sum(ignored.values())

        class_counts.update(counts)
        ignored_counts.update(ignored)

        if index % 100 == 0 or index == len(image_ids):
            print(
                f"{split_name}: "
                f"{index}/{len(image_ids)} images"
            )

    print(f"\n{split_name.upper()} statistics")
    print(f"Images              : {stats['images']}")
    print(f"Objects             : {stats['objects']}")
    print(f"Negative images     : {stats['negative_images']}")
    print(f"Missing annotations : {stats['missing_annotations']}")
    print(f"Ignored objects     : {stats['ignored_objects']}")
    print(f"Malformed objects   : {stats['malformed_objects']}")

    print("\nClass distribution:")

    for class_id, class_name in enumerate(CLASS_NAMES):
        print(
            f"  {class_id}: "
            f"{class_name:<15} "
            f"{class_counts[class_id]}"
        )

    if ignored_counts:
        print("\nIgnored raw IDs:")

        for raw_id, count in sorted(ignored_counts.items()):
            print(f"  raw ID {raw_id}: {count}")

    return {
        "statistics": stats,
        "class_counts": {
            CLASS_NAMES[class_id]: class_counts[class_id]
            for class_id in range(len(CLASS_NAMES))
        },
        "ignored_raw_ids": dict(ignored_counts),
    }


# ============================================================
# YAML generation
# ============================================================

def create_yaml():
    """Create Ultralytics dataset YAML."""

    dataset_path = OUTPUT_ROOT.as_posix()

    yaml_text = f"""# VEDAI visible-light dataset
# Converted from VEDAI 1024 annotations
# RGB/color (_co.png) images only

path: {dataset_path}

train: images/train
val: images/val
test: images/test

nc: 9

names:
  0: car
  1: truck
  2: tractor
  3: camping-car
  4: van
  5: other
  6: pickup
  7: boat
  8: plane
"""

    CONFIG_FILE.write_text(
        yaml_text,
        encoding="utf-8",
    )

    print(f"\n[OK] Dataset YAML created:\n{CONFIG_FILE}")


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Convert VEDAI 1024 visible-light images "
            "to YOLO detection format."
        )
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete and recreate an existing VEDAI-yolo directory.",
    )

    args = parser.parse_args()

    print("\n==========================================")
    print("VEDAI 1024 -> YOLO CONVERSION")
    print("Visible-light (_co.png) images only")
    print("==========================================")

    validate_sources()
    prepare_output(overwrite=args.overwrite)

    splits = build_splits()

    summary = {}

    for split_name in ["train", "val", "test"]:
        summary[split_name] = convert_split(
            split_name,
            splits[split_name],
        )

    create_yaml()

    summary["classes"] = {
        i: name
        for i, name in enumerate(CLASS_NAMES)
    }

    summary["raw_to_yolo"] = RAW_TO_YOLO

    summary["ignored_raw_ids"] = sorted(IGNORED_RAW_IDS)

    summary_file = OUTPUT_ROOT / "conversion_summary.json"

    summary_file.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"\n[OK] Conversion summary saved:\n"
        f"{summary_file}"
    )

    print("\n==========================================")
    print("VEDAI CONVERSION COMPLETED SUCCESSFULLY")
    print("==========================================")

    print(f"\nDataset directory:\n{OUTPUT_ROOT}")
    print(f"\nDataset YAML:\n{CONFIG_FILE}")


if __name__ == "__main__":
    main()