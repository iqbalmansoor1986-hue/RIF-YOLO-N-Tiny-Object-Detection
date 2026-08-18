from pathlib import Path
from datetime import datetime
import csv

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
DATA_YAML = ROOT / "configs" / "vedai.yaml"

BASE_RIF_BEST = (
    ROOT
    / "runs"
    / "vedai_all_overnight_20260816_012434"
    / "rif"
    / "rif_yolo_n_vedai_640_b4_100ep"
    / "weights"
    / "best.pt"
)

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_ROOT = ROOT / "runs" / f"vedai_rgft_1024_{RUN_ID}"
TRAIN_NAME = "rif_yolo_n_vedai_rgft_1024_b1_60ep"

EPOCHS = 60
IMGSZ = 1024
BATCH = 4
DEVICE = 0
WORKERS = 0
SEED = 42

TEST_RESOLUTIONS = [640, 832, 1024]


def check_files():
    required = {
        "VEDAI dataset YAML": DATA_YAML,
        "Base RIF checkpoint": BASE_RIF_BEST,
    }

    missing = []

    print("\n=== Checking required files ===")

    for name, path in required.items():
        if path.exists():
            print(f"[OK] {name}: {path}")
        else:
            print(f"[MISSING] {name}: {path}")
            missing.append(path)

    if missing:
        raise FileNotFoundError(
            "\nRequired file(s) are missing. Check the paths printed above."
        )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"[OK] Output root: {OUTPUT_ROOT}")


def train_rgft_1024():
    print("\n" + "=" * 76)
    print("VEDAI RGFT-1024 CONTROLLED TRAINING")
    print("=" * 76)
    print(f"Initialization : {BASE_RIF_BEST}")
    print(f"Target imgsz   : {IMGSZ}")
    print(f"Epochs         : {EPOCHS}")
    print(f"Batch          : {BATCH}")
    print("Optimizer      : AdamW")
    print("lr0            : 0.001")
    print("Mosaic         : 0.0")
    print("Scale          : 0.1")
    print("Translate      : 0.05")
    print("=" * 76)

    model = YOLO(str(BASE_RIF_BEST))
    model.info(verbose=True)

    model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,

        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,

        mosaic=0.0,
        scale=0.1,
        translate=0.05,

        hsv_h=0.015,
        hsv_s=0.4,
        hsv_v=0.3,
        fliplr=0.5,

        multi_scale=0.0,

        workers=WORKERS,
        cache=False,
        amp=True,
        device=DEVICE,

        seed=SEED,
        deterministic=True,

        val=True,
        patience=60,
        save=True,
        save_period=10,
        plots=True,
        verbose=True,

        project=str(OUTPUT_ROOT / "train"),
        name=TRAIN_NAME,
        exist_ok=True,
    )

    best_pt = OUTPUT_ROOT / "train" / TRAIN_NAME / "weights" / "best.pt"

    if not best_pt.exists():
        raise FileNotFoundError(
            f"\nTraining completed but best.pt was not found:\n{best_pt}"
        )

    print("\n[OK] RGFT-1024 training completed.")
    print(f"[OK] Best checkpoint: {best_pt}")

    return best_pt


def evaluate_checkpoint(best_pt):
    print("\n" + "=" * 76)
    print("RGFT-1024 TEST-SET RESOLUTION EVALUATION")
    print("=" * 76)

    rows = []
    model = YOLO(str(best_pt))

    for imgsz in TEST_RESOLUTIONS:
        print("\n" + "-" * 76)
        print(f"Evaluating RGFT-1024 checkpoint at {imgsz} x {imgsz}")
        print("-" * 76)

        metrics = model.val(
            data=str(DATA_YAML),
            split="test",
            imgsz=imgsz,
            batch=1,
            workers=0,
            device=DEVICE,
            verbose=True,
            plots=False,
            project=str(OUTPUT_ROOT / "test_evaluation"),
            name=f"rgft_1024_eval_{imgsz}",
        )

        precision = float(metrics.box.mp)
        recall = float(metrics.box.mr)
        map50 = float(metrics.box.map50)
        map5095 = float(metrics.box.map)

        rows.append({
            "Training target": "1024 x 1024",
            "Evaluation resolution": f"{imgsz} x {imgsz}",
            "P": precision,
            "R": recall,
            "mAP50": map50,
            "mAP50-95": map5095,
        })

        print(
            f"\nSUMMARY @ {imgsz}: "
            f"P={precision:.3f}, "
            f"R={recall:.3f}, "
            f"mAP50={map50:.3f}, "
            f"mAP50-95={map5095:.3f}"
        )

    csv_path = OUTPUT_ROOT / "rgft_1024_resolution_results.csv"
    txt_path = OUTPUT_ROOT / "rgft_1024_resolution_results.txt"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "Training target",
                "Evaluation resolution",
                "P",
                "R",
                "mAP50",
                "mAP50-95",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    with txt_path.open("w", encoding="utf-8") as f:
        f.write(
            f"{'Training target':<20}"
            f"{'Eval resolution':<20}"
            f"{'P':>8}"
            f"{'R':>8}"
            f"{'mAP50':>10}"
            f"{'mAP50-95':>12}\n"
        )
        f.write("-" * 78 + "\n")

        for row in rows:
            f.write(
                f"{row['Training target']:<20}"
                f"{row['Evaluation resolution']:<20}"
                f"{row['P']:>8.3f}"
                f"{row['R']:>8.3f}"
                f"{row['mAP50']:>10.3f}"
                f"{row['mAP50-95']:>12.3f}\n"
            )

    print("\n" + "=" * 76)
    print("FINAL RGFT-1024 RESOLUTION RESULTS")
    print("=" * 76)

    print(
        f"{'Eval resolution':<20}"
        f"{'P':>8}"
        f"{'R':>8}"
        f"{'mAP50':>10}"
        f"{'mAP50-95':>12}"
    )
    print("-" * 58)

    for row in rows:
        print(
            f"{row['Evaluation resolution']:<20}"
            f"{row['P']:>8.3f}"
            f"{row['R']:>8.3f}"
            f"{row['mAP50']:>10.3f}"
            f"{row['mAP50-95']:>12.3f}"
        )

    print(f"\nSaved CSV: {csv_path}")
    print(f"Saved TXT: {txt_path}")


def main():
    check_files()
    best_pt = train_rgft_1024()
    evaluate_checkpoint(best_pt)


if __name__ == "__main__":
    main()
