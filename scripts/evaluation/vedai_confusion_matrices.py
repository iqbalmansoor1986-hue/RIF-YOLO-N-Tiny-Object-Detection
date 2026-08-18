from ultralytics import YOLO

DATA = r"configs\vedai.yaml"

YOLO_CKPT = (
    r"runs\vedai_all_overnight_20260816_012434\baseline"
    r"\yolov8n_vedai_640_b4_100ep\weights\best.pt"
)

RIF_CKPT = (
    r"runs\vedai_rgft_1024_20260816_133510\train"
    r"\rif_yolo_n_vedai_rgft_1024_b1_60ep\weights\best.pt"
)


def validate(name, checkpoint):

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    model = YOLO(checkpoint)

    results = model.val(
        data=DATA,
        split="test",
        imgsz=1024,
        batch=1,
        device=0,
        workers=0,
        plots=True,
        project=r"runs\vedai_visuals",
        name=name,
        exist_ok=True,
        verbose=True,
    )

    print("\nSaved to:")
    print(results.save_dir)


validate(
    "YOLOv8n_1024",
    YOLO_CKPT
)

validate(
    "RIF_YOLO_N_RGFT_1024",
    RIF_CKPT
)