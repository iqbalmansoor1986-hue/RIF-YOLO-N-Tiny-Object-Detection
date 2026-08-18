from pathlib import Path
import math
import random
import cv2
import yaml
import numpy as np
from ultralytics import YOLO

# ============================================================
# SETTINGS
# ============================================================

DATA_YAML = Path(r"configs\vedai.yaml")

# Final baseline checkpoint
YOLO_MODEL = Path(
    r"runs\vedai_all_overnight_20260816_012434\baseline"
    r"\yolov8n_vedai_640_b4_100ep\weights\best.pt"
)

# Final selected RIF-YOLO-N + RGFT checkpoint at 1024
RIF_MODEL = Path(
    r"runs\vedai_rgft_1024_20260816_133510\train"
    r"\rif_yolo_n_vedai_rgft_1024_b1_60ep\weights\best.pt"
)

OUT_DIR = Path(r"runs\vedai_visuals")
VAL_OUT_YOLO = OUT_DIR / "yolo_val_1024"
VAL_OUT_RIF = OUT_DIR / "rif_rgft_val_1024"
QUAL_OUT = OUT_DIR / "qualitative"

IMGSZ = 1024
CONF_THRES = 0.25
IOU_THRES = 0.7
NUM_IMAGES = 8          # number of validation images in the grid
GRID_COLS = 3
PANEL_SIZE = 520        # resize each panel for the final collage
SEED = 42


# ============================================================
# HELPERS
# ============================================================

def check_paths():
    required = [DATA_YAML, YOLO_MODEL, RIF_MODEL]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required file(s):\n" + "\n".join(missing)
        )

def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def resolve_split_images(data_yaml_path, split_key="val"):
    data = load_yaml(data_yaml_path)
    root = Path(data.get("path", "."))

    split_value = data[split_key]

    def resolve_one(item):
        p = Path(item)
        if not p.is_absolute():
            p = root / p
        return p

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    image_paths = []

    if isinstance(split_value, str):
        p = resolve_one(split_value)

        if p.is_dir():
            image_paths = sorted(
                [x for x in p.rglob("*") if x.suffix.lower() in image_extensions]
            )
        elif p.is_file():
            if p.suffix.lower() == ".txt":
                lines = [x.strip() for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
                for line in lines:
                    q = Path(line)
                    if not q.is_absolute():
                        q = root / q
                    image_paths.append(q)
            else:
                image_paths = [p]
        else:
            raise FileNotFoundError(f"Could not resolve val split path: {p}")

    elif isinstance(split_value, list):
        for item in split_value:
            p = resolve_one(item)
            if p.is_dir():
                image_paths.extend(
                    [x for x in p.rglob("*") if x.suffix.lower() in image_extensions]
                )
            elif p.is_file():
                if p.suffix.lower() == ".txt":
                    lines = [x.strip() for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
                    for line in lines:
                        q = Path(line)
                        if not q.is_absolute():
                            q = root / q
                        image_paths.append(q)
                else:
                    image_paths.append(p)

        image_paths = sorted(image_paths)

    else:
        raise ValueError(f"Unsupported split specification in {data_yaml_path}")

    if not image_paths:
        raise RuntimeError("No validation images found.")

    return image_paths

def choose_images(image_paths, n=8, seed=42):
    rng = random.Random(seed)
    if len(image_paths) <= n:
        return image_paths
    return sorted(rng.sample(image_paths, n))

def add_filename_overlay(img, filename):
    out = img.copy()
    text = filename
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.8
    thickness = 2
    x, y = 10, 30

    # black outline
    cv2.putText(out, text, (x, y), font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    # white text
    cv2.putText(out, text, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

    return out

def render_prediction(model, img_path, imgsz=1024, conf=0.25, iou=0.7, show_conf=False):
    results = model.predict(
        source=str(img_path),
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        verbose=False,
        save=False
    )[0]

    plotted = results.plot(
        labels=True,
        conf=show_conf,
        boxes=True
    )

    plotted = add_filename_overlay(plotted, img_path.name)
    return plotted

def build_grid(images, out_file, cols=3, panel_size=520):
    resized = [cv2.resize(im, (panel_size, panel_size)) for im in images]

    rows = math.ceil(len(resized) / cols)
    total = rows * cols

    if len(resized) < total:
        blank = np.full((panel_size, panel_size, 3), 230, dtype=np.uint8)
        for _ in range(total - len(resized)):
            resized.append(blank.copy())

    row_imgs = []
    for r in range(rows):
        row = resized[r * cols:(r + 1) * cols]
        row_imgs.append(np.hstack(row))

    grid = np.vstack(row_imgs)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_file), grid)
    print(f"Saved: {out_file}")

def run_validation_plots(model_path, project_dir, run_name):
    model = YOLO(str(model_path))
    print(f"\nRunning validation plots for: {model_path}")

    model.val(
        data=str(DATA_YAML),
        imgsz=IMGSZ,
        split="val",
        conf=0.001,
        iou=0.7,
        plots=True,
        save_json=False,
        project=str(project_dir),
        name=run_name,
        exist_ok=True,
        verbose=True
    )

def make_qualitative_grids(model_path, image_list, prefix):
    model = YOLO(str(model_path))

    imgs_labels_only = []
    imgs_with_conf = []

    for img_path in image_list:
        img1 = render_prediction(
            model=model,
            img_path=img_path,
            imgsz=IMGSZ,
            conf=CONF_THRES,
            iou=IOU_THRES,
            show_conf=False
        )
        img2 = render_prediction(
            model=model,
            img_path=img_path,
            imgsz=IMGSZ,
            conf=CONF_THRES,
            iou=IOU_THRES,
            show_conf=True
        )
        imgs_labels_only.append(img1)
        imgs_with_conf.append(img2)

    build_grid(
        imgs_labels_only,
        QUAL_OUT / f"{prefix}_labels_only.jpg",
        cols=GRID_COLS,
        panel_size=PANEL_SIZE
    )

    build_grid(
        imgs_with_conf,
        QUAL_OUT / f"{prefix}_with_conf.jpg",
        cols=GRID_COLS,
        panel_size=PANEL_SIZE
    )

def save_selected_list(image_list):
    QUAL_OUT.mkdir(parents=True, exist_ok=True)
    out_txt = QUAL_OUT / "selected_val_images.txt"
    with open(out_txt, "w", encoding="utf-8") as f:
        for p in image_list:
            f.write(str(p) + "\n")
    print(f"Saved: {out_txt}")

# ============================================================
# MAIN
# ============================================================

def main():
    check_paths()

    print("==========================================")
    print("VEDAI VISUAL GENERATION")
    print("==========================================")
    print("Data YAML :", DATA_YAML)
    print("YOLO model:", YOLO_MODEL)
    print("RIF model :", RIF_MODEL)
    print("Image size:", IMGSZ)

    # 1) Validation plots (includes confusion matrix, PR/F1 curves, etc.)
    run_validation_plots(YOLO_MODEL, VAL_OUT_YOLO, "val")
    run_validation_plots(RIF_MODEL, VAL_OUT_RIF, "val")

    # 2) Same validation images for both models
    val_images = resolve_split_images(DATA_YAML, split_key="val")
    selected = choose_images(val_images, n=NUM_IMAGES, seed=SEED)
    save_selected_list(selected)

    # 3) Qualitative grids
    make_qualitative_grids(YOLO_MODEL, selected, "yolov8n")
    make_qualitative_grids(RIF_MODEL, selected, "rif_yolo_rgft")

    print("\n==========================================")
    print("DONE")
    print("==========================================")
    print("Validation plots:")
    print(" ", VAL_OUT_YOLO / "val")
    print(" ", VAL_OUT_RIF / "val")
    print("Qualitative grids:")
    print(" ", QUAL_OUT)


if __name__ == "__main__":
    main()