from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr
import argparse
import shutil
import traceback
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ultralytics import YOLO


ROOT = Path(r"C:\tiny_object_detection_paper\YOLOv8n-SA-RFE")
OLD_ROOT = Path(r"C:\tiny_object_detection_paper")

OUT = ROOT / "paper_results"
RAW = OUT / "01_raw_validation"
LOGS = OUT / "02_logs"
TABLES = OUT / "03_tables_csv"
FIGS = OUT / "04_figures"
LATEX = OUT / "05_latex_tables"
ULTRA_FIGS = FIGS / "ultralytics_curves"

for p in [OUT, RAW, LOGS, TABLES, FIGS, LATEX, ULTRA_FIGS]:
    p.mkdir(parents=True, exist_ok=True)


def first_existing(paths):
    for p in paths:
        p = Path(p)
        if p.exists():
            return p
    raise FileNotFoundError("None of these paths exist:\n" + "\n".join(str(x) for x in paths))


VISDRONE_YAML = first_existing([
    ROOT / "configs" / "visdrone.yaml",
    OLD_ROOT / "configs" / "visdrone.yaml",
])

UAVDT_YAML = first_existing([
    ROOT / "configs" / "uavdt.yaml",
])

MODEL_REGISTRY = {
    "YOLOv8n": {
        "params_m": 3.008,
        "gflops": 8.1,
        "weights_visdrone": OLD_ROOT / "runs" / "detect" / "results" / "baseline" / "yolov8n_visdrone_baseline" / "weights" / "best.pt",
        "weights_uavdt": ROOT / "runs" / "uavdt_baseline" / "yolov8n_uavdt_baseline_640_b8_100ep-2" / "weights" / "best.pt",
    },
    "RIF-YOLO-N w/o RGFT": {
        "params_m": 3.061,
        "gflops": 8.8,
        "weights_visdrone": OLD_ROOT / "runs" / "detect" / "results" / "rfe_v3" / "yolov8n_rfe_v3_visdrone-2" / "weights" / "best.pt",
        "weights_uavdt": ROOT / "runs" / "uavdt_rif_yolo" / "rif_yolo_n_uavdt_640_b8_100ep" / "weights" / "best.pt",
    },
    "RIF-YOLO-N + RGFT": {
        "params_m": 3.061,
        "gflops": 8.8,
        "weights_visdrone": OLD_ROOT / "runs" / "detect" / "runs" / "detect" / "ra_rfe" / "yolov8n_ra_rfe_finetune_832_b2_50ep" / "weights" / "best.pt",
        "weights_uavdt": ROOT / "runs" / "uavdt_rif_yolo_rgft" / "rif_yolo_n_uavdt_rgft_832_b4_50ep" / "weights" / "best.pt",
    },
    "RIF-YOLO-N w/o gate": {
        "params_m": 3.026,
        "gflops": 8.4,
        "weights_visdrone": ROOT / "runs" / "rif_ablation" / "rif_ablation_nogate_640_b4_100ep" / "weights" / "best.pt",
    },
    "RIF-YOLO-N w/o identity scale": {
        "params_m": 3.034,
        "gflops": 8.5,
        "weights_visdrone": ROOT / "runs" / "rif_ablation" / "rif_ablation_noscale_640_b4_100ep" / "weights" / "best.pt",
    },
}


def load_dataset_names(data_yaml):
    with open(data_yaml, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    names = data.get("names", {})
    if isinstance(names, list):
        return {i: n for i, n in enumerate(names)}
    return {int(k): v for k, v in names.items()}


def get_dataset_root_and_split(data_yaml, split="val"):
    data_yaml = Path(data_yaml)
    with open(data_yaml, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    root = Path(data.get("path", data_yaml.parent))
    split_rel = Path(data[split])

    if split_rel.is_absolute():
        image_dir = split_rel
    else:
        image_dir = root / split_rel

    label_dir = Path(str(image_dir).replace("\\images\\", "\\labels\\").replace("/images/", "/labels/"))
    return image_dir, label_dir


def count_class_instances(data_yaml, split="val"):
    names = load_dataset_names(data_yaml)
    image_dir, label_dir = get_dataset_root_and_split(data_yaml, split)

    instance_counts = {cid: 0 for cid in names}
    image_counts = {cid: 0 for cid in names}

    if not label_dir.exists():
        return pd.DataFrame({
            "class_id": list(names.keys()),
            "class": list(names.values()),
            "images": [np.nan] * len(names),
            "instances": [np.nan] * len(names),
        })

    for label_file in label_dir.glob("*.txt"):
        present = set()
        with open(label_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cid = int(float(parts[0]))
                    if cid in instance_counts:
                        instance_counts[cid] += 1
                        present.add(cid)
        for cid in present:
            image_counts[cid] += 1

    return pd.DataFrame({
        "class_id": list(names.keys()),
        "class": [names[cid] for cid in names],
        "images": [image_counts[cid] for cid in names],
        "instances": [instance_counts[cid] for cid in names],
    })


def get_metric(metrics, key, fallback=None):
    try:
        rd = metrics.results_dict
        if key in rd:
            return float(rd[key])
    except Exception:
        pass
    return fallback


def extract_main_metrics(metrics):
    box = metrics.box
    row = {
        "P": get_metric(metrics, "metrics/precision(B)", getattr(box, "mp", np.nan)),
        "R": get_metric(metrics, "metrics/recall(B)", getattr(box, "mr", np.nan)),
        "mAP50": get_metric(metrics, "metrics/mAP50(B)", getattr(box, "map50", np.nan)),
        "mAP50-95": get_metric(metrics, "metrics/mAP50-95(B)", getattr(box, "map", np.nan)),
    }

    speed = getattr(metrics, "speed", {}) or {}
    row["preprocess_ms"] = speed.get("preprocess", np.nan)
    row["inference_ms"] = speed.get("inference", np.nan)
    row["postprocess_ms"] = speed.get("postprocess", np.nan)
    row["total_ms"] = row["preprocess_ms"] + row["inference_ms"] + row["postprocess_ms"]
    return row


def extract_classwise_metrics(metrics, data_yaml, dataset, model_name, resolution):
    names = load_dataset_names(data_yaml)
    counts = count_class_instances(data_yaml, "val")

    box = metrics.box
    ap50 = np.asarray(getattr(box, "ap50", []), dtype=float)
    ap = np.asarray(getattr(box, "ap", []), dtype=float)

    cls_idx = getattr(box, "ap_class_index", None)
    if cls_idx is None or len(cls_idx) == 0:
        cls_idx = np.arange(len(ap50))
    cls_idx = np.asarray(cls_idx, dtype=int)

    rows = []
    for i, cid in enumerate(cls_idx):
        rows.append({
            "dataset": dataset,
            "model": model_name,
            "resolution": resolution,
            "class_id": int(cid),
            "class": names.get(int(cid), str(cid)),
            "AP50": float(ap50[i]) if i < len(ap50) else np.nan,
            "AP50-95": float(ap[i]) if i < len(ap) else np.nan,
        })

    df = pd.DataFrame(rows)
    df = df.merge(counts, on=["class_id", "class"], how="left")
    return df


def run_validation_job(dataset, data_yaml, model_name, weights, resolution, batch=1):
    weights = Path(weights)
    if not weights.exists():
        raise FileNotFoundError(f"Missing checkpoint: {weights}")

    run_name = f"{dataset}_{model_name.replace(' ', '_').replace('/', '_').replace('+', 'plus')}_{resolution}"
    save_project = RAW / dataset / model_name.replace(" ", "_").replace("/", "_").replace("+", "plus")
    log_path = LOGS / f"{run_name}.txt"

    print(f"\n[VALIDATING] {dataset} | {model_name} | {resolution} | {weights}")

    with open(log_path, "w", encoding="utf-8") as logf:
        with redirect_stdout(logf), redirect_stderr(logf):
            model = YOLO(str(weights))
            metrics = model.val(
                data=str(data_yaml),
                imgsz=resolution,
                batch=batch,
                device=0,
                split="val",
                verbose=True,
                plots=True,
                project=str(save_project),
                name=str(resolution),
                exist_ok=True,
            )

    row = extract_main_metrics(metrics)
    row.update({
        "dataset": dataset,
        "model": model_name,
        "resolution": resolution,
        "weights": str(weights),
        "params_M": MODEL_REGISTRY.get(model_name, {}).get("params_m", np.nan),
        "GFLOPs": MODEL_REGISTRY.get(model_name, {}).get("gflops", np.nan),
        "save_dir": str(getattr(metrics, "save_dir", save_project / str(resolution))),
        "log_file": str(log_path),
    })

    class_df = extract_classwise_metrics(metrics, data_yaml, dataset, model_name, resolution)

    copy_ultralytics_plots(Path(row["save_dir"]), dataset, model_name, resolution)

    return row, class_df


def copy_ultralytics_plots(save_dir, dataset, model_name, resolution):
    target = ULTRA_FIGS / dataset / model_name.replace(" ", "_").replace("/", "_").replace("+", "plus") / str(resolution)
    target.mkdir(parents=True, exist_ok=True)

    for fname in [
        "confusion_matrix.png",
        "confusion_matrix_normalized.png",
        "PR_curve.png",
        "F1_curve.png",
        "P_curve.png",
        "R_curve.png",
        "labels.jpg",
    ]:
        src = save_dir / fname
        if src.exists():
            shutil.copy2(src, target / fname)


def build_jobs():
    jobs = []

    vis_resolutions = [640, 768, 832, 960, 1024]
    for model_name in ["YOLOv8n", "RIF-YOLO-N w/o RGFT", "RIF-YOLO-N + RGFT"]:
        for res in vis_resolutions:
            jobs.append({
                "dataset": "visdrone",
                "data_yaml": VISDRONE_YAML,
                "model_name": model_name,
                "weights": MODEL_REGISTRY[model_name]["weights_visdrone"],
                "resolution": res,
                "batch": 1,
            })

    for model_name in ["RIF-YOLO-N w/o gate", "RIF-YOLO-N w/o identity scale"]:
        jobs.append({
            "dataset": "visdrone",
            "data_yaml": VISDRONE_YAML,
            "model_name": model_name,
            "weights": MODEL_REGISTRY[model_name]["weights_visdrone"],
            "resolution": 1024,
            "batch": 1,
        })

    uavdt_resolutions = [640, 832, 1024]
    for model_name in ["YOLOv8n", "RIF-YOLO-N w/o RGFT", "RIF-YOLO-N + RGFT"]:
        for res in uavdt_resolutions:
            jobs.append({
                "dataset": "uavdt",
                "data_yaml": UAVDT_YAML,
                "model_name": model_name,
                "weights": MODEL_REGISTRY[model_name]["weights_uavdt"],
                "resolution": res,
                "batch": 1,
            })

    return jobs


def run_all_validations():
    rows = []
    class_rows = []

    for job in build_jobs():
        try:
            row, cdf = run_validation_job(**job)
            rows.append(row)
            class_rows.append(cdf)
        except Exception as e:
            err = {
                "dataset": job["dataset"],
                "model": job["model_name"],
                "resolution": job["resolution"],
                "error": str(e),
            }
            rows.append(err)
            with open(LOGS / f"ERROR_{job['dataset']}_{job['model_name']}_{job['resolution']}.txt", "w", encoding="utf-8") as f:
                f.write(traceback.format_exc())

    df = pd.DataFrame(rows)
    cdf = pd.concat(class_rows, ignore_index=True) if class_rows else pd.DataFrame()

    df.to_csv(TABLES / "all_validation_results.csv", index=False)
    cdf.to_csv(TABLES / "all_classwise_results.csv", index=False)

    return df, cdf


def write_special_tables(df, cdf):
    vis = df[df["dataset"].eq("visdrone")].copy()
    uav = df[df["dataset"].eq("uavdt")].copy()

    vis_main = vis[
        (vis["resolution"].eq(1024)) &
        (vis["model"].isin(["YOLOv8n", "RIF-YOLO-N + RGFT"]))
    ].copy()
    vis_main.to_csv(TABLES / "visdrone_main_results.csv", index=False)

    vis_res = vis[vis["model"].isin(["YOLOv8n", "RIF-YOLO-N w/o RGFT", "RIF-YOLO-N + RGFT"])].copy()
    vis_res.to_csv(TABLES / "visdrone_resolution_results.csv", index=False)

    ablation_models = [
        "YOLOv8n",
        "RIF-YOLO-N w/o RGFT",
        "RIF-YOLO-N w/o gate",
        "RIF-YOLO-N w/o identity scale",
        "RIF-YOLO-N + RGFT",
    ]
    ablation = vis[(vis["resolution"].eq(1024)) & (vis["model"].isin(ablation_models))].copy()
    ablation.to_csv(TABLES / "component_ablation_results.csv", index=False)

    class_vis_1024 = cdf[
        (cdf["dataset"].eq("visdrone")) &
        (cdf["resolution"].eq(1024)) &
        (cdf["model"].isin(["YOLOv8n", "RIF-YOLO-N + RGFT"]))
    ].copy()

    if not class_vis_1024.empty:
        base = class_vis_1024[class_vis_1024["model"].eq("YOLOv8n")]
        rif = class_vis_1024[class_vis_1024["model"].eq("RIF-YOLO-N + RGFT")]

        merged = base.merge(
            rif,
            on=["class_id", "class", "images", "instances"],
            suffixes=("_YOLOv8n", "_RIF")
        )
        merged["Delta_AP50"] = merged["AP50_RIF"] - merged["AP50_YOLOv8n"]
        merged["Delta_AP50-95"] = merged["AP50-95_RIF"] - merged["AP50-95_YOLOv8n"]
        merged.to_csv(TABLES / "visdrone_classwise_results.csv", index=False)

    size_df = pd.DataFrame([
        {"object_size": "Tiny", "instances": 25967, "YOLOv8n_AP50": 0.1298, "RIF_YOLO_N_AP50": 0.1346, "Gain": 0.0048},
        {"object_size": "Small", "instances": 11894, "YOLOv8n_AP50": 0.3771, "RIF_YOLO_N_AP50": 0.3852, "Gain": 0.0081},
    ])
    size_df.to_csv(TABLES / "visdrone_size_stratified_ap50.csv", index=False)

    uav.to_csv(TABLES / "uavdt_external_validation.csv", index=False)

    arch_df = pd.DataFrame([
        {"variant": "RIF-YOLO-N final", "params_M": 3.061, "GFLOPs": 8.8, "P": 0.520, "R": 0.424, "mAP50": 0.415, "mAP50-95": 0.243, "finding": "Best accuracy-complexity balance."},
        {"variant": "P2/P3/P4 without P5", "params_M": 2.099, "GFLOPs": 12.3, "P": 0.463, "R": 0.386, "mAP50": 0.360, "mAP50-95": 0.209, "finding": "Removing P5 reduced semantic context."},
        {"variant": "Heavy LEAF-style V3", "params_M": np.nan, "GFLOPs": np.nan, "P": 0.510, "R": 0.410, "mAP50": 0.404, "mAP50-95": 0.236, "finding": "Higher complexity but below final RIF-YOLO-N."},
        {"variant": "V3-Efficient", "params_M": 3.064, "GFLOPs": 8.4, "P": 0.472, "R": 0.408, "mAP50": 0.383, "mAP50-95": 0.221, "finding": "Lightweight but weaker than final design."},
    ])
    arch_df.to_csv(TABLES / "architecture_design_results.csv", index=False)

    write_latex_tables()


def write_latex_tables():
    for csv_file in TABLES.glob("*.csv"):
        try:
            df = pd.read_csv(csv_file)
            tex = df.to_latex(index=False, escape=False, float_format="%.3f")
            with open(LATEX / f"{csv_file.stem}.tex", "w", encoding="utf-8") as f:
                f.write(tex)
        except Exception:
            pass


def save_fig(fig, name):
    fig.tight_layout()
    fig.savefig(FIGS / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIGS / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_resolution_curve():
    f = TABLES / "visdrone_resolution_results.csv"
    if not f.exists():
        return
    df = pd.read_csv(f)
    df = df[df["model"].isin(["YOLOv8n", "RIF-YOLO-N w/o RGFT", "RIF-YOLO-N + RGFT"])]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for model_name, g in df.groupby("model"):
        g = g.sort_values("resolution")
        ax.plot(g["resolution"], g["mAP50"], marker="o", label=f"{model_name} mAP50")
    ax.set_xlabel("Input resolution")
    ax.set_ylabel("mAP50")
    ax.set_title("VisDrone resolution sensitivity")
    ax.grid(True, alpha=0.3)
    ax.legend()
    save_fig(fig, "visdrone_resolution_curve_map50")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for model_name, g in df.groupby("model"):
        g = g.sort_values("resolution")
        ax.plot(g["resolution"], g["mAP50-95"], marker="o", label=f"{model_name} mAP50:95")
    ax.set_xlabel("Input resolution")
    ax.set_ylabel("mAP50:95")
    ax.set_title("VisDrone localization quality across resolutions")
    ax.grid(True, alpha=0.3)
    ax.legend()
    save_fig(fig, "visdrone_resolution_curve_map5095")


def plot_main_bar():
    f = TABLES / "visdrone_main_results.csv"
    if not f.exists():
        return
    df = pd.read_csv(f).sort_values("model")
    x = np.arange(len(df))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - width / 2, df["mAP50"], width, label="mAP50")
    ax.bar(x + width / 2, df["mAP50-95"], width, label="mAP50:95")
    ax.set_xticks(x)
    ax.set_xticklabels(df["model"], rotation=15, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Main VisDrone performance at 1024")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    save_fig(fig, "visdrone_main_map_bar")


def plot_classwise_delta():
    f = TABLES / "visdrone_classwise_results.csv"
    if not f.exists():
        return
    df = pd.read_csv(f).sort_values("Delta_AP50")

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.barh(df["class"], df["Delta_AP50"])
    ax.axvline(0, linewidth=1)
    ax.set_xlabel("Delta AP50")
    ax.set_ylabel("Class")
    ax.set_title("Class-wise AP50 change: RIF-YOLO-N minus YOLOv8n")
    ax.grid(True, axis="x", alpha=0.3)
    save_fig(fig, "visdrone_classwise_delta_ap50")

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.barh(df["class"], df["Delta_AP50-95"])
    ax.axvline(0, linewidth=1)
    ax.set_xlabel("Delta AP50:95")
    ax.set_ylabel("Class")
    ax.set_title("Class-wise AP50:95 change: RIF-YOLO-N minus YOLOv8n")
    ax.grid(True, axis="x", alpha=0.3)
    save_fig(fig, "visdrone_classwise_delta_ap5095")


def plot_size_ap():
    f = TABLES / "visdrone_size_stratified_ap50.csv"
    if not f.exists():
        return
    df = pd.read_csv(f)
    x = np.arange(len(df))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - width / 2, df["YOLOv8n_AP50"], width, label="YOLOv8n")
    ax.bar(x + width / 2, df["RIF_YOLO_N_AP50"], width, label="RIF-YOLO-N")
    ax.set_xticks(x)
    ax.set_xticklabels(df["object_size"])
    ax.set_ylabel("AP50")
    ax.set_title("Size-stratified AP50 on VisDrone")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    save_fig(fig, "visdrone_size_ap50_bar")


def plot_complexity_scatter():
    f = TABLES / "component_ablation_results.csv"
    if not f.exists():
        return
    df = pd.read_csv(f)
    df = df.dropna(subset=["GFLOPs", "mAP50"])

    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.scatter(df["GFLOPs"], df["mAP50"])
    for _, r in df.iterrows():
        ax.annotate(r["model"], (r["GFLOPs"], r["mAP50"]), fontsize=8)
    ax.set_xlabel("GFLOPs")
    ax.set_ylabel("mAP50")
    ax.set_title("Accuracy-complexity trade-off on VisDrone")
    ax.grid(True, alpha=0.3)
    save_fig(fig, "visdrone_accuracy_complexity_scatter")


def plot_uavdt_external():
    f = TABLES / "uavdt_external_validation.csv"
    if not f.exists():
        return
    df = pd.read_csv(f)
    df = df[df["model"].isin(["YOLOv8n", "RIF-YOLO-N w/o RGFT", "RIF-YOLO-N + RGFT"])]
    df["label"] = df["model"] + " " + df["resolution"].astype(str)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = np.arange(len(df))
    width = 0.35
    ax.bar(x - width / 2, df["mAP50"], width, label="mAP50")
    ax.bar(x + width / 2, df["mAP50-95"], width, label="mAP50:95")
    ax.set_xticks(x)
    ax.set_xticklabels(df["label"], rotation=35, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("UAVDT external validation")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    save_fig(fig, "uavdt_external_validation_bar")


def make_figures():
    plot_main_bar()
    plot_resolution_curve()
    plot_classwise_delta()
    plot_size_ap()
    plot_complexity_scatter()
    plot_uavdt_external()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-val", action="store_true", help="Run all validation jobs and save raw outputs.")
    parser.add_argument("--make-figures", action="store_true", help="Generate CSV-derived figures.")
    parser.add_argument("--all", action="store_true", help="Run validation and generate all tables/figures.")
    args = parser.parse_args()

    if args.all:
        args.run_val = True
        args.make_figures = True

    if args.run_val:
        df, cdf = run_all_validations()
    else:
        df_path = TABLES / "all_validation_results.csv"
        cdf_path = TABLES / "all_classwise_results.csv"
        if not df_path.exists() or not cdf_path.exists():
            raise FileNotFoundError("Run with --run-val first because all_validation_results.csv does not exist.")
        df = pd.read_csv(df_path)
        cdf = pd.read_csv(cdf_path)

    write_special_tables(df, cdf)

    if args.make_figures:
        make_figures()

    print("\nDone.")
    print(f"Results saved to: {OUT}")


if __name__ == "__main__":
    main()