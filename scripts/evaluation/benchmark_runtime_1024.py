import time
import torch
import numpy as np
from ultralytics import YOLO

DEVICE = "cuda:0"
IMGSZ = 1024
WARMUP = 50
REPEATS = 200

BASELINE = (
    r"runs\uavdt_baseline\yolov8n_uavdt_baseline_640_b8_100ep-2"
    r"\weights\best.pt"
)

RIF = (
    r"runs\uavdt_rif_yolo_rgft"
    r"\rif_yolo_n_uavdt_rgft_832_b4_50ep"
    r"\weights\best.pt"
)


def benchmark(name, checkpoint):
    print(f"\n{'=' * 70}")
    print(name)
    print(f"{'=' * 70}")

    yolo = YOLO(checkpoint)
    model = yolo.model.to(DEVICE)
    model.eval()

    x = torch.randn(
        1, 3, IMGSZ, IMGSZ,
        device=DEVICE,
        dtype=torch.float32
    )

    # -------------------------
    # GPU warm-up
    # -------------------------
    with torch.inference_mode():
        for _ in range(WARMUP):
            _ = model(x)

    torch.cuda.synchronize()

    # -------------------------
    # Timed inference
    # -------------------------
    times = []

    with torch.inference_mode():
        for _ in range(REPEATS):

            torch.cuda.synchronize()
            start = time.perf_counter()

            _ = model(x)

            torch.cuda.synchronize()
            end = time.perf_counter()

            times.append((end - start) * 1000.0)

    times = np.asarray(times)

    mean_ms = times.mean()
    std_ms = times.std(ddof=1)
    median_ms = np.median(times)
    min_ms = times.min()
    max_ms = times.max()

    fps = 1000.0 / mean_ms

    print(f"Resolution     : {IMGSZ}x{IMGSZ}")
    print(f"Warm-up        : {WARMUP}")
    print(f"Timed runs     : {REPEATS}")
    print(f"Mean latency   : {mean_ms:.3f} ms/image")
    print(f"Std. deviation : {std_ms:.3f} ms")
    print(f"Median latency : {median_ms:.3f} ms/image")
    print(f"Minimum        : {min_ms:.3f} ms")
    print(f"Maximum        : {max_ms:.3f} ms")
    print(f"Throughput     : {fps:.2f} FPS")

    return {
        "mean": mean_ms,
        "std": std_ms,
        "median": median_ms,
        "fps": fps
    }


if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU is not available.")

print("GPU:", torch.cuda.get_device_name(0))

baseline = benchmark(
    "YOLOv8n",
    BASELINE
)

rif = benchmark(
    "RIF-YOLO-N + RGFT",
    RIF
)

overhead_ms = rif["mean"] - baseline["mean"]

overhead_pct = (
    overhead_ms / baseline["mean"]
) * 100.0

print("\n" + "=" * 70)
print("FINAL COMPARISON")
print("=" * 70)

print(
    f"YOLOv8n       : "
    f"{baseline['mean']:.3f} +/- "
    f"{baseline['std']:.3f} ms "
    f"({baseline['fps']:.2f} FPS)"
)

print(
    f"RIF+RGFT      : "
    f"{rif['mean']:.3f} +/- "
    f"{rif['std']:.3f} ms "
    f"({rif['fps']:.2f} FPS)"
)

print(
    f"Latency change: "
    f"{overhead_ms:+.3f} ms "
    f"({overhead_pct:+.2f}%)"
)