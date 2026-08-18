import argparse
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Train YOLO model.")
    parser.add_argument("--model", type=str, required=True, help="Model YAML or weights path.")
    parser.add_argument("--data", type=str, required=True, help="Dataset YAML path.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--project", type=str, default="runs/detect")
    parser.add_argument("--pretrained", type=str, default="yolov8n.pt")

    args = parser.parse_args()

    model = YOLO(args.model)

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        seed=args.seed,
        deterministic=True,
        pretrained=args.pretrained,
        project=args.project,
        name=args.name,
    )


if __name__ == "__main__":
    main()