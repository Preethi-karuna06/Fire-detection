"""
train_fire_model.py
────────────────────────────────────────────────────────────────────────────────
Train a custom YOLOv8 model on a fire & smoke dataset.

QUICK START:
    1. Download a fire/smoke dataset from Roboflow Universe:
       https://universe.roboflow.com/search?q=fire+smoke&t=object-detection
       Export in YOLOv8 format → saves to ./fire_smoke_dataset/

    2. Update DATA_YAML to point to your dataset.yaml

    3. Run:  python train_fire_model.py

OUTPUT:
    runs/detect/fire_smoke_vN/weights/best.pt  ← copy this to your app dir
"""

from ultralytics import YOLO
import os

# ── Config ──────────────────────────────────────────────────────────────────────

DATA_YAML   = "fire_smoke_dataset/data.yaml"   # path to your dataset.yaml
BASE_MODEL  = "yolov8n.pt"                      # start from nano (fast); use yolov8s.pt for more accuracy
EPOCHS      = 100
IMGSZ       = 640
BATCH       = 16
WORKERS     = 4
PROJECT     = "runs/detect"
RUN_NAME    = "fire_smoke_v1"
DEVICE      = 0                                 # 0 = first GPU; 'cpu' for CPU-only


# ── Sample data.yaml format ─────────────────────────────────────────────────────
SAMPLE_YAML = """
# data.yaml — place this in your dataset folder
path: ./fire_smoke_dataset      # root dir
train: images/train
val:   images/val
test:  images/test               # optional

nc: 2                            # number of classes
names:
  0: fire
  1: smoke
"""

# ── Train ────────────────────────────────────────────────────────────────────────

def train():
    if not os.path.exists(DATA_YAML):
        print("=" * 60)
        print("❌  Dataset not found:", DATA_YAML)
        print()
        print("📦  To get a fire/smoke dataset:")
        print("    1. Visit  https://universe.roboflow.com")
        print("    2. Search  'fire smoke detection'")
        print("    3. Export  YOLOv8 format → fire_smoke_dataset/")
        print()
        print("📄  Expected data.yaml structure:")
        print(SAMPLE_YAML)
        print("=" * 60)
        return

    print("🔥  Loading base model:", BASE_MODEL)
    model = YOLO(BASE_MODEL)

    print("🚀  Starting training ...")
    results = model.train(
        data     = DATA_YAML,
        epochs   = EPOCHS,
        imgsz    = IMGSZ,
        batch    = BATCH,
        workers  = WORKERS,
        project  = PROJECT,
        name     = RUN_NAME,
        device   = DEVICE,
        patience = 20,           # early stopping
        save     = True,
        plots    = True,
        val      = True,
        verbose  = True,
    )

    best_path = f"{PROJECT}/{RUN_NAME}/weights/best.pt"
    print()
    print("✅  Training complete!")
    print(f"📂  Best model → {best_path}")
    print()
    print("📋  Next step: copy best.pt to your app directory")
    print("    cp", best_path, "best.pt")

    # Quick validation
    print()
    print("📊  Running validation ...")
    metrics = model.val(data=DATA_YAML)
    print(f"    mAP50   : {metrics.box.map50:.3f}")
    print(f"    mAP50-95: {metrics.box.map:.3f}")


# ── Evaluate a saved model ────────────────────────────────────────────────────

def evaluate(model_path="best.pt"):
    """Evaluate an existing model on the validation set."""
    model = YOLO(model_path)
    metrics = model.val(data=DATA_YAML)
    print("mAP50     :", metrics.box.map50)
    print("mAP50-95  :", metrics.box.map)
    print("Precision :", metrics.box.mp)
    print("Recall    :", metrics.box.mr)


# ── Export ────────────────────────────────────────────────────────────────────

def export(model_path="best.pt", fmt="onnx"):
    """Export model to ONNX / TensorRT / CoreML etc."""
    model = YOLO(model_path)
    model.export(format=fmt)
    print(f"✅  Exported to {fmt}")


if __name__ == "__main__":
    train()
