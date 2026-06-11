# 🔥 Fire & Smoke Detection System
### YOLOv8 + Streamlit · Real-Time Detection

---

## 📁 Project Structure

```
fire_smoke_detection/
├── app.py                  ← Main Streamlit application
├── train_fire_model.py     ← Custom model training script
├── requirements.txt        ← Python dependencies
├── best.pt                 ← (optional) Your trained fire/smoke model
└── README.md
```

---

## ⚡ Quick Start

### 1 — Install dependencies

```bash
pip install -r requirements.txt
```

> **GPU users:** install the CUDA version of PyTorch first:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
> ```

### 2 — Run the app

```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## 🔬 Modes

| Mode | Description |
|------|-------------|
| 📷 Image | Upload a single image for detection |
| 🎬 Video | Upload a video file (frame-by-frame analysis) |
| 📹 Webcam | Live detection from your webcam |

---

## 🎯 Using a Custom Fire/Smoke Model (Recommended)

The app defaults to **YOLOv8n** (COCO, 80 classes). For accurate
fire/smoke detection, train or download a specialized model:

### Option A — Download a pre-trained model

1. Visit [Roboflow Universe](https://universe.roboflow.com/search?q=fire+smoke&t=object-detection)
2. Find a fire/smoke model and download `best.pt`
3. Place `best.pt` in the project directory

### Option B — Train your own model

```bash
# 1. Download fire/smoke dataset from Roboflow (YOLOv8 format)
#    https://universe.roboflow.com → search "fire smoke detection"
#    Export → YOLOv8 → fire_smoke_dataset/

# 2. Train
python train_fire_model.py

# 3. Copy best weights to app dir
cp runs/detect/fire_smoke_v1/weights/best.pt best.pt

# 4. Restart the app
streamlit run app.py
```

### Option C — Popular public datasets

| Dataset | Link |
|---------|------|
| Fire Detection | https://universe.roboflow.com/firedetection-6scop/fire-detection-r8unt |
| Wildfire Smoke | https://universe.roboflow.com/ericgcc/wildfire-smoke-detection |
| Fire & Smoke v2 | https://universe.roboflow.com/fire-detection-fhkox/fire-and-smoke-2 |

---

## ⚙️ Configuration

All parameters are adjustable in the **sidebar**:

| Parameter | Description | Default |
|-----------|-------------|---------|
| Model Size | yolov8n → yolov8l (speed vs accuracy) | yolov8n |
| Confidence Threshold | Min confidence for detection | 0.40 |
| IoU Threshold | Non-max suppression overlap | 0.45 |
| Fire/Smoke Alerts | Toggle alert banners | ON |

---

## 🏗 Architecture

```
Input (Image / Video / Webcam)
        │
        ▼
  Preprocessing (OpenCV)
        │
        ▼
  YOLOv8 Inference
        │
        ▼
  Post-processing (NMS + label filter)
        │
        ▼
  Annotation (bounding boxes + HUD)
        │
        ▼
  Streamlit Display + Alert System
```

---

## 📊 Performance (approximate)

| Model | Speed (CPU) | Speed (GPU) | mAP (fire dataset) |
|-------|-------------|-------------|---------------------|
| YOLOv8n | ~30ms/frame | ~5ms/frame | ~85% |
| YOLOv8s | ~50ms/frame | ~8ms/frame | ~88% |
| YOLOv8m | ~100ms/frame | ~12ms/frame | ~91% |

*Actual mAP depends on your training dataset.*

---

## 🚀 Deployment

### Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

```bash
docker build -t fire-detection .
docker run -p 8501:8501 fire-detection
```

### Streamlit Community Cloud

1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Deploy from your repo

---

## 📄 License

MIT License — free to use and modify.
