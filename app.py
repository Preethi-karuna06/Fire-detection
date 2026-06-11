import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
import time
from datetime import datetime
import io

st.set_page_config(
    page_title="🔥 Fire & Smoke Detection System",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&family=Share+Tech+Mono&display=swap');
    html, body, [class*="css"] { font-family: 'Rajdhani', sans-serif; }
    .stApp { background: #0a0a0f; color: #e0e0e0; }

    .main-header {
        background: linear-gradient(135deg, #1a0a00 0%, #2d0a00 50%, #1a0500 100%);
        border: 1px solid #ff4400; border-radius: 12px;
        padding: 24px 32px; margin-bottom: 24px;
        box-shadow: 0 0 40px rgba(255,68,0,0.3);
        text-align: center;
    }
    .main-header h1 {
        font-family: 'Rajdhani', sans-serif; font-size: 2.8rem; font-weight: 700;
        color: #ff6622; text-shadow: 0 0 20px rgba(255,100,34,0.8);
        margin: 0; letter-spacing: 3px; text-transform: uppercase;
    }
    .main-header p {
        color: #ff9966; font-size: 1rem; letter-spacing: 2px;
        margin-top: 8px; font-family: 'Share Tech Mono', monospace;
    }
    .stat-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 10px; padding: 20px; text-align: center; border: 1px solid #333;
    }
    .stat-card.fire  { border-color: #ff4400; box-shadow: 0 0 20px rgba(255,68,0,0.2); }
    .stat-card.smoke { border-color: #888;    box-shadow: 0 0 20px rgba(136,136,136,0.2); }
    .stat-card.safe  { border-color: #00ff88; box-shadow: 0 0 20px rgba(0,255,136,0.2); }
    .stat-number { font-size: 2.5rem; font-weight: 700; font-family: 'Share Tech Mono', monospace; }
    .stat-label  { font-size: 0.8rem; letter-spacing: 2px; text-transform: uppercase; opacity: 0.7; margin-top: 4px; }

    .alert-banner {
        padding: 16px 24px; border-radius: 8px; margin: 12px 0;
        font-weight: 600; font-size: 1.1rem; letter-spacing: 1px;
    }
    .alert-fire  { background: rgba(255,68,0,0.15);    border: 2px solid #ff4400; color: #ff6622; animation: pulse 1.5s infinite; }
    .alert-smoke { background: rgba(150,150,150,0.15); border: 2px solid #999;    color: #bbbbbb; }
    .alert-safe  { background: rgba(0,255,136,0.1);    border: 2px solid #00ff88; color: #00ff88; }

    @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.6} }

    .detection-log {
        background: #0d1117; border: 1px solid #333; border-radius: 8px;
        padding: 16px; font-family: 'Share Tech Mono', monospace;
        font-size: 0.8rem; max-height: 300px; overflow-y: auto; color: #00ff88;
    }
    .log-entry-fire  { color: #ff4400; }
    .log-entry-smoke { color: #aaaaaa; }
    .log-entry-safe  { color: #00cc66; }

    div[data-testid="stSidebar"] { background: #0d0d1a; border-right: 1px solid #222; }

    .stButton > button {
        background: linear-gradient(135deg, #cc3300, #ff6600);
        color: white; border: none; padding: 12px 28px; border-radius: 6px;
        font-family: 'Rajdhani', sans-serif; font-weight: 600;
        letter-spacing: 2px; text-transform: uppercase; width: 100%;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #ff3300, #ff8800);
        box-shadow: 0 0 20px rgba(255,100,0,0.5);
    }
    .section-title {
        font-family: 'Share Tech Mono', monospace; font-size: 0.75rem;
        letter-spacing: 3px; text-transform: uppercase; color: #ff6622;
        border-bottom: 1px solid #ff4400; padding-bottom: 8px; margin-bottom: 16px;
    }
    .method-badge {
        display: inline-block; padding: 4px 12px; border-radius: 4px;
        font-family: 'Share Tech Mono', monospace; font-size: 0.72rem; letter-spacing: 1px;
    }
    .badge-yolo { background: rgba(0,150,255,0.15); border: 1px solid #0096ff; color: #44aaff; }
    .badge-hsv  { background: rgba(255,180,0,0.15); border: 1px solid #ffb400;  color: #ffcc44; }
</style>
""", unsafe_allow_html=True)


# ─── HSV-Based Fire & Smoke Detection (works without custom model) ──────────────

def detect_fire_hsv(frame):
    """
    Detect fire regions using HSV color thresholding.
    Fire = bright red/orange/yellow hues with high saturation.
    Returns list of (x1,y1,x2,y2,confidence) bounding boxes.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Fire hue: red/orange/yellow  (H: 0-35 and 160-180 for red wraparound)
    mask1 = cv2.inRange(hsv, np.array([0,   150, 150]), np.array([35,  255, 255]))
    mask2 = cv2.inRange(hsv, np.array([160, 150, 150]), np.array([180, 255, 255]))
    fire_mask = cv2.bitwise_or(mask1, mask2)

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_CLOSE, kernel)
    fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_OPEN,  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))

    contours, _ = cv2.findContours(fire_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = frame.shape[:2]
    min_area = (h * w) * 0.002   # at least 0.2% of frame

    boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        # Confidence proxy: pixel density in bbox
        roi_mask = fire_mask[y:y+bh, x:x+bw]
        density  = np.sum(roi_mask > 0) / max(bw * bh, 1)
        conf = min(0.50 + density * 0.50, 0.99)
        boxes.append((x, y, x+bw, y+bh, conf))

    return boxes


def detect_smoke_hsv(frame):
    """
    Detect smoke regions using grayscale + low-saturation + brightness analysis.
    Smoke = low saturation, medium-high value, grayish hue.
    Returns list of (x1,y1,x2,y2,confidence) bounding boxes.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Smoke: any hue, low saturation (0–60), medium brightness (80–230)
    smoke_mask = cv2.inRange(hsv, np.array([0,  0,  80]), np.array([180, 60, 230]))

    # Exclude very green regions (trees/grass) — reduces false positives outdoors
    green_mask = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 200]))
    smoke_mask = cv2.bitwise_and(smoke_mask, cv2.bitwise_not(green_mask))

    # Exclude blue sky
    sky_mask = cv2.inRange(hsv, np.array([90, 30, 150]), np.array([130, 150, 255]))
    smoke_mask = cv2.bitwise_and(smoke_mask, cv2.bitwise_not(sky_mask))

    # Larger kernel for smoke blobs
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    smoke_mask = cv2.morphologyEx(smoke_mask, cv2.MORPH_CLOSE, kernel)
    smoke_mask = cv2.morphologyEx(smoke_mask, cv2.MORPH_OPEN,
                                  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))

    contours, _ = cv2.findContours(smoke_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = frame.shape[:2]
    min_area = (h * w) * 0.008   # smoke blobs are larger

    boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        density = np.sum(smoke_mask[y:y+bh, x:x+bw] > 0) / max(bw * bh, 1)
        conf = min(0.40 + density * 0.45, 0.95)
        boxes.append((x, y, x+bw, y+bh, conf))

    return boxes


def nms_boxes(boxes, iou_thresh=0.3):
    """Simple NMS to merge overlapping detections."""
    if not boxes:
        return []
    boxes_arr = np.array([[x1, y1, x2, y2, c] for x1, y1, x2, y2, c in boxes], dtype=float)
    x1, y1, x2, y2, scores = boxes_arr[:, 0], boxes_arr[:, 1], boxes_arr[:, 2], boxes_arr[:, 3], boxes_arr[:, 4]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep  = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou   = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[np.where(iou <= iou_thresh)[0] + 1]
    return [tuple(map(int, boxes_arr[k][:4])) + (float(boxes_arr[k][4]),) for k in keep]


# ─── YOLO Model Load ─────────────────────────────────────────────────────────────

@st.cache_resource
def load_yolo_model(model_size="yolov8n"):
    try:
        from ultralytics import YOLO
        for path in ["fire_smoke_model.pt", "best.pt"]:
            if os.path.exists(path):
                return YOLO(path), "custom"
        return YOLO(f"{model_size}.pt"), "coco"
    except Exception:
        return None, "error"


FIRE_SMOKE_CLASSES = {"fire", "smoke", "flame", "wildfire"}

def detect_with_yolo(frame, model, model_type, conf_thresh, iou_thresh):
    """Run YOLO and return only fire/smoke detections (for custom model)."""
    results = model(frame, conf=conf_thresh, iou=iou_thresh, verbose=False)
    detections = []
    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            cls_id = int(box.cls[0])
            label  = result.names.get(cls_id, "").lower()
            conf   = float(box.conf[0])
            # Custom model: use all classes; COCO model: only fire/smoke keywords
            if model_type == "coco" and not any(k in label for k in FIRE_SMOKE_CLASSES):
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append({"label": label, "confidence": conf,
                                "bbox": (x1, y1, x2, y2), "source": "yolo"})
    return detections


# ─── Main Detection Pipeline ─────────────────────────────────────────────────────

def run_detection(frame, model, model_type, conf_thresh, iou_thresh,
                  use_hsv_fire, use_hsv_smoke):
    """Combine YOLO + HSV detections and annotate frame."""
    annotated   = frame.copy()
    detections  = []
    used_method = set()

    # 1. YOLO (custom model gives best results)
    if model is not None:
        yolo_dets = detect_with_yolo(frame, model, model_type, conf_thresh, iou_thresh)
        detections.extend(yolo_dets)
        if yolo_dets:
            used_method.add("yolo")

    # 2. HSV fallback (always runs alongside; great for COCO model situation)
    if use_hsv_fire:
        fire_boxes = nms_boxes(detect_fire_hsv(frame))
        for (x1, y1, x2, y2, conf) in fire_boxes:
            if conf >= conf_thresh:
                detections.append({"label": "fire", "confidence": conf,
                                    "bbox": (x1, y1, x2, y2), "source": "hsv"})
                used_method.add("hsv")

    if use_hsv_smoke:
        smoke_boxes = nms_boxes(detect_smoke_hsv(frame))
        for (x1, y1, x2, y2, conf) in smoke_boxes:
            if conf >= conf_thresh:
                detections.append({"label": "smoke", "confidence": conf,
                                    "bbox": (x1, y1, x2, y2), "source": "hsv"})
                used_method.add("hsv")

    # 3. Draw annotations
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        label  = det["label"].lower()
        conf   = det["confidence"]
        source = det.get("source", "")

        if "fire" in label:
            color = (0, 60, 255)       # BGR → red/orange
        elif "smoke" in label:
            color = (180, 180, 180)    # gray
        else:
            color = (0, 200, 80)

        thickness = 3 if "fire" in label else 2
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

        badge   = "[YOLO]" if source == "yolo" else "[HSV]"
        txt     = f"{label.upper()} {conf:.0%} {badge}"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        # Label pill
        cv2.rectangle(annotated, (x1, y1 - th - 14), (x1 + tw + 10, y1), color, -1)
        cv2.putText(annotated, txt, (x1 + 5, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    # 4. Status bar
    h, w = annotated.shape[:2]
    ts   = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    has_fire  = any("fire"  in d["label"].lower() for d in detections)
    has_smoke = any("smoke" in d["label"].lower() for d in detections)

    if has_fire:
        status_txt, sc = "⚠ FIRE DETECTED",  (0, 60, 255)
    elif has_smoke:
        status_txt, sc = "⚠ SMOKE DETECTED", (180, 180, 180)
    else:
        status_txt, sc = "✓ ALL CLEAR",       (0, 200, 100)

    cv2.rectangle(annotated, (0, h - 36), (w, h), (15, 15, 15), -1)
    cv2.putText(annotated, ts, (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (100, 100, 100), 1, cv2.LINE_AA)
    cv2.putText(annotated, status_txt, (w - 300, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.60, sc, 2, cv2.LINE_AA)
    cv2.putText(annotated, f"DETECTIONS: {len(detections)}", (w - 200, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (80, 80, 80), 1, cv2.LINE_AA)

    return annotated, detections, used_method


# ─── Sidebar ─────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown('<div class="section-title">⚙ Configuration</div>', unsafe_allow_html=True)

    model_size     = st.selectbox("YOLO Model Size", ["yolov8n", "yolov8s", "yolov8m"], index=0)
    conf_threshold = st.slider("Confidence Threshold", 0.10, 0.95, 0.35, 0.05)
    iou_threshold  = st.slider("IoU Threshold (NMS)",  0.10, 0.95, 0.40, 0.05)

    st.markdown('<div class="section-title" style="margin-top:18px">🔬 Detection Methods</div>', unsafe_allow_html=True)
    use_hsv_fire  = st.checkbox("HSV Fire Detection",  value=True,
                                 help="Color-based fire detection — works without custom model")
    use_hsv_smoke = st.checkbox("HSV Smoke Detection", value=True,
                                 help="Color-based smoke detection")

    st.markdown('<div class="section-title" style="margin-top:18px">📡 Input Source</div>', unsafe_allow_html=True)
    input_mode = st.radio("Mode", ["📷 Image", "🎬 Video", "📹 Webcam"], label_visibility="collapsed")

    st.markdown('<div class="section-title" style="margin-top:18px">🔔 Alerts</div>', unsafe_allow_html=True)
    alert_fire  = st.checkbox("Fire Alert",  value=True)
    alert_smoke = st.checkbox("Smoke Alert", value=True)
    show_log    = st.checkbox("Detection Log", value=True)

    st.markdown("""
    <div style="font-family:'Share Tech Mono',monospace;font-size:0.7rem;color:#555;line-height:2;margin-top:16px">
    ℹ HSV mode works out of the box.<br>
    For higher accuracy, place a<br>
    trained <span style="color:#ff4400">best.pt</span> in app dir.<br><br>
    Dataset: roboflow.com<br>
    search "fire smoke detection"
    </div>""", unsafe_allow_html=True)


# ─── Header ──────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="main-header">
    <h1>🔥 Fire & Smoke Detection System</h1>
    <p>YOLOv8 + HSV COLOR ANALYSIS · REAL-TIME DETECTION ENGINE</p>
</div>
""", unsafe_allow_html=True)

# ─── Load Model ──────────────────────────────────────────────────────────────────

with st.spinner("Loading model..."):
    model, model_type = load_yolo_model(model_size)

type_label = "Custom Fire Model ✓" if model_type == "custom" else "COCO Base (HSV fills the gap)"
color_dot  = "🟢" if model_type == "custom" else "🟡"
st.markdown(f"""
<div style="font-family:'Share Tech Mono',monospace;font-size:0.8rem;
     color:#aaa;margin-bottom:18px">
{color_dot} MODEL: {model_size}.pt &nbsp;|&nbsp; TYPE: {type_label} &nbsp;|&nbsp;
<span class="method-badge badge-yolo">YOLO</span>
&nbsp;+&nbsp;
<span class="method-badge badge-hsv">HSV COLOR</span>
&nbsp;active
</div>
""", unsafe_allow_html=True)

# ─── Session State ───────────────────────────────────────────────────────────────

for k, v in [("detection_log", []), ("total_fire", 0),
             ("total_smoke", 0), ("frames_processed", 0)]:
    if k not in st.session_state:
        st.session_state[k] = v


def add_log(detections):
    ts        = datetime.now().strftime("%H:%M:%S")
    has_fire  = any("fire"  in d["label"].lower() for d in detections)
    has_smoke = any("smoke" in d["label"].lower() for d in detections)
    if has_fire:
        st.session_state.total_fire += 1
        n = sum(1 for d in detections if "fire" in d["label"].lower())
        st.session_state.detection_log.append(
            f'<span class="log-entry-fire">[{ts}] 🔥 FIRE — {n} region(s)</span>')
    if has_smoke:
        st.session_state.total_smoke += 1
        n = sum(1 for d in detections if "smoke" in d["label"].lower())
        st.session_state.detection_log.append(
            f'<span class="log-entry-smoke">[{ts}] 💨 SMOKE — {n} region(s)</span>')
    if not has_fire and not has_smoke:
        st.session_state.detection_log.append(
            f'<span class="log-entry-safe">[{ts}] ✓ Clear</span>')
    st.session_state.detection_log = st.session_state.detection_log[-60:]


# ─── Stats ───────────────────────────────────────────────────────────────────────

c1, c2, c3, c4 = st.columns(4)
c1.markdown(f'<div class="stat-card fire"><div class="stat-number" style="color:#ff4400">'
            f'{st.session_state.total_fire}</div><div class="stat-label">Fire Alerts</div></div>',
            unsafe_allow_html=True)
c2.markdown(f'<div class="stat-card smoke"><div class="stat-number" style="color:#aaa">'
            f'{st.session_state.total_smoke}</div><div class="stat-label">Smoke Alerts</div></div>',
            unsafe_allow_html=True)
c3.markdown(f'<div class="stat-card safe"><div class="stat-number" style="color:#00ff88">'
            f'{st.session_state.frames_processed}</div><div class="stat-label">Frames</div></div>',
            unsafe_allow_html=True)
status_str  = "🔴 ALERT" if st.session_state.total_fire > 0 else "🟢 MONITORING"
status_col  = "#ff4400" if st.session_state.total_fire > 0 else "#00ff88"
c4.markdown(f'<div class="stat-card" style="border-color:{status_col}"><div class="stat-number" '
            f'style="color:{status_col};font-size:1.3rem">{status_str}</div>'
            f'<div class="stat-label">System Status</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─── Main Layout ─────────────────────────────────────────────────────────────────

left, right = st.columns([3, 2])

with left:
    st.markdown('<div class="section-title">🎯 Detection Output</div>', unsafe_allow_html=True)

    # ── IMAGE ────────────────────────────────────────────────────────────────────
    if "Image" in input_mode:
        uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png", "bmp", "webp"],
                                     label_visibility="collapsed")
        if uploaded:
            file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
            frame      = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            with st.spinner("Analyzing for fire & smoke..."):
                annotated, detections, methods = run_detection(
                    frame, model, model_type,
                    conf_threshold, iou_threshold,
                    use_hsv_fire, use_hsv_smoke
                )

            st.session_state.frames_processed += 1
            add_log(detections)

            rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            st.image(rgb, caption=f"Result — {len(detections)} detection(s) via {', '.join(methods) or 'none'}",
                     use_container_width=True)

            # Download
            buf = io.BytesIO()
            Image.fromarray(rgb).save(buf, format="PNG")
            st.download_button("⬇ Download Result", buf.getvalue(),
                                "fire_detection_result.png", "image/png")

    # ── VIDEO ────────────────────────────────────────────────────────────────────
    elif "Video" in input_mode:
        uploaded_vid = st.file_uploader("Upload video", type=["mp4", "avi", "mov", "mkv"],
                                         label_visibility="collapsed")
        if uploaded_vid:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_vid.read()); tfile.flush()

            cap    = cv2.VideoCapture(tfile.name)
            nf     = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps    = cap.get(cv2.CAP_PROP_FPS) or 25
            st.info(f"📹 {nf} frames @ {fps:.1f} FPS")

            run_col, stop_col = st.columns(2)
            run_btn  = run_col.button("▶ Start")
            stop_btn = stop_col.button("⏹ Stop")

            if run_btn:
                fph  = st.empty()
                prog = st.progress(0)
                idx  = 0
                while cap.isOpened() and not stop_btn:
                    ret, frame = cap.read()
                    if not ret: break
                    annotated, detections, _ = run_detection(
                        frame, model, model_type, conf_threshold, iou_threshold,
                        use_hsv_fire, use_hsv_smoke)
                    st.session_state.frames_processed += 1
                    add_log(detections)
                    fph.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)
                    prog.progress(min(idx / max(nf, 1), 1.0))
                    idx += 1
                cap.release(); os.unlink(tfile.name)
                st.success(f"✅ Processed {idx} frames.")

    # ── WEBCAM ───────────────────────────────────────────────────────────────────
    elif "Webcam" in input_mode:
        run_cam = st.checkbox("🔴 Start Live Feed", key="cam_run")
        win     = st.image([])
        if run_cam:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("❌ Cannot access webcam.")
            else:
                while st.session_state.get("cam_run", False):
                    ret, frame = cap.read()
                    if not ret: break
                    annotated, detections, _ = run_detection(
                        frame, model, model_type, conf_threshold, iou_threshold,
                        use_hsv_fire, use_hsv_smoke)
                    st.session_state.frames_processed += 1
                    add_log(detections)
                    win.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)
                cap.release()


# ─── Right Panel ─────────────────────────────────────────────────────────────────

with right:
    st.markdown('<div class="section-title">🚨 Alert Status</div>', unsafe_allow_html=True)

    af = st.session_state.total_fire  > 0
    as_ = st.session_state.total_smoke > 0

    if af and alert_fire:
        st.markdown("""<div class="alert-banner alert-fire">
            🔥 FIRE DETECTED — IMMEDIATE ACTION<br>
            <small>Evacuate · Call emergency services</small>
        </div>""", unsafe_allow_html=True)
    elif as_ and alert_smoke:
        st.markdown("""<div class="alert-banner alert-smoke">
            💨 SMOKE DETECTED — Investigate source<br>
            <small>Check for fire · Ventilate area</small>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""<div class="alert-banner alert-safe">
            ✓ ALL CLEAR — No fire or smoke detected
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔄 Reset Counters"):
        for k in ["total_fire","total_smoke","frames_processed","detection_log"]:
            st.session_state[k] = [] if k == "detection_log" else 0
        st.rerun()

    if show_log:
        st.markdown('<div class="section-title" style="margin-top:20px">📋 Detection Log</div>',
                    unsafe_allow_html=True)
        log_html = ("<br>".join(reversed(st.session_state.detection_log[-20:]))
                    if st.session_state.detection_log
                    else '<span style="color:#444">[ Awaiting detections... ]</span>')
        st.markdown(f'<div class="detection-log">{log_html}</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title" style="margin-top:20px">ℹ Detection Methods</div>',
                unsafe_allow_html=True)
    st.markdown(f"""
    <div style="font-family:'Share Tech Mono',monospace;font-size:0.78rem;color:#888;line-height:2">
    YOLO &nbsp;: {model_size} ({model_type})<br>
    HSV fire &nbsp;: {'ON ✓' if use_hsv_fire  else 'off'}<br>
    HSV smoke: {'ON ✓' if use_hsv_smoke else 'off'}<br>
    Confidence: {conf_threshold:.0%}<br>
    Device &nbsp; : {'CUDA ⚡' if __import__('torch').cuda.is_available() else 'CPU'}
    </div>
    <div style="margin-top:12px;font-family:'Share Tech Mono',monospace;font-size:0.7rem;color:#555">
    💡 HSV detection works immediately.<br>
    &nbsp;&nbsp;&nbsp;YOLO improves with best.pt<br>
    &nbsp;&nbsp;&nbsp;(custom fire/smoke weights).
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("""
<div style="text-align:center;font-family:'Share Tech Mono',monospace;font-size:0.7rem;color:#333;padding:8px">
    FIRE & SMOKE DETECTION · YOLOv8 + HSV COLOR ANALYSIS · STREAMLIT
</div>""", unsafe_allow_html=True)