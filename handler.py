import runpod
import base64
import numpy as np
import cv2
from ultralytics import YOLO

# -------------------------
# Load model
# -------------------------
model = YOLO("/models/yoloe-26x-seg.pt")

# -------------------------
# Utils
# -------------------------
def load_base64_file(b64_string):
    file_bytes = base64.b64decode(b64_string)
    np_arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return img

# -------------------------
# Handler
# -------------------------
def handler(event):
    try:
        data = event["input"]

        file_b64 = data.get("file")
        pattern = data.get("pattern", "")

        conf = float(data.get("conf", 0.25))
        iou = float(data.get("iou", 0.45))
        imgsz = int(data.get("imgsz", 1024))

        if not file_b64:
            return {"error": "file (base64) required"}

        if not pattern:
            return {"error": "pattern required"}

        # -------------------------
        # Load image
        # -------------------------
        img = load_base64_file(file_b64)

        # -------------------------
        # TEXT PROMPT
        # -------------------------
        classes = [p.strip() for p in pattern.split(",") if p.strip()]
        model.set_classes(classes)

        # -------------------------
        # Inference
        # -------------------------
        results = model.predict(
            source=img,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            verbose=False
        )

        result = results[0]

        detections = []

        if result.boxes is not None:
            for box in result.boxes:
                xyxy = box.xyxy[0].tolist()
                confidence = float(box.conf[0])
                cls = int(box.cls[0])

                detections.append({
                    "bbox": xyxy,
                    "confidence": confidence,
                    "label": result.names.get(cls, "unknown")
                })

        return {
            "detections": detections,
            "count": len(detections)
        }

    except Exception as e:
        return {"error": str(e)}


runpod.serverless.start({"handler": handler})
