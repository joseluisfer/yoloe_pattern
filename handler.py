import runpod
import numpy as np
import base64
import cv2
import requests
from ultralytics import YOLO
from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

# Carga del modelo (una sola vez)
model = YOLO("yoloe-26x-seg.pt")

# -------------------------
# Loader (base64 o URL)
# -------------------------
def load_image(data):
    if not isinstance(data, str):
        raise ValueError("Input must be string")

    # URL
    if data.startswith("http"):
        resp = requests.get(data)
        img_array = np.frombuffer(resp.content, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    else:
        # limpiar base64 tipo data:image/jpeg;base64,...
        if data.startswith("data:image"):
            data = data.split(",")[1]

        img_bytes = base64.b64decode(data)
        img_array = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Image decode failed")

    return img

# -------------------------
# Handler
# -------------------------
def handler(job):
    try:
        input_data = job["input"]

        # 🔹 imagen grande + patrón visual
        img = load_image(input_data["file"])
        ref_img = load_image(input_data["pattern"])

        # 🔹 dimensiones del patrón
        h, w = ref_img.shape[:2]

        # 🔹 visual prompt (imagen completa como referencia)
        visual_prompts = dict(
            bboxes=np.array([[0, 0, w, h]]),
            cls=np.array([0])
        )

        # 🔹 inferencia
        results = model.predict(
            source=img,
            refer_image=ref_img,
            visual_prompts=visual_prompts,
            predictor=YOLOEVPSegPredictor,
            conf=input_data.get("conf", 0.10),   # 👈 clave para etiquetas
            iou=input_data.get("iou", 0.45),
            imgsz=input_data.get("imgsz", 1536)  # 👈 clave para objetos pequeños
        )

        detections = []

        if len(results) > 0:
            for res in results[0].summary():
                detections.append({
                    "bbox": res["box"],
                    "confidence": res["confidence"]
                })

        return {
            "status": "success",
            "count": len(detections),
            "detections": detections
        }

    except Exception as e:
        return {"error": str(e)}

# -------------------------
# RunPod
# -------------------------
runpod.serverless.start({"handler": handler})
