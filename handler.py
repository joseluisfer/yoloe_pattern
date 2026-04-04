import runpod
import torch
import numpy as np
from ultralytics import YOLOWorld
from PIL import Image
import io
import base64
import requests

# Carga del modelo
device = "0" if torch.cuda.is_available() else "cpu"
model = YOLOWorld("yoloe-26x-seg.pt").to(device)

def get_image(source):
    """Detecta si es Base64 o URL y devuelve una PIL Image"""
    if source.startswith('http'):
        response = requests.get(source, timeout=10)
        return Image.open(io.BytesIO(response.content)).convert("RGB")
    else:
        if "," in source:
            source = source.split(",")[1]
        img_bytes = base64.b64decode(source)
        return Image.open(io.BytesIO(img_bytes)).convert("RGB")

def handler(job):
    try:
        data = job.get("input", {})
        
        # Mapeo exacto a tus campos de Java/Kotlin
        base64_scene = data.get("file")
        base64_pattern = data.get("pattern")
        conf_thresh = float(data.get("conf", 0.25))
        iou_thresh = float(data.get("iou", 0.45))
        img_size = int(data.get("imgsz", 1280))

        if not base64_scene or not base64_pattern:
            return {"error": "Faltan 'file' o 'pattern'"}

        # 1. Cargar imágenes
        img_scene = get_image(base64_scene)
        img_pattern = get_image(base64_pattern)

        # 2. VISUAL PROMPTING: Usar la imagen como clase
        # Importante: YOLO-World v8.4 acepta una lista de imágenes para set_classes
        model.set_classes([img_pattern])

        # 3. Inferencia
        results = model.predict(
            np.array(img_scene),
            conf=conf_thresh,
            iou=iou_thresh,
            imgsz=img_size,
            verbose=False
        )

        # 4. Formatear salida
        detections = []
        if results and len(results) > 0:
            r = results[0]
            for i in range(len(r.boxes)):
                box = r.boxes[i]
                d = {
                    "confidence": float(box.conf[0]),
                    "bbox": [float(x) for x in box.xyxy[0].tolist()]
                }
                if r.masks:
                    # Devolvemos los puntos de la segmentación (normalizados 0-1)
                    d["segments"] = r.masks[i].xyn[0].tolist()
                detections.append(d)

        return {"detections": detections}

    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
