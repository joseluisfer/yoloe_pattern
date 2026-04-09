import runpod
import numpy as np
import base64
import cv2
import os
from ultralytics import YOLO
from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

# Cargar el modelo YOLO26 (recomendado) o YOLO11
model = YOLO("yoloe-26x-seg.pt")

def process_input(data):
    """Detecta si el input es una ruta de archivo o base64 y lo procesa."""
    if isinstance(data, str):
        # Si es una ruta de archivo local existente (.jpg, .png, etc.)
        if os.path.exists(data):
            return data
        # Si es base64, decodificar a imagen de OpenCV (numpy array)
        try:
            nparr = np.frombuffer(base64.b64decode(data), np.uint8)
            return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except Exception:
            return data
    return data

def handler(job):
    input_data = job["input"]
    
    # Procesar imagen principal y de referencia
    img = process_input(input_data["file"])
    ref_img = process_input(input_data["pattern"])
    
    # Obtener dimensiones para el prompt visual (si ref_img es array)
    if isinstance(ref_img, np.ndarray):
        h, w = ref_img.shape[:2]
    else:
        # Si es ruta, cargamos temporalmente para obtener dimensiones
        temp_img = cv2.imread(ref_img)
        h, w = temp_img.shape[:2]

    # Definir visual_prompts según la referencia de YOLOE
    # https://docs.ultralytics.com/reference/models/yolo/model/
    visual_prompts = dict(
        bboxes=np.array([[0, 0, w, h]]), 
        cls=np.array([0])
    )

    results = model.predict(
        source=img,
        refer_image=ref_img,
        visual_prompts=visual_prompts,
        predictor=YOLOEVPSegPredictor,
        conf=input_data.get("conf", 0.25),
        iou=input_data.get("iou", 0.45),
        imgsz=input_data.get("imgsz", 1280)
    )

    # Formatear resultados
    detections = []
    for res in results[0].summary():
        detections.append({
            "name": res["name"],
            "bbox": res["box"],
            "confidence": res["confidence"]
        })

    return {"status": "success", "detections": detections}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
