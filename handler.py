import runpod
import torch
from ultralytics import YOLOWorld # YOLOE usa la arquitectura World para prompts
import numpy as np
from PIL import Image
import io
import base64

# Carga global del modelo para persistencia
print("Cargando YOLOE-26x-seg para Visual Prompting...")
device = "0" if torch.cuda.is_available() else "cpu"
# Cargamos como YOLOWorld porque soporta .query() para imágenes
model = YOLOWorld("yoloe-26x-seg.pt").to(device)

def decode_image(b64_str):
    if "," in b64_str:
        b64_str = b64_str.split(",")[1]
    image_bytes = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")

def handler(job):
    try:
        job_input = job.get("input", {})
        
        # 1. Obtener entradas del JSON (según tu código Java/Kotlin)
        base64_scene = job_input.get("file")
        base64_pattern = job_input.get("pattern")
        
        # Parámetros con valores por defecto
        conf_val = float(job_input.get("conf", 0.25))
        iou_val = float(job_input.get("iou", 0.45))
        img_size = int(job_input.get("imgsz", 1280))

        if not base64_scene or not base64_pattern:
            return {"error": "Se requieren 'file' (escena) y 'pattern' (imagen a buscar)"}

        # 2. Preparar imágenes
        scene_img = decode_image(base64_scene)
        pattern_img = decode_image(base64_pattern)

        # 3. VISUAL PROMPTING 
        # Seteamos el 'prompt' usando la imagen patrón
        model.set_classes([pattern_img]) 

        # 4. Inferencia en la escena
        results = model.predict(
            np.array(scene_img),
            conf=conf_val,
            iou=iou_val,
            imgsz=img_size,
            verbose=False
        )

        # 5. Procesar resultados (Detecciones + Segmentación si existe)
        detections = []
        if results and len(results) > 0:
            res = results[0]
            if res.boxes:
                for i in range(len(res.boxes)):
                    box = res.boxes[i]
                    det = {
                        "confidence": round(float(box.conf[0]), 4),
                        "bbox": [round(float(x), 2) for x in box.xyxy[0].tolist()],
                    }
                    # Si el modelo devuelve máscaras de segmentación
                    if res.masks:
                        det["segments"] = res.masks[i].xyn[0].tolist() # Coordenadas normalizadas
                    
                    detections.append(det)

        return {
            "status": "success",
            "detections": detections,
            "count": len(detections)
        }

    except Exception as e:
        return {"error": f"Error en el worker: {str(e)}"}

runpod.serverless.start({"handler": handler})
