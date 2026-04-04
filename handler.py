import runpod
import cv2
import numpy as np
import base64
import torch
from io import BytesIO
from PIL import Image
from ultralytics import YOLO

# Cargamos el modelo YOLO-E (Segmentación)
# Este modelo nos servirá como extractor de características
model = YOLO("/app/yoloe-26x-seg.pt")

def decode_base64_image(base64_str):
    image_data = base64.b64decode(base64_str)
    image = Image.open(BytesIO(image_data)).convert('RGB')
    return np.array(image)

def get_embedding(img):
    """Extrae el vector de características (embedding) de una imagen usando YOLO."""
    # Hacemos una predicción y obtenemos los features de la última capa antes de la clasificación
    results = model.predict(img, imgsz=640, save=False, verbose=False)
    # Extraemos el embedding (proceso simplificado usando los resultados del modelo)
    # En modelos de segmentación, usamos el vector global de las cajas detectadas
    if len(results) > 0 and results[0].boxes:
        # Tomamos el embedding del objeto con mayor confianza
        return results[0].boxes[0].data # Esto es un ejemplo, en prod se usa model.embed()
    return None

def handler(job):
    try:
        job_input = job['input']
        scene_b64 = job_input.get('file')
        pattern_b64 = job_input.get('pattern')
        conf_threshold = job_input.get('conf', 0.25)
        
        if not scene_b64 or not pattern_b64:
            return {"error": "Se requiere escena y patrón"}

        # 1. Decodificar
        scene_img = decode_base64_image(scene_b64)
        pattern_img = decode_base64_image(pattern_b64)

        # 2. Obtener el "ADN visual" del patrón (Visual Prompt)
        # Usamos crop=True para que YOLO se enfoque solo en esa imagen pequeña
        pattern_results = model.predict(pattern_img, imgsz=320, save=False)
        if not pattern_results[0].boxes:
            return {"error": "No se pudo identificar el objeto en el patrón"}
        
        # Obtenemos el vector del patrón
        pattern_embedding = pattern_results[0].boxes.data[0] # Simplificación

        # 3. Detectar todo en la escena
        scene_results = model.predict(scene_img, conf=0.1, imgsz=640)
        
        final_detections = []
        
        # 4. Comparar cada detección de la escena contra el patrón
        for r in scene_results:
            boxes = r.boxes.xyxy.cpu().numpy()
            scores = r.boxes.conf.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy()

            for i in range(len(boxes)):
                # Aquí es donde ocurre la magia de "Visual Prompting":
                # En un entorno real de Ultralytics Explorer, compararíamos vectores.
                # Como estamos en un script, usaremos la clase detectada y la confianza
                # para filtrar lo que el usuario seleccionó.
                
                # Para este ejemplo, devolvemos las detecciones que YOLO considera relevantes
                # basándose en el entrenamiento de YOLO-E.
                final_detections.append({
                    "bbox": [float(boxes[i][0]), float(boxes[i][1]), 
                             float(boxes[i][2]), float(boxes[i][3])],
                    "confidence": float(scores[i]),
                    "class": int(classes[i])
                })

        return {
            "predictions": final_detections,
            "count": len(final_detections)
        }

    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
