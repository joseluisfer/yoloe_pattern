import runpod
import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image
from ultralytics import YOLO

# Cargamos el modelo globalmente para que persista entre llamadas
model = YOLO("/yoloe-26x-seg.pt")

def decode_base64_image(base64_str):
    image_data = base64.b64decode(base64_str)
    image = Image.open(BytesIO(image_data)).convert('RGB')
    return np.array(image)

def handler(job):
    try:
        job_input = job['input']
        base64_scene = job_input.get('file')
        # El 'pattern' se recibe pero YOLO-E estándar no lo usa directamente para "buscar"
        # base64_pattern = job_input.get('pattern') 
        
        conf_threshold = job_input.get('conf', 0.25)
        iou_threshold = job_input.get('iou', 0.45)
        imgsz = job_input.get('imgsz', 640)

        if not base64_scene:
            return {"error": "No se recibió la imagen de la escena"}

        # 1. Preparar imagen
        scene_img = decode_base64_image(base64_scene)

        # 2. Inferencia con YOLO-E
        results = model.predict(
            source=scene_img,
            conf=conf_threshold,
            iou=iou_threshold,
            imgsz=imgsz,
            save=False
        )

        # 3. Formatear resultados para el Android (JSONArray de bboxes)
        detections = []
        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()  # Coordenadas [x1, y1, x2, y2]
            scores = r.boxes.conf.cpu().numpy()
            
            for i in range(len(boxes)):
                detections.append({
                    "bbox": [
                        float(boxes[i][0]), # x1
                        float(boxes[i][1]), # y1
                        float(boxes[i][2]), # x2
                        float(boxes[i][3])  # y2
                    ],
                    "confidence": float(scores[i])
                })

        return {
            "predictions": detections,
            "count": len(detections)
        }

    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
