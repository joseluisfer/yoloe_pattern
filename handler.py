import runpod
import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image

def decode_base64_image(base64_str):
    image_data = base64.b64decode(base64_str)
    image = Image.open(BytesIO(image_data))
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

def handler(job):
    try:
        job_input = job['input']
        
        # 1. Obtener parámetros del JSON enviado por Android
        base64_scene = job_input.get('file')
        base64_pattern = job_input.get('pattern')
        threshold = job_input.get('conf', 0.5)
        
        if not base64_scene or not base64_pattern:
            return {"error": "Faltan imágenes de escena o patrón"}

        # 2. Decodificar imágenes
        scene_img = decode_base64_image(base64_scene)
        pattern_img = decode_base64_image(base64_pattern)
        
        h, w = pattern_img.shape[:2]

        # 3. Procesar: Template Matching (Detección de patrón)
        # Nota: Para algo más avanzado usaríamos descriptores SIFT/ORB o YOLO-World
        res = cv2.matchTemplate(scene_img, pattern_img, cv2.TM_CCOEFF_NORMED)
        
        # Filtrar por confianza (threshold)
        loc = np.where(res >= threshold)
        
        detections = []
        # Agrupar puntos cercanos para evitar múltiples rectángulos en el mismo objeto
        # (Simplificado para este ejemplo)
        for pt in zip(*loc[::-1]):
            detections.append({
                "bbox": [
                    int(pt[0]),           # x1
                    int(pt[1]),           # y1
                    int(pt[0] + w),       # x2
                    int(pt[1] + h)        # y2
                ],
                "confidence": float(res[pt[1], pt[0]])
            })

        # Tu App Android busca "predictions" o "detections" dentro de "output"
        return {
            "predictions": detections,
            "count": len(detections)
        }

    except Exception as e:
        return {"error": str(e)}

# Iniciar el servidor de RunPod
runpod.serverless.start({"handler": handler})
