import runpod
import torch
from ultralytics import YOLOE
import numpy as np
from PIL import Image
import io
import base64
import cv2

# --- 1. CARGA GLOBAL DEL MODELO ---
print("🚀 Iniciando YOLOE-26x-seg (Visual Prompting Mode)...")
try:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Cargamos el modelo X (Extra Large) que es el más potente para segmentación
    model = YOLOE("yoloe-26x-seg.pt").to(device)
    print(f"✅ Modelo cargado en: {device}")
except Exception as e:
    print(f"❌ Error crítico en carga de modelo: {e}")

def decode_base64_to_image(b64_str):
    """Limpia y decodifica el string base64 a una imagen PIL"""
    if "," in b64_str:
        b64_str = b64_str.split(",")[1]
    img_bytes = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")

def handler(job):
    """
    Handler para detección por patrón visual y segmentación de contornos
    """
    try:
        job_input = job.get("input", {})
        
        # Extraer imagen principal y el patrón (crop)
        scene_b64 = job_input.get("file")      # La foto de la escena
        pattern_b64 = job_input.get("pattern") # La foto del objeto a buscar
        conf_threshold = job_input.get("threshold", 0.25)

        if not scene_b64 or not pattern_b64:
            return {"error": "Se requieren los campos 'file' y 'pattern' en base64"}

        # 2. Decodificar imágenes
        img_scene = decode_base64_to_image(scene_b64)
        img_pattern = decode_base64_to_image(pattern_b64)

        # 3. Inferencia con Visual Prompt
        # En YOLOE-26x, pasamos el patrón directamente en el argumento 'visual_prompt'
        results = model.predict(
            source=img_scene, 
            visual_prompt=img_pattern, 
            conf=conf_threshold,
            verbose=False
        )

        detections = []
        if results and len(results) > 0:
            res = results[0]
            
            # 4. Procesar Boxes y Segmentación (Contornos)
            if res.boxes:
                boxes = res.boxes.xyxy.cpu().numpy()
                confs = res.boxes.conf.cpu().numpy()
                
                # Extraer máscaras si existen (YOLOE-seg)
                masks = res.masks.xy if res.masks is not None else [None] * len(boxes)

                for i in range(len(boxes)):
                    # Simplificación del contorno para aligerar el JSON
                    contour_data = []
                    if masks[i] is not None:
                        # Reducimos puntos con approxPolyDP para eficiencia en Android
                        points = masks[i].astype(np.float32)
                        epsilon = 0.001 * cv2.arcLength(points, True)
                        approx = cv2.approxPolyDP(points, epsilon, True)
                        contour_data = [{"x": round(float(p[0][0]), 2), "y": round(float(p[0][1]), 2)} for p in approx]

                    detections.append({
                        "bbox": [round(float(x), 2) for x in boxes[i].tolist()],
                        "confidence": round(float(confs[i]), 4),
                        "contour": contour_data
                    })

        return {
            "status": "success",
            "total": len(detections),
            "detections": detections
        }

    except Exception as e:
        return {"error": f"Fallo en el worker: {str(e)}"}

# --- 5. INICIAR WORKER ---
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
