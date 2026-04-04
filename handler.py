import runpod
import torch
from ultralytics import YOLOE
import numpy as np
from PIL import Image
import io
import base64
import cv2
import os

# --- 1. CARGA GLOBAL DEL MODELO ---
# Usamos una función de carga para manejar mejor los reintentos
model = None

def load_model():
    global model
    model_path = "yoloe-26x-seg.pt"
    print(f"🚀 Intentando cargar {model_path}...")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No se encuentra el archivo {model_path} en /app")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Cargamos el modelo y lo movemos al dispositivo
    model = YOLOE(model_path).to(device)
    print(f"✅ Modelo cargado exitosamente en: {device}")

try:
    load_model()
except Exception as e:
    print(f"❌ ERROR CRÍTICO AL INICIAR: {e}")
    # No levantamos la excepción aquí para que RunPod no entre en loop infinito 
    # y podamos ver los logs si algo falla.

def decode_base64_to_image(b64_str):
    """Limpia y decodifica el string base64 a una imagen PIL"""
    try:
        if "," in b64_str:
            b64_str = b64_str.split(",")[1]
        img_bytes = base64.b64decode(b64_str)
        return Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        raise ValueError(f"Error al decodificar Base64: {str(e)}")

def handler(job):
    if model is None:
        return {"error": "El modelo no se cargó correctamente en el arranque."}

    try:
        job_input = job.get("input", {})
        scene_b64 = job_input.get("file")
        pattern_b64 = job_input.get("pattern")
        conf_threshold = job_input.get("threshold", 0.25)

        if not scene_b64 or not pattern_b64:
            return {"error": "Faltan parámetros 'file' o 'pattern'"}

        # 2. Procesamiento de imágenes
        img_scene = decode_base64_to_image(scene_b64)
        img_pattern = decode_base64_to_image(pattern_b64)

        # 3. Inferencia
        results = model.predict(
            source=img_scene, 
            visual_prompt=img_pattern, 
            conf=conf_threshold,
            verbose=False
        )

        detections = []
        if results and len(results) > 0:
            res = results[0]
            if res.boxes:
                boxes = res.boxes.xyxy.cpu().numpy()
                confs = res.boxes.conf.cpu().numpy()
                masks = res.masks.xy if res.masks is not None else [None] * len(boxes)

                for i in range(len(boxes)):
                    contour_data = []
                    if masks[i] is not None:
                        # Simplificación para Android
                        points = masks[i].astype(np.float32)
                        epsilon = 0.001 * cv2.arcLength(points, True)
                        approx = cv2.approxPolyDP(points, epsilon, True)
                        contour_data = [{"x": round(float(p[0][0]), 1), "y": round(float(p[0][1]), 1)} for p in approx]

                    detections.append({
                        "bbox": [round(float(x), 1) for x in boxes[i].tolist()],
                        "confidence": round(float(confs[i]), 3),
                        "contour": contour_data
                    })

        return {
            "status": "success",
            "detections": detections
        }

    except Exception as e:
        return {"error": f"Error en procesamiento: {str(e)}"}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
