import runpod
import torch
from ultralytics import YOLO # Clase base que detecta la arquitectura YOLOE
from PIL import Image
import io
import base64
import os

# Optimización para RTX 5090
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True

model = None

def load_model():
    global model
    # Nota: Ultralytics renombró yoloe-26x a yolo26x en sus últimos releases
    model_path = "yolo26x-seg.pt" 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # La clase YOLO carga automáticamente las cabeceras de Open-Vocabulary
    model = YOLO(model_path).to(device)
    print(f"✅ YOLOE-26x (Open-Vocab) cargado en: {device}")

try:
    load_model()
except Exception as e:
    print(f"❌ Error carga: {e}")

def decode_b64(b64_str):
    if b64_str and "," in b64_str:
        b64_str = b64_str.split(",")[1]
    return Image.open(io.BytesIO(base64.b64decode(b64_str))).convert("RGB")

def handler(job):
    if model is None: return {"error": "Modelo no inicializado"}

    try:
        job_input = job.get("input", {})
        scene_img = decode_b64(job_input.get("file"))
        pattern_img = decode_b64(job_input.get("pattern"))
        
        # --- VISUAL PROMPTING SEGÚN DOCUMENTACIÓN 2026 ---
        # Definimos que nuestra 'clase' a buscar es la imagen patrón.
        # Esto inyecta el embedding visual en el cabezal del modelo.
        model.set_classes([pattern_img]) 

        # Ahora el predict solo buscará lo que definimos arriba
        results = model.predict(
            source=scene_img,
            conf=job_input.get("threshold", 0.25),
            imgsz=640,
            verbose=False
        )

        detections = []
        if results:
            res = results[0]
            if res.boxes:
                boxes = res.boxes.xyxy.cpu().numpy()
                confs = res.boxes.conf.cpu().numpy()
                for i in range(len(boxes)):
                    detections.append({
                        "bbox": [round(float(x), 1) for x in boxes[i]],
                        "confidence": round(float(confs[i]), 3)
                    })

        return {"status": "success", "detections": detections}

    except Exception as e:
        return {"error": f"Fallo en Visual Prompting: {str(e)}"}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
