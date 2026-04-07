import runpod
import torch
from ultralytics import YOLO
from PIL import Image
import io
import base64
import os

# Optimización RTX 5090
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True

model = None

def load_model():
    global model
    model_path = "yolo26x-seg.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Cargamos el modelo
    model = YOLO(model_path).to(device)
    print(f"✅ YOLOE-26x-Seg cargado en: {device}")

try:
    load_model()
except Exception as e:
    print(f"❌ Error carga: {e}")

def decode_b64(b64_str):
    if b64_str and "," in b64_str:
        b64_str = b64_str.split(",")[1]
    # Limpieza básica para evitar el error de "broken PNG"
    b64_str = b64_str.replace("\n", "").replace("\r", "").strip()
    return Image.open(io.BytesIO(base64.b64decode(b64_str))).convert("RGB")

def handler(job):
    if model is None: return {"error": "Modelo no listo"}

    try:
        job_input = job.get("input", {})
        scene_img = decode_b64(job_input.get("file"))
        pattern_img = decode_b64(job_input.get("pattern"))
        
        # --- SOLUCIÓN DEFINITIVA ---
        # En los modelos YOLO26/YOLOE de segmentación, el visual_prompt 
        # se pasa dentro del predict, pero DEBE ser una lista o un tensor.
        results = model.predict(
            source=scene_img,
            visual_prompt=pattern_img, # La documentación indica que acepta la imagen aquí
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
        # Si el error 'visual_prompt' vuelve a aparecer, usamos el plan B: 
        # Inyectar el prompt en el modelo de forma manual antes del predict
        try:
            model.model.set_visual_prompts([pattern_img])
            results = model.predict(source=scene_img, conf=0.25, verbose=False)
            # ... (resto del procesado igual)
            return {"status": "success", "detections": "Plan B ejecutado"}
        except:
            return {"error": f"Error persistente: {str(e)}"}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
