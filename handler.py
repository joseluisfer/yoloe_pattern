import runpod
import torch
from ultralytics import ASSET_PATH, YOLOE
from PIL import Image
import io
import base64
import os

# Optimizaciones para la arquitectura Blackwell (RTX 5090)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True

model = None

def load_model():
    global model
    model_path = "yoloe-26x-seg.pt"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No se encontró el peso {model_path}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Cargamos específicamente con la clase YOLOE
    model = YOLOE(model_path).to(device)
    print(f"✅ YOLOE-26x-seg cargado en: {device}")

try:
    load_model()
except Exception as e:
    print(f"❌ Error crítico de inicio: {e}")

def decode_b64(b64_str):
    if b64_str and "," in b64_str:
        b64_str = b64_str.split(",")[1]
    img_bytes = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")

def handler(job):
    if model is None:
        return {"error": "Modelo no cargado correctamente"}

    try:
        job_input = job.get("input", {})
        scene_img = decode_b64(job_input.get("file"))      # Imagen donde buscar
        pattern_img = decode_b64(job_input.get("pattern")) # Imagen de referencia
        
        conf_threshold = job_input.get("threshold", 0.25)

        # INFERENCIA SEGÚN DOCS OFICIALES
        # El parámetro 'visual_prompt' inyecta la imagen patrón al modelo
        results = model.predict(
            source=scene_img,
            visual_prompt=pattern_img, 
            conf=conf_threshold,
            imgsz=640,
            verbose=False
        )

        detections = []
        if results:
            res = results[0]
            # Procesamos las cajas detectadas
            if res.boxes:
                boxes = res.boxes.xyxy.cpu().numpy()
                confs = res.boxes.conf.cpu().numpy()
                
                for i in range(len(boxes)):
                    detections.append({
                        "bbox": [round(float(x), 1) for x in boxes[i]],
                        "confidence": round(float(confs[i]), 3)
                    })

        return {
            "status": "success",
            "total_found": len(detections),
            "detections": detections
        }

    except Exception as e:
        return {"error": f"Fallo en el proceso: {str(e)}"}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
