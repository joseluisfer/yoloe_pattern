import runpod
import torch
from ultralytics import YOLOE
from PIL import Image
import io
import base64
import os

# Optimización para RTX 5090 (Blackwell)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True

model = None

def load_model():
    global model
    model_path = "yoloe-26x-seg.pt"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No existe el modelo en {model_path}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Cargamos el modelo. Aunque sea -seg, solo usaremos las boxes.
    model = YOLOE(model_path).to(device)
    print(f"✅ YOLOE cargado en: {device}")

try:
    load_model()
except Exception as e:
    print(f"❌ Error carga: {e}")

def decode_b64(b64_str):
    if "," in b64_str:
        b64_str = b64_str.split(",")[1]
    return Image.open(io.BytesIO(base64.b64decode(b64_str))).convert("RGB")

def handler(job):
    if model is None: return {"error": "Modelo no cargado"}

    try:
        job_input = job.get("input", {})
        scene_img = decode_b64(job_input.get("file"))
        pattern_img = decode_b64(job_input.get("pattern"))
        
        # Inferencia pura de boxes
        results = model.predict(
            source=scene_img,
            visual_prompt=pattern_img,
            conf=job_input.get("threshold", 0.25),
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
        return {"error": str(e)}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
