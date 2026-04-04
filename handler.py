import runpod
import torch
from ultralytics import YOLOE
import numpy as np
from PIL import Image
import io
import base64

print("Cargando modelo...")
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Usando dispositivo: {device}")

# Cargar modelo
model = YOLOE("yoloe-26x-seg.pt")
model.to(device)
print("Modelo listo")

def decode_image(b64_string):
    if "," in b64_string:
        b64_string = b64_string.split(",")[1]
    img = Image.open(io.BytesIO(base64.b64decode(b64_string))).convert("RGB")
    return np.array(img)

def handler(job):
    try:
        inp = job.get("input", {})
        target_b64 = inp.get("file")
        pattern_b64 = inp.get("pattern")
        
        # Parámetros configurables
        conf_threshold = float(inp.get("conf", 0.25))
        iou_threshold = float(inp.get("iou", 0.45))
        imgsz = int(inp.get("imgsz", 640))
        
        if not target_b64 or not pattern_b64:
            return {"error": "Faltan 'file' o 'pattern'"}
        
        # Decodificar imagen
        target = decode_image(target_b64)
        
        # Configurar clases (text prompt)
        model.set_classes(["object"])
        
        # Inferencia con parámetros configurables
        results = model.predict(
            target, 
            verbose=False, 
            imgsz=imgsz,
            conf=conf_threshold,
            iou=iou_threshold
        )
        
        detections = []
        if results and results[0].boxes:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            
            for i, conf in enumerate(confs):
                detections.append({
                    "bbox": [float(x) for x in boxes[i]],
                    "confidence": float(conf)
                })
        
        return {
            "detections": detections,
            "count": len(detections),
            "params_used": {
                "conf": conf_threshold,
                "iou": iou_threshold,
                "imgsz": imgsz
            },
            "device": device
        }
        
    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
