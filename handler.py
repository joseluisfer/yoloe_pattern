import runpod
import torch
from ultralytics import YOLOE
import numpy as np
from PIL import Image
import io
import base64

# Cargar modelo
print("Cargando modelo...")
device = "0" if torch.cuda.is_available() else "cpu"
model = YOLOE("yoloe-26x-seg.pt").to(device)
print(f"Modelo listo en {device}")

def decode_image(b64_string):
    """Decodifica base64 a imagen"""
    if "," in b64_string:
        b64_string = b64_string.split(",")[1]
    return np.array(Image.open(io.BytesIO(base64.b64decode(b64_string))).convert("RGB"))

def handler(job):
    try:
        # Obtener inputs
        inp = job.get("input", {})
        target_b64 = inp.get("file")
        pattern_b64 = inp.get("pattern")
        threshold = float(inp.get("threshold", 0.25))
        
        if not target_b64 or not pattern_b64:
            return {"error": "Faltan 'file' o 'pattern'"}
        
        # Decodificar imágenes
        target = decode_image(target_b64)
        pattern = decode_image(pattern_b64)
        
        # Configurar YOLOE con texto genérico (por ahora)
        model.set_classes(["object"])
        
        # Detectar objetos
        results = model.predict(target, verbose=False)
        
        # Filtrar por threshold (simulación visual)
        detections = []
        if results and results[0].boxes:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            
            for i, conf in enumerate(confs):
                if conf >= threshold:
                    detections.append({
                        "bbox": [float(x) for x in boxes[i].tolist()],
                        "confidence": float(conf)
                    })
        
        return {"detections": detections, "count": len(detections)}
        
    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
