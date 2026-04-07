import runpod
import numpy as np
from ultralytics import YOLO
from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

# Carga del modelo global para optimizar "warm starts"
model = YOLO("yoloe-26x-seg.pt")

def handler(job):
    input_data = job["input"]
    target_img = input_data.get("image")        # Imagen donde buscar
    ref_img = input_data.get("ref_image")      # Imagen de referencia
    bboxes = input_data.get("bboxes")          # Ejemplo: [[x1, y1, x2, y2]]
    
    # Configuración de prompts visuales
    visual_prompts = dict(
        bboxes=np.array(bboxes),
        cls=np.array([i for i in range(len(bboxes))])
    )

    results = model.predict(
        source=target_img,
        refer_image=ref_img,
        visual_prompts=visual_prompts,
        predictor=YOLOEVPSegPredictor
    )

    return {"status": "success", "summary": results[0].tojson()}

runpod.serverless.start({"handler": handler})
