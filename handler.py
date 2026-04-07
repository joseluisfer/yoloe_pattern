import runpod
import numpy as np
import base64
import cv2
from ultralytics import YOLO
from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

model = YOLO("yoloe-26x-seg.pt")

def decode_base64(b64_str):
    nparr = np.frombuffer(base64.b64decode(b64_str), np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

def handler(job):
    input_data = job["input"]
    
    img = decode_base64(input_data["file"])
    ref_img = decode_base64(input_data["pattern"])
    
    h, w = ref_img.shape[:2]
    visual_prompts = dict(
        bboxes=np.array([[0, 0, w, h]]), 
        cls=np.array([0])
    )

    results = model.predict(
        source=img,
        refer_image=ref_img,
        visual_prompts=visual_prompts,
        predictor=YOLOEVPSegPredictor,
        conf=input_data.get("conf", 0.25),
        iou=input_data.get("iou", 0.45),
        imgsz=input_data.get("imgsz", 1280)
    )

    detections = []
    for res in results[0].summary():
        detections.append({
            "bbox": res["box"],
            "confidence": res["confidence"]
        })

    return {"status": "success", "detections": detections}

runpod.serverless.start({"handler": handler})
