import runpod
import torch
from ultralytics import YOLOE
import numpy as np
from PIL import Image
import io
import base64
import cv2

print("=" * 50)
print("Iniciando YOLOE Visual Prompt Worker")
print("=" * 50)

# Verificar GPU
if torch.cuda.is_available():
    device = "cuda"
    gpu_name = torch.cuda.get_device_name(0)
    cuda_version = torch.version.cuda
    print(f"✅ GPU disponible: {gpu_name}")
    print(f"✅ CUDA version: {cuda_version}")
    print(f"✅ PyTorch version: {torch.__version__}")
else:
    device = "cpu"
    print("⚠️  GPU no disponible, usando CPU")
    print("⚠️  El rendimiento será mucho más lento")

print(f"✅ Usando dispositivo: {device}")
print("=" * 50)

# Cargar modelo YOLOE
print("Cargando modelo YOLOE-26x-seg...")
try:
    model = YOLOE("yoloe-26x-seg.pt")
    model.to(device)
    print("✅ Modelo cargado exitosamente")
except Exception as e:
    print(f"❌ Error cargando modelo: {e}")
    raise

def decode_image(b64_string):
    """Decodifica base64 a imagen numpy"""
    try:
        if "," in b64_string:
            b64_string = b64_string.split(",")[1]
        image_bytes = base64.b64decode(b64_string)
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return np.array(img)
    except Exception as e:
        print(f"Error decodificando imagen: {e}")
        raise

def find_similar_visual_prompts(target_img, pattern_img, conf_threshold=0.25, iou_threshold=0.45, imgsz=640):
    """
    Encuentra regiones similares a la imagen pattern usando YOLOE con visual prompt
    """
    try:
        # Redimensionar pattern para usarlo como referencia visual
        h, w = pattern_img.shape[:2]
        
        # YOLOE necesita bounding box de ejemplo para visual prompt
        # Usamos toda la imagen pattern como referencia
        visual_prompts = {
            "bboxes": np.array([[0, 0, w, h]]),
            "cls": np.array([0])
        }
        
        # Inferencia con visual prompt
        results = model.predict(
            target_img,
            visual_prompts=visual_prompts,
            conf=conf_threshold,
            iou=iou_threshold,
            imgsz=imgsz,
            verbose=False,
            device=device
        )
        
        detections = []
        if results and len(results) > 0 and results[0].boxes:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            cls_ids = results[0].boxes.cls.cpu().numpy().astype(int)
            
            for i in range(len(boxes)):
                detections.append({
                    "bbox": [float(x) for x in boxes[i]],
                    "confidence": float(confs[i]),
                    "class_id": int(cls_ids[i])
                })
        
        return detections
        
    except Exception as e:
        print(f"Error en visual prompt: {e}")
        # Fallback a texto simple
        print("Usando fallback a text prompt 'object'")
        model.set_classes(["object"])
        results = model.predict(
            target_img,
            conf=conf_threshold,
            iou=iou_threshold,
            imgsz=imgsz,
            verbose=False,
            device=device
        )
        
        detections = []
        if results and len(results) > 0 and results[0].boxes:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            
            for i in range(len(boxes)):
                detections.append({
                    "bbox": [float(x) for x in boxes[i]],
                    "confidence": float(confs[i]),
                    "class_id": -1
                })
        
        return detections

def handler(job):
    """Handler principal de RunPod"""
    try:
        start_time = torch.cuda.Event(enable_timing=True) if device == "cuda" else None
        end_time = torch.cuda.Event(enable_timing=True) if device == "cuda" else None
        
        if start_time:
            start_time.record()
        
        # Obtener inputs
        inp = job.get("input", {})
        target_b64 = inp.get("file")
        pattern_b64 = inp.get("pattern")
        
        # Parámetros configurables
        conf_threshold = float(inp.get("conf", 0.25))
        iou_threshold = float(inp.get("iou", 0.45))
        imgsz = int(inp.get("imgsz", 640))
        use_visual_prompt = inp.get("use_visual_prompt", True)
        
        if not target_b64:
            return {"error": "Falta 'file' (imagen objetivo)"}
        
        if use_visual_prompt and not pattern_b64:
            return {"error": "Falta 'pattern' para visual prompt"}
        
        # Decodificar imágenes
        target_img = decode_image(target_b64)
        pattern_img = decode_image(pattern_b64) if pattern_b64 else None
        
        print(f"Target size: {target_img.shape}")
        if pattern_img is not None:
            print(f"Pattern size: {pattern_img.shape}")
        print(f"Params: conf={conf_threshold}, iou={iou_threshold}, imgsz={imgsz}")
        print(f"Visual prompt: {use_visual_prompt}")
        
        # Detectar
        if use_visual_prompt and pattern_img is not None:
            detections = find_similar_visual_prompts(
                target_img, pattern_img, conf_threshold, iou_threshold, imgsz
            )
        else:
            # Fallback a texto
            model.set_classes(["object"])
            results = model.predict(
                target_img, 
                conf=conf_threshold, 
                iou=iou_threshold, 
                imgsz=imgsz,
                verbose=False,
                device=device
            )
            detections = []
            if results and len(results) > 0 and results[0].boxes:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                confs = results[0].boxes.conf.cpu().numpy()
                for i in range(len(boxes)):
                    detections.append({
                        "bbox": [float(x) for x in boxes[i]],
                        "confidence": float(confs[i])
                    })
        
        # Calcular tiempo de inferencia
        inference_time = None
        if start_time and end_time:
            end_time.record()
            torch.cuda.synchronize()
            inference_time = start_time.elapsed_time(end_time) / 1000.0
        
        return {
            "detections": detections,
            "count": len(detections),
            "params_used": {
                "conf": conf_threshold,
                "iou": iou_threshold,
                "imgsz": imgsz,
                "use_visual_prompt": use_visual_prompt
            },
            "device": device,
            "inference_time_seconds": inference_time,
            "target_size": [int(target_img.shape[1]), int(target_img.shape[0])]
        }
        
    except Exception as e:
        print(f"Error en handler: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

# Iniciar worker
print("Iniciando RunPod serverless worker...")
runpod.serverless.start({"handler": handler})
