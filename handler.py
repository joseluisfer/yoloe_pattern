import runpod
import torch
import numpy as np
from PIL import Image
import io
import base64
from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

print("=" * 60)
print("Iniciando YOLOE Visual Prompt Worker")
print("=" * 60)

# Configurar dispositivo
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"✅ Dispositivo: {device}")
if device == "cuda:0":
    print(f"   GPU: {torch.cuda.get_device_name(0)}")

# Cargar modelo
print("📦 Cargando modelo YOLOE-26x-seg...")
model = YOLOE("yoloe-26x-seg.pt")
model.to(device)
print("✅ Modelo cargado exitosamente")

def decode_image(b64_string):
    """Decodifica base64 a imagen PIL RGB"""
    if "," in b64_string:
        b64_string = b64_string.split(",")[1]
    image_bytes = base64.b64decode(b64_string)
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return img

def handler(job):
    """
    Handler para visual prompting con YOLOE
    
    Input esperado:
    {
        "input": {
            "file": "base64_imagen_donde_buscar",
            "pattern": "base64_imagen_de_referencia",
            "bbox": [x1, y1, x2, y2],  # opcional: bbox del objeto en pattern
            "conf": 0.25,
            "iou": 0.45,
            "imgsz": 640
        }
    }
    """
    try:
        inp = job.get("input", {})
        
        # Obtener imágenes con tus nombres de campo
        target_b64 = inp.get("file")      # imagen donde buscar
        pattern_b64 = inp.get("pattern")  # imagen de referencia (lo que buscamos)
        
        # Parámetros opcionales
        conf_threshold = float(inp.get("conf", 0.25))
        iou_threshold = float(inp.get("iou", 0.45))
        imgsz = int(inp.get("imgsz", 640))
        
        # Bounding box opcional en la imagen pattern
        custom_bbox = inp.get("bbox", None)
        
        if not target_b64:
            return {"error": "Falta 'file' (imagen donde buscar)"}
        
        if not pattern_b64:
            return {"error": "Falta 'pattern' (imagen de referencia)"}
        
        # Decodificar imágenes
        target_img = decode_image(target_b64)
        pattern_img = decode_image(pattern_b64)
        
        print(f"📐 Target size (file): {target_img.size}")
        print(f"📐 Pattern size (pattern): {pattern_img.size}")
        
        # Si no se proporciona bbox, usar toda la imagen pattern como referencia
        if custom_bbox:
            bboxes = np.array([custom_bbox], dtype=np.float32)
            print(f"📦 Usando bbox personalizado: {custom_bbox}")
        else:
            # Usar toda la imagen pattern como referencia
            w, h = pattern_img.size
            bboxes = np.array([[0, 0, w, h]], dtype=np.float32)
            print(f"📦 Usando pattern completo: {w}x{h}")
        
        # Configurar visual prompts según documentación oficial
        # IMPORTANTE: cls debe ser secuencial empezando desde 0
        visual_prompts = dict(
            bboxes=bboxes,
            cls=np.array([0])  # ID 0 para el objeto a buscar
        )
        
        print(f"🎯 Visual prompt configurado con {len(bboxes)} bounding box(es)")
        print(f"🔍 Parámetros: conf={conf_threshold}, iou={iou_threshold}, imgsz={imgsz}")
        
        # Ejecutar inferencia con visual prompt
        results = model.predict(
            target_img,
            refer_image=pattern_img,  # Imagen de referencia (pattern)
            visual_prompts=visual_prompts,
            predictor=YOLOEVPSegPredictor,
            conf=conf_threshold,
            iou=iou_threshold,
            imgsz=imgsz,
            verbose=False
        )
        
        # Procesar resultados
        detections = []
        if results and len(results) > 0:
            res = results[0]
            if res.boxes:
                boxes = res.boxes.xyxy.cpu().numpy()
                confs = res.boxes.conf.cpu().numpy()
                
                for i, (box, conf) in enumerate(zip(boxes, confs)):
                    detections.append({
                        "bbox": [float(x) for x in box],
                        "confidence": float(conf)
                    })
                    
                    # Incluir máscara si está disponible
                    if hasattr(res, 'masks') and res.masks is not None:
                        mask = res.masks.data[i].cpu().numpy()
                        mask_b64 = base64.b64encode(mask.tobytes()).decode('utf-8')
                        detections[-1]["mask"] = mask_b64
        
        print(f"✅ Encontradas {len(detections)} detecciones")
        
        return {
            "success": True,
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
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

# Iniciar worker
runpod.serverless.start({"handler": handler})
