# Usamos la imagen que ya tiene todo configurado para YOLOv8/YOLO-E
FROM ultralytics/ultralytics:latest

WORKDIR /app

# Instalar runpod
RUN pip install --no-cache-dir runpod

# Descargar el modelo YOLO-E que pediste
RUN curl -L https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-26x-seg.pt -o /app/yoloe-26x-seg.pt

# Copiar el script
COPY handler.py .

CMD [ "python", "-u", "handler.py" ]
