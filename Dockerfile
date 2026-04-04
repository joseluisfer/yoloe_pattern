# Usamos la imagen oficial que ya tiene Python, PyTorch y OpenCV configurados
FROM ultralytics/ultralytics:latest

# Instalamos runpod para el servidor serverless
RUN pip install --no-cache-dir runpod

# Descargamos el modelo específico para que ya esté dentro de la imagen (más rápido al arrancar)
RUN curl -L https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-26x-seg.pt -o /yoloe-26x-seg.pt

# Copiamos el handler
COPY handler.py /handler.py

# Ejecutamos el handler
CMD [ "python", "-u", "/handler.py" ]
