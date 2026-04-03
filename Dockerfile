# 1. Imagen base con CUDA 12.1 (Imprescindible para YOLOE-26x y CLIP en GPU)
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# 2. Variables de entorno
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# 3. Dependencias de sistema (Añadimos git para que pip pueda bajar CLIP)
RUN apt-get update && apt-get install -y \
    wget \
    git \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 4. Copiar e instalar requerimientos
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Descargar el modelo v26x-seg (Asegúrate de que la URL sea válida)
RUN wget -q https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-26x-seg.pt -O yoloe-26x-seg.pt

# 6. Copiar el código del handler
COPY handler.py .

# 7. Ejecutar el worker de RunPod
CMD ["python", "-u", "handler.py"]
