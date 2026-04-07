FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive

# Instalamos dependencias de sistema para OpenCV y procesamiento de imagen
RUN apt-get update && apt-get install -y \
    wget \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Instalamos requerimientos de Python
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Descarga oficial del modelo YOLOE-26x-seg
RUN wget -q https://github.com/ultralytics/assets/releases/download/v8.3.0/yoloe-26x-seg.pt -O yoloe-26x-seg.pt

# Copiamos el código del worker
COPY handler.py .

# Ejecutamos el handler en modo unbuffered para ver los logs en tiempo real
CMD ["python", "-u", "handler.py"]
