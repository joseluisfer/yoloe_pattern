FROM python:3.10-slim-bookworm

# Instalar dependencias de sistema (git es obligatorio para CLIP)
RUN apt-get update && apt-get install -y \
    wget \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Instalar dependencias de Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Descargar el modelo específico de Visual Prompting v8.4.0
RUN wget -q https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-26x-seg.pt -O yoloe-26x-seg.pt

COPY handler.py .

# Ejecutar el worker de RunPod
CMD ["python", "-u", "handler.py"]
