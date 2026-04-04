FROM runpod/base:0.6.3-cuda11.8.0

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Instalar Python packages con numpy bloqueado en 1.26.4
RUN pip3 install --no-cache-dir \
    torch==2.0.1 \
    torchvision==0.15.2 \
    --index-url https://download.pytorch.org/whl/cu118

RUN pip3 install --no-cache-dir \
    numpy==1.26.4 \
    ultralytics>=8.3.40,<9.0.0 \
    runpod \
    opencv-python-headless \
    pillow

WORKDIR /app

# Descargar modelo
RUN wget -q https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-26x-seg.pt -O yoloe-26x-seg.pt

COPY handler.py .

CMD ["python3", "-u", "handler.py"]
