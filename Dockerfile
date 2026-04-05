FROM runpod/base:0.6.3-cuda11.8.0

RUN apt-get update && apt-get install -y \
    libgl1 libglib2.0-0 wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copiar requirements primero
COPY requirements.txt .

# Instalar PyTorch con CUDA 11.8
RUN pip3 install --no-cache-dir torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

# Instalar resto de dependencias
RUN pip3 install --no-cache-dir -r requirements.txt

# Descargar modelo
RUN wget -q https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-26x-seg.pt -O yoloe-26x-seg.pt

COPY handler.py .

CMD ["python3", "-u", "handler.py"]
