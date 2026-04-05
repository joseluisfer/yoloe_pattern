FROM runpod/base:0.6.3-cuda11.8.0

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Instalar Python packages con verbose
RUN pip3 install --verbose --no-cache-dir \
    torch==2.0.1 \
    torchvision==0.15.2 \
    --index-url https://download.pytorch.org/whl/cu118

RUN pip3 install --verbose --no-cache-dir \
    numpy==1.23.5

RUN pip3 install --verbose --no-cache-dir \
    ultralytics==8.3.40

RUN pip3 install --verbose --no-cache-dir \
    runpod \
    pillow \
    opencv-python-headless

# VERIFICAR instalación
RUN python3 -c "import runpod; print('✅ runpod instalado correctamente')"
RUN python3 -c "import torch; print(f'✅ torch version: {torch.__version__}')"
RUN python3 -c "import ultralytics; print(f'✅ ultralytics version: {ultralytics.__version__}')"

# Descargar modelo
RUN wget -q https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-26x-seg.pt -O yoloe-26x-seg.pt

COPY handler.py .

CMD ["python3", "-u", "handler.py"]
