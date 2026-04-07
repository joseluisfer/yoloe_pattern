FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    wget git libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Descarga directa del modelo YOLOE-26x-seg (v8.4.0 assets)
RUN wget -q https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26x-seg.pt -O yolo26x-seg.pt

COPY handler.py .

CMD ["python", "-u", "handler.py"]
