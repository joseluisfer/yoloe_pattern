FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y \
    wget \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY handler.py .

RUN mkdir -p /models
WORKDIR /models
RUN wget -O yoloe.pt https://huggingface.co/ultralytics/yoloe/resolve/main/yoloe-26x-seg.pt

WORKDIR /app

CMD ["python", "-u", "handler.py"]
