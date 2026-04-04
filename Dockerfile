FROM pytorch/pytorch:2.0.1-cuda11.8-cudnn8-runtime

RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 wget && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    ultralytics==8.3.40 \
    runpod \
    opencv-python-headless \
    pillow

WORKDIR /app
RUN wget -q https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-26x-seg.pt -O yoloe-26x-seg.pt
COPY handler.py .

CMD ["python", "-u", "handler.py"]
