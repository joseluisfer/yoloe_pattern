FROM ultralytics/ultralytics:latest

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY handler.py .
# Descarga el modelo durante la construcción
RUN python -c "from ultralytics import YOLO; YOLO('yoloe-26x-seg.pt')"

CMD ["python", "-u", "handler.py"]

