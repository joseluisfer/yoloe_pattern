FROM python:3.10-slim-bookworm

# Evitar que Python genere archivos .pyc y habilitar logs en tiempo real
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    wget \
    git \
    libgl1 \
    libglib2.0-0 \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Actualizar pip es CRITICO para que respete numpy<2.0.0
RUN pip install --no-cache-dir --upgrade pip

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Descarga del modelo
RUN wget -q https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-26x-seg.pt -O yoloe-26x-seg.pt

COPY handler.py .

CMD ["python", "-u", "handler.py"]
