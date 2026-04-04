# Usar una imagen base ligera con Python
FROM python:3.10-slim

# Instalar dependencias del sistema para OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Definir el directorio de trabajo
WORKDIR /

# Copiar los archivos de requisitos e instalarlos
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el handler
COPY handler.py .

# Comando para ejecutar el handler al iniciar el contenedor
CMD [ "python", "-u", "/handler.py" ]
