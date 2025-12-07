FROM ubuntu:22.04

# Установка системных зависимостей
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3-dev \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Создаем символьную ссылку python3 → python (для совместимости)
RUN ln -sf /usr/bin/python3.10 /usr/bin/python

WORKDIR /app

# Обновление pip и установка основных пакетов
RUN python -m pip install --upgrade pip setuptools wheel

# Установка совместимых версий aiobotocore и botocore
RUN python -m pip install --no-cache-dir \
    botocore==1.31.17 \
    aiobotocore==2.5.4

# Установка torch с явным указанием python
RUN python -m pip install --no-cache-dir \
    torch==2.0.1 \
    torchvision==0.15.2

# Копируем и устанавливаем остальные зависимости
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH=/app

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]