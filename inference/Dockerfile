FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends build-essential wget && rm -rf /var/lib/apt/lists/*

# torch cpu wheels
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2

RUN pip install --no-cache-dir flask==3.0.3 FlagEmbedding==1.2.10 numpy==1.26.4 uvicorn==0.30.6

WORKDIR /app
COPY app.py /app/app.py

EXPOSE 5001
CMD ["python", "app.py"]


