FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    fonts-liberation \
    fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-cloud.txt .
RUN pip install --no-cache-dir -r requirements-cloud.txt

COPY . .

RUN mkdir -p uploads outputs fonts

EXPOSE 5050

CMD ["sh", "-c", "gunicorn app:app --bind 0.0.0.0:${PORT:-5050} --timeout 300 --workers 1"]
