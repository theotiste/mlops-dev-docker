# Dockerfile
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# 1) Déps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2) Code + modèle + front statique
COPY app.py ./app.py
COPY RandomForest.pkl ./RandomForest.pkl
COPY static ./static

# 3) Gunicorn (1 worker + threads suffisent pour une démo)
ENV GUNICORN_CMD_ARGS="--workers=1 --threads=2 --timeout=60"
EXPOSE 5000

# 4) Démarrage
CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:app"]
