# -------- Base image --------
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Paquets système minimaux (certificats, curl pour le healthcheck optionnel)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl && \
    rm -rf /var/lib/apt/lists/*

# Répertoire de travail
WORKDIR /app

# -------- Dépendances Python --------
# On copie d’abord uniquement le requirements.txt pour profiter du cache
COPY requirements.txt /app/requirements.txt

# Important : --no-cache-dir pour limiter la taille de l'image
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

# -------- Code & assets --------
# Dossiers nécessaires
RUN mkdir -p /app/model /app/templates /app/static

# Code de l’app
COPY app.py                /app/app.py
COPY templates/            /app/templates/

# …
# … tes étapes avant
COPY templates/ /app/templates/
COPY static/config.json /app/static/config.json   # ← SANS slash en tête
# … tes étapes après

# Modèle ML
COPY RandomForest.pkl      /app/model/RandomForest.pkl

# Configuration front lue par le formulaire (statique)
# => Accessible côté navigateur à l’URL: /static/config.json
#COPY config.json           /app/static/config.json

# (Optionnel) si tu as d’autres fichiers Python:
# COPY *.py /app/

# -------- Exposition & lancement --------
# ... ton build
EXPOSE 5000

HEALTHCHECK --interval=5s --timeout=2s --retries=10 \
  CMD curl -fsS http://localhost:5000/health || exit 1

CMD ["gunicorn","-b","0.0.0.0:5000","app:app","--workers","1","--threads","4","--timeout","60"]



