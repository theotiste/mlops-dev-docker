#!/usr/bin/env bash
set -euo pipefail

APP_SERVICE=api          # nom du service dans compose.yaml
PORT_HOST=5001           # port exposé côté hôte (compose.yaml: "5001:5000")

banner(){ printf "\n\033[1;36m==> %s\033[0m\n" "$*"; }

banner "Vérification Docker Compose"
docker compose version >/dev/null

banner "Arrêt & nettoyage (orphelins)"
docker compose down --remove-orphans || true

banner "Rebuild sans cache"
docker compose build --no-cache

banner "Démarrage en détaché"
docker compose up -d

banner "Services en cours"
docker compose ps

banner "Attente du boot de l'API (10s)…"
sleep 10

banner "Tests rapides"
set +e
curl -sS -i "http://localhost:${PORT_HOST}/health" || true
echo
curl -sS -i -X POST "http://localhost:${PORT_HOST}/predict" \
  -H "Content-Type: application/json" \
  -d '{"features":[13.54,14.36,87.46,566.3,0.09779,0.08129,0.06664,0.04781,0.1885,0.05766,0.2699,0.7886,2.058,23.56,0.008462,0.0146,0.02387,0.01315,0.0198,0.0023,15.11,19.26,99.7,711.2,0.144,0.1773,0.239,0.1288,0.2977,0.07259]}' \
  || true
set -e

banner "Logs (Ctrl+C pour quitter)"
docker compose logs -f ${APP_SERVICE}

