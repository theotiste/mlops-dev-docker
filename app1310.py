# app.py
from __future__ import annotations

import json, os, re
from typing import List, Sequence
from pathlib import Path

from flask import Flask, jsonify, request, redirect, url_for
from flask_cors import CORS

import numpy as np
import pandas as pd
from joblib import dump, load
import joblib

from flask import send_from_directory

BASE_DIR = Path(__file__).resolve().parent  

# -----------------------------
# 1) App & CORS
# -----------------------------
app = Flask(__name__, static_folder="static", static_url_path="/static")

CORS(
    app,
    resources={r"/*": {"origins": [
        "http://localhost:8081", "http://127.0.0.1:8081", "http://wsl.localhost:8081", "*"
    ]}},
    supports_credentials=False,
    methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Origin", "Accept"]
)

@app.get("/")
def index():
    return redirect(url_for("static", filename="form.html"))

@app.get("/form")
def form_page():
    return redirect(url_for("static", filename="form.html"))

@app.get("/result")
def result_page():
    return redirect(url_for("static", filename="result_standalone.html"))

@app.get("/health")
def health():
    return jsonify(ok=True)

@app.get("/config.json")
def serve_config():
    return send_from_directory(
        directory=BASE_DIR,
        path="config.json",
        mimetype="application/json",
        as_attachment=False,
        etag=False,
        max_age=0
    )


# -----------------------------
# 2) Caractéristiques & exemples
# -----------------------------
FEATURES: List[str] = [
    "radius_mean","texture_mean","perimeter_mean","area_mean",
    "smoothness_mean","compactness_mean","concavity_mean","concave points_mean",
    "symmetry_mean","fractal_dimension_mean",
    "radius_se","texture_se","perimeter_se","area_se","smoothness_se",
    "compactness_se","concavity_se","concave points_se","symmetry_se","fractal_dimension_se",
    "radius_worst","texture_worst","perimeter_worst","area_worst","smoothness_worst",
    "compactness_worst","concavity_worst","concave points_worst","symmetry_worst","fractal_dimension_worst",
]

SAMPLE_1 = [17.99,10.38,122.8,1001,0.1184,0.2776,0.3001,0.1471,0.2419,0.07871,
            1.095,0.9053,8.589,153.4,0.0064,0.0490,0.0537,0.0159,0.0303,0.0062,
            25.38,17.33,184.6,2019,0.1622,0.6656,0.7119,0.2654,0.4601,0.1189]
SAMPLE_2 = [13.540,14.36,87.46,566.3,0.09779,0.08129,0.06664,0.047810,0.1885,0.05766,
            0.2699,0.7886,2.058,23.560,0.008462,0.014600,0.02387,0.013150,0.01980,0.002300,
            15.110,19.26,99.70,711.2,0.14400,0.17730,0.23900,0.12880,0.2977,0.07259]
SAMPLE_3 = [20.57,17.77,132.90,1326.0,0.08474,0.07864,0.08690,0.07017,0.1812,0.05667,
            0.5435,0.7339,3.398,74.08,0.005225,0.01308,0.01860,0.01340,0.01389,0.003532,
            24.99,23.41,158.80,1956.0,0.1238,0.1866,0.2416,0.1860,0.2750,0.08902]
SAMPLE_4 = [13.080,15.71,85.63,520.0,0.10750,0.12700,0.04568,0.031100,0.1967,0.06811,
            0.1852,0.7477,1.383,14.670,0.004097,0.018980,0.01698,0.006490,0.01678,0.0024215,
            14.500,20.49,96.09,630.5,0.13120,0.27760,0.18900,0.07283,0.3184,0.08183]

SAMPLES = {"1": SAMPLE_1, "2": SAMPLE_2, "3": SAMPLE_3, "4": SAMPLE_4}

# -----------------------------
# 3) Modèle (cache + chargement + fallback)
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "RandomForest.pkl"

_model = None  # cache process-wide

def get_model():
    """Retourne le modèle en cache, le charge depuis le disque, ou l'entraîne si absent."""
    global _model
    if _model is not None:
        return _model

    if MODEL_PATH.exists():
        _model = joblib.load(MODEL_PATH)
        return _model

    # Fallback training (dev)
    from sklearn.datasets import load_breast_cancer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)

    mapper = {
        "mean radius": "radius_mean",
        "mean texture": "texture_mean",
        "mean perimeter": "perimeter_mean",
        "mean area": "area_mean",
        "mean smoothness": "smoothness_mean",
        "mean compactness": "compactness_mean",
        "mean concavity": "concavity_mean",
        "mean concave points": "concave points_mean",
        "mean symmetry": "symmetry_mean",
        "mean fractal dimension": "fractal_dimension_mean",
        "radius error": "radius_se",
        "texture error": "texture_se",
        "perimeter error": "perimeter_se",
        "area error": "area_se",
        "smoothness error": "smoothness_se",
        "compactness error": "compactness_se",
        "concavity error": "concavity_se",
        "concave points error": "concave points_se",
        "symmetry error": "symmetry_se",
        "fractal dimension error": "fractal_dimension_se",
        "worst radius": "radius_worst",
        "worst texture": "texture_worst",
        "worst perimeter": "perimeter_worst",
        "worst area": "area_worst",
        "worst smoothness": "smoothness_worst",
        "worst compactness": "compactness_worst",
        "worst concavity": "concavity_worst",
        "worst concave points": "concave points_worst",
        "worst symmetry": "symmetry_worst",
        "worst fractal dimension": "fractal_dimension_worst",
    }

    inv = {v: k for k, v in mapper.items()}
    X_wdbc = pd.DataFrame(index=X.index, columns=FEATURES, dtype=float)
    for f in FEATURES:
        X_wdbc[f] = X[inv[f]] if f in inv else 0.0

    y = data.target
    Xtr, Xte, ytr, yte = train_test_split(X_wdbc, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(Xtr, ytr)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    dump(clf, MODEL_PATH)

    _model = clf
    return _model

# -----------------------------
# 4) Utils parsing
# -----------------------------
def _to_float(s: str) -> float:
    return float(s.replace(",", ".").strip())

def parse_features_arg(raw: str) -> List[float]:
    if raw is None:
        raise ValueError("features manquant")
    s = raw.strip().replace(";", ",")
    if s.startswith("[") and s.endswith("]"):
        arr = json.loads(s)
        return [_to_float(str(x)) for x in arr]
    parts = re.split(r"[,\s]+", s.strip("[] "))
    parts = [p for p in parts if p]
    return [_to_float(p) for p in parts]

def _predict_row(row: Sequence[float]):
    model = get_model()
    X = pd.DataFrame([row], columns=FEATURES)
    proba = model.predict_proba(X)[0].tolist()
    y = int(model.predict(X)[0])
    return {"prediction": y, "probability": proba}

# -----------------------------
# 5) Routes
# -----------------------------

@app.get("/")
def root():
    return "OK", 200

@app.get("/health")
def health():
    return jsonify(ok=True), 200

@app.get("/samples")
def list_samples():
    return jsonify({"samples": SAMPLES})

@app.get("/form")
def form():
    values = {}
    if request.args.get("sample"):
        row = SAMPLES.get(request.args.get("sample"))
        if row:
            values = dict(zip(FEATURES, row))
    elif "features" in request.args:
        try:
            row = parse_features_arg(request.args["features"])
            if len(row) == len(FEATURES):
                values = dict(zip(FEATURES, row))
        except Exception:
            pass
    try:
        return render_template("form.html", values=values)
    except Exception:
        return jsonify(info="Static form.html introuvable. Placez-le dans ./statict.",
                       values=values), 200

@app.route("/predict", methods=["POST", "GET", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        return "", 204

    if request.method == "GET":
        sample_id = request.args.get("sample")
        if sample_id:
            row = SAMPLES.get(sample_id)
            if not row:
                return jsonify(error="sample inconnu"), 400
            return jsonify(_predict_row(row))
        if "features" in request.args:
            try:
                row = parse_features_arg(request.args["features"])
                if len(row) != len(FEATURES):
                    return jsonify(error=f"features doit contenir {len(FEATURES)} valeurs"), 400
                return jsonify(_predict_row(row))
            except Exception as e:
                return jsonify(error=f"features invalide: {e}"), 400
        return jsonify(error="Utilisez POST JSON ou GET ?sample=1..4 / ?features=..."), 400

    data = request.get_json(silent=True, force=False) or {}
    vec = data.get("features")
    if not isinstance(vec, (list, tuple)):
        return jsonify(error="Corps JSON attendu: {'features': [..30..]}"), 400
    try:
        row = [_to_float(str(v)) for v in vec]
    except Exception:
        return jsonify(error="features contient des valeurs non numériques"), 400
    if len(row) != len(FEATURES):
        return jsonify(error=f"features doit contenir {len(FEATURES)} valeurs"), 400
    return jsonify(_predict_row(row))

# -----------------------------
# 6) Entrée
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
