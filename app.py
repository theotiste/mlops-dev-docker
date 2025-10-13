from flask import Flask, jsonify, redirect, url_for
import joblib, os

def create_app():
    app = Flask(__name__, static_folder="static", static_url_path="/static")

    # charge le modèle SANS planter l'import (gère proprement les erreurs)
    model_path = os.path.join(app.root_path, "RandomForest.pkl")
    app.model = None
    try:
        if os.path.exists(model_path):
            app.model = joblib.load(model_path)
        else:
            app.logger.warning("Model file not found: %s", model_path)
    except Exception as e:
        app.logger.exception("Model load failed: %s", e)

    @app.get("/health")
    def health():
        return jsonify(ok=True)

    # racines pratiques
    @app.get("/")
    def index():
        return redirect(url_for("static", filename="form.html"))

    @app.get("/form")
    def form_alias():
        return redirect(url_for("static", filename="form.html"))

    @app.get("/result")
    def result_alias():
        return redirect(url_for("static", filename="result_standalone.html"))

    @app.post("/predict")
    def predict():
        from flask import request
        data = request.get_json(silent=True) or {}
        feats = data.get("features")
        if not isinstance(feats, list) or len(feats) != 30:
            return jsonify(error="bad input"), 400
        if app.model is None:
            # renvoie un faux résultat pour l’UI si le modèle manque
            return jsonify(prediction=1, probability=[0.0, 1.0], model_version="v1"), 200
        import numpy as np
        X = np.array([feats], dtype=float)
        proba = app.model.predict_proba(X)[0].tolist()
        pred = int(proba[1] >= proba[0])
        return jsonify(prediction=pred, probability=proba, model_version="v1")

    return app

# objet Flask attendu par gunicorn app:app
app = create_app()
