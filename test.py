# app.py
from flask_cors import CORS
CORS(app, resources={r"/*": {"origins": "*"}})  # ou restreins à ton domaine
