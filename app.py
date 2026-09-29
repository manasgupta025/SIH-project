"""
app.py
------
Flask backend serving:
  GET /api/hotspots  -> classified hotspot data as JSON (for the map dashboard)
  GET /api/stats     -> summary statistics
  GET /              -> dashboard UI
"""

from flask import Flask, jsonify, send_from_directory
from pipeline import run_pipeline
import pandas as pd
import os

app = Flask(__name__, static_folder="static")

_cache = {"df": None}


def get_data():
    if _cache["df"] is None:
        _cache["df"] = run_pipeline(use_live_data=False, n_synthetic=120)
    return _cache["df"]


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/hotspots")
def api_hotspots():
    df = get_data()
    return jsonify(df.to_dict(orient="records"))


@app.route("/api/stats")
def api_stats():
    df = get_data()
    stats = {
        "total": int(len(df)),
        "by_class": df["predicted_class"].value_counts().to_dict(),
        "by_priority": df["risk_priority"].value_counts().to_dict(),
        "avg_frp": round(float(df["frp"].mean()), 1),
    }
    return jsonify(stats)


@app.route("/api/refresh")
def api_refresh():
    _cache["df"] = None
    df = get_data()
    return jsonify({"status": "refreshed", "count": len(df)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
