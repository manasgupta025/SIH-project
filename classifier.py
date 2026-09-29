"""
classifier.py
-------------
Core ML logic: classifies each hotspot into one of:

    WILDFIRE            - genuine vegetation/forest fire
    INDUSTRIAL_FIRE      - fire/flare at an industrial facility (real event, needs response)
    PERSISTENT_THERMAL   - recurring non-fire heat source (flare stack, kiln, factory - "normal" hot)
    AGRI_BURNING         - crop residue / agricultural burning
    FALSE_ALARM          - low confidence, likely sensor artifact

Uses a RandomForest trained on engineered features combining:
    - FIRMS radiometric signal (FRP, brightness, confidence)
    - OSM land-use context (near_industrial, near_forest, near_agri, near_urban)
    - Temporal persistence (how many consecutive days a hotspot re-appears at ~same location)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import joblib

FEATURE_COLS = [
    "frp", "brightness", "persistence_days",
    "near_industrial", "near_forest", "near_agri", "near_urban",
    "conf_score",
]

CONF_MAP = {"low": 0.3, "nominal": 0.6, "high": 0.9}


def label_ground_truth(row) -> str:
    """
    Rule-based ground-truth labeling used ONLY to build a training set from
    synthetic data (mirrors how a real labeled dataset — e.g. matched against
    NIFC/CPCB industrial-facility records — would be constructed).
    """
    kind = row["_true_kind"]
    frp = row["frp"]
    persistence = row["persistence_days"]

    if row["confidence"] == "low" and frp < 20:
        return "FALSE_ALARM"
    if kind == "industrial":
        if persistence >= 5:
            return "PERSISTENT_THERMAL"
        return "INDUSTRIAL_FIRE"
    if kind == "forest":
        return "WILDFIRE"
    if kind == "agri":
        return "AGRI_BURNING"
    return "FALSE_ALARM"


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["conf_score"] = df["confidence"].map(CONF_MAP).fillna(0.5)
    for c in ["near_industrial", "near_forest", "near_agri", "near_urban"]:
        df[c] = df[c].astype(int)
    return df


class HotspotClassifier:
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=200, max_depth=8, random_state=42, class_weight="balanced"
        )
        self.label_encoder = LabelEncoder()
        self.is_fitted = False

    def train(self, df: pd.DataFrame, verbose: bool = True):
        df = build_features(df)
        df["label"] = df.apply(label_ground_truth, axis=1)

        X = df[FEATURE_COLS]
        y = self.label_encoder.fit_transform(df["label"])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
        self.model.fit(X_train, y_train)
        self.is_fitted = True

        preds = self.model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        if verbose:
            print(f"Validation accuracy: {acc:.3f}\n")
            print(classification_report(
                y_test, preds, target_names=self.label_encoder.classes_, zero_division=0
            ))
        return acc

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("Model not trained yet. Call .train() first or .load().")
        feat_df = build_features(df)
        X = feat_df[FEATURE_COLS]
        preds = self.model.predict(X)
        probs = self.model.predict_proba(X)

        out = df.copy()
        out["predicted_class"] = self.label_encoder.inverse_transform(preds)
        out["confidence_score"] = probs.max(axis=1).round(3)
        out["risk_priority"] = out["predicted_class"].map({
            "WILDFIRE": "CRITICAL",
            "INDUSTRIAL_FIRE": "HIGH",
            "AGRI_BURNING": "MEDIUM",
            "PERSISTENT_THERMAL": "LOW",
            "FALSE_ALARM": "IGNORE",
        })
        return out

    def feature_importance(self) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("Model not trained yet.")
        imp = pd.DataFrame({
            "feature": FEATURE_COLS,
            "importance": self.model.feature_importances_
        }).sort_values("importance", ascending=False)
        return imp

    def save(self, path="model.joblib"):
        joblib.dump({"model": self.model, "label_encoder": self.label_encoder}, path)

    def load(self, path="model.joblib"):
        data = joblib.load(path)
        self.model = data["model"]
        self.label_encoder = data["label_encoder"]
        self.is_fitted = True


if __name__ == "__main__":
    from firms_client import generate_synthetic_hotspots
    from osm_context import generate_synthetic_context

    print("Generating training data...")
    raw = generate_synthetic_hotspots(n=2000, seed=1)
    enriched = generate_synthetic_context(raw, seed=2)

    print("Training classifier...\n")
    clf = HotspotClassifier()
    clf.train(enriched)

    print("\nFeature importance:")
    print(clf.feature_importance().to_string(index=False))

    clf.save("model.joblib")
    print("\nModel saved to model.joblib")
