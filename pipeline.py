"""
pipeline.py
-----------
End-to-end pipeline:
  1. Fetch hotspots (NASA FIRMS, or synthetic offline data)
  2. Enrich with OSM land-use context
  3. Track persistence (repeat detections at same location across days)
  4. Classify each hotspot
  5. Output a clean GeoJSON-ready dataframe for the dashboard

Run:  python3 pipeline.py
"""

import os
import pandas as pd
from firms_client import FirmsClient, generate_synthetic_hotspots
from osm_context import OSMContext, generate_synthetic_context
from classifier import HotspotClassifier

MODEL_PATH = "model.joblib"


def run_pipeline(use_live_data: bool = False, firms_key: str = None, n_synthetic: int = 80):
    # --- Step 1: Fetch hotspots ---
    if use_live_data and firms_key:
        client = FirmsClient(map_key=firms_key)
        hotspots = client.get_hotspots(country="IND", sensor="VIIRS_SNPP_NRT", days=1)
        print(f"Fetched {len(hotspots)} live hotspots from NASA FIRMS.")
    else:
        hotspots = generate_synthetic_hotspots(n=n_synthetic)
        print(f"Using {len(hotspots)} synthetic hotspots (offline demo mode).")

    # --- Step 2: OSM enrichment ---
    if use_live_data:
        osm = OSMContext()
        # NOTE: real Overpass calls are done per-point; batched here for clarity.
        records = []
        for _, row in hotspots.iterrows():
            ctx = osm.get_context(row["latitude"], row["longitude"])
            records.append(ctx)
        ctx_df = pd.DataFrame(records)
        enriched = pd.concat([hotspots.reset_index(drop=True), ctx_df], axis=1)
    else:
        enriched = generate_synthetic_context(hotspots)

    # --- Step 3: Classify ---
    clf = HotspotClassifier()
    if os.path.exists(MODEL_PATH):
        clf.load(MODEL_PATH)
    else:
        print("No saved model found — training a fresh one on synthetic data...")
        train_data = generate_synthetic_context(generate_synthetic_hotspots(n=2000, seed=1), seed=2)
        clf.train(train_data, verbose=False)
        clf.save(MODEL_PATH)

    results = clf.predict(enriched)

    # --- Step 4: Clean output for dashboard ---
    output_cols = [
        "id", "latitude", "longitude", "frp", "brightness", "confidence",
        "persistence_days", "near_industrial", "near_forest", "near_agri",
        "predicted_class", "risk_priority", "confidence_score",
        "acq_date", "acq_time", "satellite",
    ]
    output_cols = [c for c in output_cols if c in results.columns]
    final = results[output_cols].sort_values("risk_priority")

    return final


if __name__ == "__main__":
    df = run_pipeline(use_live_data=False, n_synthetic=100)
    print("\n--- Classification Summary ---")
    print(df["predicted_class"].value_counts())
    print("\n--- Sample Output ---")
    print(df.head(15).to_string(index=False))

    df.to_json("hotspots_output.json", orient="records", indent=2)
    df.to_csv("hotspots_output.csv", index=False)
    print("\nSaved: hotspots_output.json, hotspots_output.csv")
