"""
firms_client.py
----------------
Fetches active-fire / thermal-anomaly hotspots from NASA FIRMS.

Real usage (once you have a free MAP_KEY from https://firms.modaps.eosdis.nasa.gov/api/):
    client = FirmsClient(map_key="YOUR_KEY")
    df = client.get_hotspots(country="IND", sensor="VIIRS_SNPP_NRT", days=1)

If no key / no internet is available, falls back to a realistic synthetic
dataset so the rest of the pipeline can be built, tested and demoed offline.
"""

import io
import random
import requests
import pandas as pd
import numpy as np


class FirmsClient:
    BASE_URL = "https://firms.modaps.eosdis.nasa.gov/api/country/csv"

    def __init__(self, map_key: str = None):
        self.map_key = map_key

    def get_hotspots(self, country: str = "IND", sensor: str = "VIIRS_SNPP_NRT", days: int = 1) -> pd.DataFrame:
        """
        Fetch real hotspot data from NASA FIRMS CSV API.
        Docs: https://firms.modaps.eosdis.nasa.gov/api/country/
        """
        if not self.map_key:
            raise ValueError("No FIRMS map_key provided. Get a free one at "
                              "https://firms.modaps.eosdis.nasa.gov/api/ "
                              "or use generate_synthetic_hotspots() for offline dev.")

        url = f"{self.BASE_URL}/{self.map_key}/{sensor}/{country}/{days}"
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
        return self._standardize(df)

    @staticmethod
    def _standardize(df: pd.DataFrame) -> pd.DataFrame:
        """Normalize FIRMS column names across sensors (MODIS vs VIIRS differ slightly)."""
        rename_map = {
            "bright_ti4": "brightness",
            "bright_ti5": "brightness2",
            "brightness": "brightness",
            "frp": "frp",  # Fire Radiative Power (MW) - key intensity signal
            "confidence": "confidence",
            "acq_date": "acq_date",
            "acq_time": "acq_time",
            "latitude": "latitude",
            "longitude": "longitude",
            "daynight": "daynight",
            "satellite": "satellite",
        }
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
        return df


def generate_synthetic_hotspots(n: int = 60, seed: int = 42) -> pd.DataFrame:
    """
    Generates a realistic synthetic hotspot dataset mimicking NASA FIRMS VIIRS output,
    centered loosely over Indian industrial + forest regions, for offline development
    and demo purposes.
    """
    rng = np.random.default_rng(seed)

    # Rough real-world anchor points: (lat, lon, label) — mix of industrial belts & forest belts
    anchors = [
        (23.02, 72.57, "industrial"),   # Ahmedabad industrial belt
        (21.17, 72.83, "industrial"),   # Surat / textile & refineries
        (22.30, 73.20, "industrial"),   # Vadodara petrochemical
        (19.07, 72.88, "industrial"),   # Mumbai / Trombay refineries
        (25.45, 81.85, "forest"),       # Vindhya forest belt near Prayagraj
        (23.60, 85.28, "forest"),       # Jharkhand forest/mining
        (21.25, 81.63, "forest"),       # Chhattisgarh forest
        (26.85, 80.95, "agri"),         # UP agricultural belt (crop residue burning)
        (30.90, 75.85, "agri"),         # Punjab stubble burning belt
        (17.68, 83.21, "industrial"),   # Visakhapatnam steel/port
    ]

    rows = []
    for i in range(n):
        base_lat, base_lon, kind = anchors[rng.integers(0, len(anchors))]
        lat = base_lat + rng.normal(0, 0.15)
        lon = base_lon + rng.normal(0, 0.15)

        if kind == "industrial":
            frp = rng.uniform(15, 60)          # steady moderate FRP
            brightness = rng.uniform(320, 345)
            persistence_days = rng.integers(5, 30)  # shows up repeatedly
            confidence = rng.choice(["nominal", "high"], p=[0.4, 0.6])
        elif kind == "forest":
            frp = rng.uniform(40, 250)         # wildfires can spike very high
            brightness = rng.uniform(330, 400)
            persistence_days = rng.integers(1, 4)
            confidence = rng.choice(["nominal", "high", "low"], p=[0.5, 0.3, 0.2])
        else:  # agri burning
            frp = rng.uniform(10, 80)
            brightness = rng.uniform(320, 360)
            persistence_days = rng.integers(1, 3)
            confidence = rng.choice(["nominal", "low"], p=[0.6, 0.4])

        rows.append({
            "id": f"HS{i:04d}",
            "latitude": round(lat, 5),
            "longitude": round(lon, 5),
            "brightness": round(brightness, 1),
            "frp": round(frp, 1),
            "confidence": confidence,
            "acq_date": "2026-08-26",
            "acq_time": f"{rng.integers(0,23):02d}{rng.integers(0,59):02d}",
            "daynight": rng.choice(["D", "N"]),
            "satellite": rng.choice(["SNPP", "NOAA-20", "NOAA-21"]),
            "persistence_days": int(persistence_days),
            "_true_kind": kind,  # hidden ground-truth label, used only to build training data
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = generate_synthetic_hotspots(20)
    print(df.head(10).to_string())
