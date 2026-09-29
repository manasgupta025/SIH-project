"""
osm_context.py
---------------
Enriches each FIRMS hotspot with land-use / infrastructure context pulled
from OpenStreetMap (via the Overpass API), so the classifier can tell the
difference between "hotspot in the middle of a steel plant" vs
"hotspot in the middle of a forest".

Real usage:
    ctx = OSMContext()
    tags = ctx.get_context(lat, lon, radius_m=500)
    # -> {'near_industrial': True, 'near_forest': False, 'nearest_feature': 'industrial=steel_plant', 'distance_m': 120}

Offline fallback:
    Since Overpass requires internet access, generate_synthetic_context()
    produces plausible OSM-style context tags for the synthetic hotspot set,
    keeping the interface identical so swapping in the real API later is a
    one-line change.
"""

import time
import requests
import numpy as np
import pandas as pd

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# OSM tags that matter for this problem
INDUSTRIAL_TAGS = [
    'landuse=industrial', 'man_made=works', 'industrial=refinery',
    'industrial=steel_plant', 'industrial=brick_kiln', 'industrial=oil',
    'man_made=chimney', 'man_made=flare', 'power=plant',
]
FOREST_TAGS = ['landuse=forest', 'natural=wood']
AGRI_TAGS = ['landuse=farmland', 'landuse=orchard']
URBAN_TAGS = ['landuse=residential', 'landuse=commercial']


class OSMContext:
    def __init__(self, use_overpass: bool = True):
        self.use_overpass = use_overpass

    def get_context(self, lat: float, lon: float, radius_m: int = 500) -> dict:
        """Query Overpass API for features within radius_m of (lat, lon)."""
        query = f"""
        [out:json][timeout:25];
        (
          node["landuse"](around:{radius_m},{lat},{lon});
          way["landuse"](around:{radius_m},{lat},{lon});
          node["man_made"](around:{radius_m},{lat},{lon});
          way["man_made"](around:{radius_m},{lat},{lon});
          node["industrial"](around:{radius_m},{lat},{lon});
          way["industrial"](around:{radius_m},{lat},{lon});
        );
        out center tags;
        """
        resp = requests.post(OVERPASS_URL, data={"data": query}, timeout=30)
        resp.raise_for_status()
        elements = resp.json().get("elements", [])
        return self._summarize(elements)

    @staticmethod
    def _summarize(elements: list) -> dict:
        tags_found = []
        for el in elements:
            tags = el.get("tags", {})
            for k in ("landuse", "man_made", "industrial", "power"):
                if k in tags:
                    tags_found.append(f"{k}={tags[k]}")

        near_industrial = any(t in INDUSTRIAL_TAGS or "industrial" in t or "man_made" in t for t in tags_found)
        near_forest = any(t in FOREST_TAGS for t in tags_found)
        near_agri = any(t in AGRI_TAGS for t in tags_found)
        near_urban = any(t in URBAN_TAGS for t in tags_found)

        return {
            "near_industrial": near_industrial,
            "near_forest": near_forest,
            "near_agri": near_agri,
            "near_urban": near_urban,
            "osm_tags": tags_found[:5],
        }


def generate_synthetic_context(hotspots_df: pd.DataFrame, seed: int = 7) -> pd.DataFrame:
    """
    Attaches plausible OSM-style context to each synthetic hotspot, correlated
    with its hidden _true_kind so the resulting features are realistic (not
    pure noise) — mirroring how real OSM context would behave.
    """
    rng = np.random.default_rng(seed)
    df = hotspots_df.copy()

    near_industrial, near_forest, near_agri, near_urban, tags_list = [], [], [], [], []

    for _, row in df.iterrows():
        kind = row.get("_true_kind", "forest")
        # 85% chance the OSM context correctly reflects the true land use (some noise/edge cases)
        correct = rng.random() < 0.85

        if kind == "industrial":
            ni = correct or rng.random() < 0.5
            nf, na, nu = False, False, rng.random() < 0.3
            tags = rng.choice([
                ["industrial=refinery"], ["landuse=industrial"],
                ["man_made=chimney"], ["industrial=steel_plant"], ["man_made=flare"]
            ])
        elif kind == "forest":
            nf = correct or rng.random() < 0.5
            ni, na, nu = False, False, False
            tags = rng.choice([["landuse=forest"], ["natural=wood"]])
        else:  # agri
            na = correct or rng.random() < 0.5
            ni, nf, nu = False, False, rng.random() < 0.2
            tags = rng.choice([["landuse=farmland"], ["landuse=orchard"]])

        near_industrial.append(bool(ni) if kind == "industrial" else bool(rng.random() < 0.05))
        near_forest.append(bool(nf) if kind == "forest" else bool(rng.random() < 0.05))
        near_agri.append(bool(na) if kind == "agri" else bool(rng.random() < 0.05))
        near_urban.append(bool(nu))
        tags_list.append(list(tags) if isinstance(tags, (list, np.ndarray)) else [tags])

    df["near_industrial"] = near_industrial
    df["near_forest"] = near_forest
    df["near_agri"] = near_agri
    df["near_urban"] = near_urban
    df["osm_tags"] = tags_list
    return df


if __name__ == "__main__":
    from firms_client import generate_synthetic_hotspots
    hs = generate_synthetic_hotspots(10)
    enriched = generate_synthetic_context(hs)
    print(enriched[["id", "_true_kind", "near_industrial", "near_forest", "near_agri", "osm_tags"]].to_string())
