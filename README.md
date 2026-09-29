# THERMOSENTINEL
### AI-Based Detection and Classification of Industrial Fires and Persistent Thermal Sources
**Smart India Hackathon 2026 · SIH26162 · Team HackCode**

---

## The Problem

NASA FIRMS (Fire Information for Resource Management System) gives near-real-time
thermal anomaly detections from satellites (MODIS/VIIRS). But not every "hotspot" is
a wildfire — refineries, steel plants, gas flares, brick kilns, and crop burning all
show up as thermal anomalies too. This creates alert fatigue: emergency responders
can't tell a genuine, spreading wildfire from a factory that's *always* hot.

## What THERMOSENTINEL Does

It classifies every hotspot into one of five categories and assigns a response priority:

| Class | Meaning | Priority |
|---|---|---|
| `WILDFIRE` | Genuine vegetation/forest fire | CRITICAL |
| `INDUSTRIAL_FIRE` | Active fire/incident at an industrial site | HIGH |
| `AGRI_BURNING` | Crop residue / agricultural burning | MEDIUM |
| `PERSISTENT_THERMAL` | Recurring non-emergency heat source (flare stack, kiln) | LOW |
| `FALSE_ALARM` | Low-confidence sensor artifact | IGNORE |

## How It Works

```
NASA FIRMS API  ──┐
  (thermal hotspots,        │
   FRP, brightness)         │
                             ▼
OpenStreetMap Overpass ──► Feature Engineering ──► RandomForest ──► Classified
  (land-use context:        (FRP + persistence +      Classifier      Hotspots
   industrial zone?          OSM context +                              │
   forest? farmland?)        confidence)                                ▼
                                                                   Live Map Dashboard
Persistence Tracking ─────┘                                       (Flask + Leaflet)
  (does this hotspot
   reappear day after day
   at the same spot?)
```

**Key signal used for classification:**
1. **FRP (Fire Radiative Power)** — intensity of the thermal signal in megawatts
2. **Persistence** — a spot that lights up every single day for 3 weeks is a flare
   stack, not a wildfire; a spot that appears once and grows is likely a real fire
3. **OSM land-use context** — is this hotspot sitting inside an industrial zone,
   a forest reserve, or farmland?
4. **Detection confidence** — FIRMS' own low/nominal/high confidence flag

## Project Structure

```
thermosentinel/
├── firms_client.py     # Pulls hotspot data from NASA FIRMS API
├── osm_context.py      # Enriches hotspots with OSM land-use context
├── classifier.py        # RandomForest model: features → classification
├── pipeline.py           # Orchestrates: fetch → enrich → classify → export
├── app.py                # Flask API + dashboard server
├── static/index.html     # Mission-control style live map dashboard
├── model.joblib           # Pre-trained classifier
├── requirements.txt
└── README.md
```

## Running It

```bash
pip install -r requirements.txt
python3 app.py
```

Then open **http://localhost:5000** — you'll see the live dashboard with classified
hotspots on a dark map, a priority queue sidebar, and toggleable classification layers.

By default it runs on **synthetic offline data** (realistic simulated hotspots) so it
works immediately with no API keys. This is because the sandbox this was built in has
no internet access — but the architecture is real and production-ready.

## Going Live (Real Data)

To switch from synthetic demo data to live satellite data:

1. **Get a free NASA FIRMS API key**: https://firms.modaps.eosdis.nasa.gov/api/
2. In `pipeline.py`, call:
   ```python
   run_pipeline(use_live_data=True, firms_key="YOUR_KEY_HERE")
   ```
3. OSM context already uses the real public Overpass API (`overpass-api.de`) when
   `use_live_data=True` — no key needed, just internet access.

That's it — same pipeline, same classifier, same dashboard, real data.

## Model Details

- **Algorithm**: RandomForestClassifier (scikit-learn), 200 trees, balanced class weights
- **Features**: FRP, brightness, persistence_days, near_industrial, near_forest,
  near_agri, near_urban, confidence_score
- **Training data**: Currently synthetic (2000 samples) built from realistic
  distributions per source type. For the real submission, this should be retrained on:
  - Historical FIRMS detections matched against known industrial facility locations
    (CPCB/SPCB industrial registries, OSM industrial polygons)
  - Verified wildfire incident records (Forest Survey of India / state forest depts.)
  - Verified false-positive logs from FIRMS itself

## Why This Matters (for the pitch)

- Cuts down false alarms flooding disaster response teams
- Separates "needs a fire truck NOW" from "that's just the steel plant, as usual"
- Persistent thermal source tracking can double as **industrial emissions monitoring**
  (repeated flaring, non-compliant kilns) — a bonus use case for pollution control boards
- Built entirely on free/open data: NASA FIRMS (free), OpenStreetMap (free), no
  proprietary satellite subscriptions needed — keeps it deployable at national scale

## Next Steps / Roadmap

- [ ] Swap synthetic training labels for real historical incident-matched data
- [ ] Add Sentinel-2 imagery overlay for visual confirmation of large events
- [ ] SMS/push alert integration for CRITICAL and HIGH priority detections
- [ ] Time-series view per hotspot (trend of FRP over days — growing vs. stable)
- [ ] District/state-level aggregation view for state disaster management authorities
