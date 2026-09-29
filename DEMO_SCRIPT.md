# THERMOSENTINEL — Demo Script
### SIH26162 · Team HackCode
Target length: 3–4 minutes (adjust pacing to your round's time limit)

---

## 0:00 – 0:20 | Hook + Problem

> "Every day, NASA's satellites detect hundreds of 'fire' hotspots across India. But here's the problem — most of them aren't wildfires at all. They're steel plants, oil refineries, gas flares, brick kilns. Emergency teams can't tell the difference in real time, and that alert fatigue costs precious response minutes during actual wildfire emergencies."

**[Show Slide 2 — The Problem]**

---

## 0:20 – 0:50 | Solution Overview

> "We built THERMOSENTINEL — an AI system that automatically classifies every satellite hotspot into five categories: genuine wildfires, industrial fires, agricultural burning, persistent thermal sources like factory flares, and false alarms. Each gets a priority level, so responders know exactly what needs attention right now."

**[Show Slide 3 — Our Solution / Classification Types]**

---

## 0:50 – 1:30 | How It Works (Architecture)

> "Here's how it works. We pull thermal hotspot data from NASA FIRMS — that's free, publicly available satellite data. We cross-reference each hotspot's location against OpenStreetMap to understand the context — is this sitting inside an industrial zone, a forest, or farmland? Then we track persistence — does this spot show up every single day for weeks? That's a strong signal it's a flare stack, not a wildfire. All of this feeds into a RandomForest classifier we trained, which outputs a classification and priority level through a Flask API."

**[Show Slide 4 — System Architecture]**

---

## 1:30 – 2:30 | Live Demo

> "Let me show you the actual working prototype."

**[Switch to live dashboard — localhost:5000]**

Talking points while demoing:
1. Point out the **live map** — colored markers, sized by fire intensity (FRP)
2. Click a **CRITICAL (red) marker** — show the popup with FRP, confidence, persistence days, OSM context
3. Click a **LOW priority (cyan) marker** — contrast it: "See — this one's been persistent for 18 days, sitting right in an industrial zone. That's not an emergency, that's a factory."
4. Toggle a **classification layer** off/on in the sidebar to show filtering
5. Point to the **priority queue** — "Responders see critical items first, automatically sorted."

> "This isn't a mockup — it's a real Flask backend, a trained ML model, and a live map, all working end to end right now."

---

## 2:30 – 3:00 | Impact

> "This means faster wildfire response because false alarms are filtered out automatically. It also doubles as an industrial emissions monitor — repeated flaring or non-compliant kilns get flagged for pollution control boards. And because it's built entirely on free NASA and OpenStreetMap data, any state disaster management authority can deploy it at zero licensing cost."

**[Show Slide 7 — Impact & Use Cases]**

---

## 3:00 – 3:20 | Roadmap + Close

> "Right now we're running on realistic simulated data to prove the full pipeline works end to end. Next step: connect live FIRMS and OSM APIs, and retrain our classifier on real historical incident data. Long-term, we're looking at SMS alerts for critical detections and state-level dashboards."

**[Show Slide 9 — Roadmap]**

> "Team HackCode, SIH26162 — THERMOSENTINEL. Thank you."

**[Show Slide 10 — Thank You]**

---

## Tips for Recording

- **Screen record the live dashboard** at 1080p — run `python app.py`, open `localhost:5000`, resize browser window cleanly before recording
- Let the map **fully load** before starting narration on that segment
- If judges ask "is this real data?" — be upfront: "The pipeline and model are fully real and working; we're using realistic simulated data for the demo since live API keys need to be provisioned per deployment. Swapping to live NASA FIRMS data is a one-line config change — the code is already written for it."
- Have the **README.md** open in a second tab in case judges want to see code/architecture during Q&A
