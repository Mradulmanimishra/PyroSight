<div align="center">

# 🔥 PyroSight

### Wildfire Risk Prediction · AlphaEarth Foundation Embeddings · NASA FIRMS

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-FF6600?style=flat-square)](https://xgboost.readthedocs.io)
[![Google Earth Engine](https://img.shields.io/badge/Google%20Earth%20Engine-API-4285F4?style=flat-square&logo=google&logoColor=white)](https://earthengine.google.com)
[![SHAP](https://img.shields.io/badge/SHAP-Explainability-8A2BE2?style=flat-square)](https://shap.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-22C55E?style=flat-square)](LICENSE)

<br/>

> **Predicts wildfire risk zones at 10m resolution** by combining Google DeepMind's AlphaEarth Foundation embeddings — 64-dimensional compressed signatures of a full year of Sentinel-1/2, Landsat 8/9, GEDI LiDAR, and ERA5 climate data — with NASA FIRMS historical fire detections.

<br/>

</div>

---

## What makes this different

Most wildfire ML projects use raw spectral bands (NDVI, NBR) or weather variables. PyroSight operates in **foundation model latent space** — the AlphaEarth embeddings compress a full year of multi-sensor satellite data into 64 dimensions per pixel, pre-trained by Google DeepMind on petabytes of imagery.

Three design decisions that separate this from a tutorial project:

| Decision | Why it matters |
|---|---|
| **Blocked spatial cross-validation** | Standard k-fold inflates AUC-ROC by 15–30% due to spatial autocorrelation. Blocked CV withholds entire geographic regions per fold. |
| **Temporal delta features** | Year-over-year embedding drift captures vegetation drying trends that static embeddings miss entirely. |
| **SHAP TreeExplainer** | Exact Shapley values at inference time. Insurance and government clients need to know *why* a pixel is extreme risk. |

---

## Data sources

| Source | What | Resolution | Years |
|---|---|---|---|
| `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL` | 64-dim AEF per pixel | 10m native | 2017–2024 |
| NASA FIRMS MODIS Collection 6.1 | Active fire detections | ~375m | 2000–present |
| NASA FIRMS VIIRS SNPP | Higher-res fire detections | ~375m | 2012–present |
| USGS SRTM | Elevation, slope, aspect | 30m | Static |

---

## Architecture

```
AlphaEarth Embedding (64-dim, year t)
  + Temporal Δ (embedding_t − embedding_t-1)          ← captures drying trend
  + Spatial lag features (3×3 neighborhood mean/std)  ← local landscape context
  + Historical fire frequency (3yr / 5yr / 7yr)       ← from NASA FIRMS
  + Topographic features (elevation, slope, aspect)   ← SRTM via GEE
          ↓
  XGBoostClassifier (55%) + LightGBMClassifier (45%)
          ↓
  Calibrated ensemble probabilities
          ↓
  Risk tiers: Low · Medium · High · Extreme
          ↓
  GeoTIFF risk map + per-pixel SHAP attribution layer
```

---

## Project structure

```
PyroSight/
├── config.py                    ← regions, hyperparameters, GEE asset IDs
├── requirements.txt
├── main.py                      ← CLI pipeline (dev / train / predict / evaluate)
├── src/
│   ├── data_loader.py           ← GEE AEF fetching + NASA FIRMS API
│   ├── features.py              ← temporal deltas, spatial lags, label assignment
│   ├── model.py                 ← ensemble, blocked spatial CV, SHAP explainer
│   └── visualize.py            ← risk maps, PCA plots, dryness trends, eval dashboard
└── notebooks/
    └── 01_eda_and_modeling.ipynb
```

---

## Quickstart

### Option A — dev mode (no GEE needed, runs in ~2 min)

```bash
git clone https://github.com/Mradulmanimishra/PyroSight.git
cd PyroSight
pip install -r requirements.txt
python main.py --task dev
```

Runs on synthetic data that mirrors real AEF structure. Validates the full pipeline end-to-end.

### Option B — real satellite data

```bash
# 1. Authenticate with Google Earth Engine
earthengine authenticate

# 2. Get a free NASA FIRMS API key at https://firms.modaps.eosdis.nasa.gov/api/

# 3. Set GEE_PROJECT and FIRMS_API_KEY in config.py

# 4. Train on California 2017–2024
python main.py --task train --region california

# 5. Generate 2024 risk map
python main.py --task predict --region california --year 2024
```

---

## Supported regions

| Region | bbox | CRS |
|---|---|---|
| California | [-124.5, 32.5, -113.9, 42.1] | UTM 10N |
| Eastern Australia | [148.0, -38.0, 154.0, -25.0] | UTM 55S |
| Mediterranean | [-10.0, 35.0, 40.0, 48.0] | UTM 33N |
| Amazon arc | [-73.0, -18.0, -44.0, -3.0] | UTM 21S |

---

## Evaluation metrics

| Metric | Why |
|---|---|
| AUC-ROC | Overall discrimination |
| PR-AUC | Primary metric under 30:1 class imbalance |
| Brier Score | Calibration quality — probabilities must be real for pricing |
| Blocked spatial CV | 100km blocks — honest evaluation, not inflated k-fold |

---

## Outputs

Every run produces:

```
outputs/
├── risk_map.png            ← static geospatial heatmap (Low → Extreme)
├── risk_tiers.png          ← 4-class categorical map
├── risk_map.html           ← interactive Folium map
├── evaluation.png          ← ROC + PR + calibration + spatial CV dashboard
├── shap_summary.png        ← feature importance beeswarm
├── feature_groups.png      ← AEF vs other signals contribution
├── predictions.csv         ← lat, lon, fire_prob, risk_tier per pixel
└── models/
    └── wildfire_ensemble.pkl
```

---

## Real-world applications

| Sector | Use case |
|---|---|
| Insurance | Price wildfire risk into property premiums at parcel level |
| Emergency services | Pre-position resources before fire season peaks |
| Land management | Prioritize prescribed burns by risk tier |
| Climate finance | Quantify carbon risk in forested asset portfolios |

---

## Extending this project

- Swap XGBoost for a GNN — model pixel neighborhoods as a spatial graph
- Add real-time inference — stream daily FIRMS detections, update risk continuously
- Add conformal prediction — uncertainty intervals per pixel
- Fine-tune per region — California fire behavior differs from Australian or Mediterranean

---

## License

MIT — free to use, modify, and build on.

---

<div align="center">
Built by <a href="https://github.com/Mradulmanimishra">Mradul Mani Mishra</a>
<br/><br/>
⭐ Star this repo if you found it useful
</div>
