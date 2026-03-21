"""
config.py — Central configuration for Wildfire Risk Prediction pipeline.
Edit this file to change regions, years, model hyperparameters, and output paths.
"""

from pathlib import Path

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = ROOT / "outputs"
MODEL_DIR = OUTPUTS_DIR / "models"

for d in [RAW_DIR, PROCESSED_DIR, OUTPUTS_DIR, MODEL_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────
# Google Earth Engine
# ─────────────────────────────────────────────
GEE_PROJECT = "your-gcp-project-id"          # Replace with your GEE project

# AlphaEarth Foundation Model Embeddings
AEF_COLLECTION = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
AEF_EMBEDDING_DIM = 64
AEF_RESOLUTION_M = 10                         # Native 10m resolution
AEF_EXPORT_SCALE = 100                        # Export at 100m for manageability (1000m for dev)
AEF_BAND_PREFIX = "embedding_"               # Bands are named embedding_0 … embedding_63

# For dev/testing: sample at coarser resolution
DEV_SCALE = 500                               # 500m during development
PROD_SCALE = 100                              # 100m in production

# ─────────────────────────────────────────────
# Study regions (GEE geometry format)
# ─────────────────────────────────────────────
REGIONS = {
    "california": {
        "bbox": [-124.5, 32.5, -113.9, 42.1],   # [west, south, east, north]
        "crs": "EPSG:32610",                      # UTM Zone 10N
        "description": "California, USA",
    },
    "australia_east": {
        "bbox": [148.0, -38.0, 154.0, -25.0],
        "crs": "EPSG:32755",
        "description": "Eastern Australia (NSW/QLD)",
    },
    "mediterranean": {
        "bbox": [-10.0, 35.0, 40.0, 48.0],
        "crs": "EPSG:32633",
        "description": "Mediterranean Basin",
    },
    "amazon_arc": {
        "bbox": [-73.0, -18.0, -44.0, -3.0],
        "crs": "EPSG:32721",
        "description": "Amazon Arc of Deforestation",
    },
}
DEFAULT_REGION = "california"

# ─────────────────────────────────────────────
# Time range
# ─────────────────────────────────────────────
YEARS = list(range(2017, 2025))               # 2017–2024 (AEF availability)
FIRE_LOOKBACK_YEARS = [3, 5, 7]              # Historical fire windows for features
FIRE_LABEL_YEAR_OFFSET = 1                    # Predict fires 1 year ahead (t+1)

# ─────────────────────────────────────────────
# NASA FIRMS
# ─────────────────────────────────────────────
FIRMS_API_KEY = "your_firms_api_key"          # Register at https://firms.modaps.eosdis.nasa.gov/api/
FIRMS_SOURCES = ["MODIS_NRT", "VIIRS_SNPP_NRT", "VIIRS_NOAA20_NRT"]
FIRMS_MIN_CONFIDENCE = 50                     # 0–100 (MODIS) or "nominal/high" (VIIRS)
FIRMS_MIN_FRP = 5.0                           # Minimum Fire Radiative Power (MW) to include

# ─────────────────────────────────────────────
# Feature Engineering
# ─────────────────────────────────────────────
SPATIAL_LAG_KERNEL = 3                        # NxN neighborhood for spatial lag features (3=3x3)
EMBEDDING_PCA_COMPONENTS = 16                 # Reduce 64-dim to 16 for visualization (not training)
TEMPORAL_DIFF_YEARS = [1, 2, 3]             # Year gaps for temporal delta features

# ─────────────────────────────────────────────
# Model Hyperparameters (XGBoost)
# ─────────────────────────────────────────────
XGBOOST_PARAMS = {
    "n_estimators": 800,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 10,
    "gamma": 1.0,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "scale_pos_weight": 30,                   # ~30:1 non-fire:fire ratio — tune per region
    "tree_method": "hist",
    "device": "cuda",                         # Falls back to cpu if no GPU
    "eval_metric": ["auc", "aucpr", "logloss"],
    "early_stopping_rounds": 50,
    "random_state": 42,
}

LIGHTGBM_PARAMS = {
    "n_estimators": 800,
    "max_depth": 6,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_samples": 50,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "scale_pos_weight": 30,
    "device": "gpu",
    "random_state": 42,
}

# Ensemble weights (XGB, LGB) — tune via CV
ENSEMBLE_WEIGHTS = [0.55, 0.45]

# ─────────────────────────────────────────────
# Spatial Cross-Validation
# ─────────────────────────────────────────────
SPATIAL_CV_BLOCK_SIZE_KM = 100               # Block size for spatial CV
SPATIAL_CV_N_FOLDS = 5

# ─────────────────────────────────────────────
# Risk Tiers (probability thresholds)
# ─────────────────────────────────────────────
RISK_THRESHOLDS = {
    "low":     (0.00, 0.15),
    "medium":  (0.15, 0.35),
    "high":    (0.35, 0.60),
    "extreme": (0.60, 1.00),
}
RISK_COLORS = {
    "low":     "#4caf50",
    "medium":  "#ff9800",
    "high":    "#f44336",
    "extreme": "#7b1fa2",
}
