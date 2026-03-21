"""
src/features.py
Feature engineering pipeline for wildfire risk prediction.

Key feature groups:
  1. Raw AEF embeddings (64 dims) — the foundation model output
  2. Temporal delta features — year-over-year embedding drift (captures drying)
  3. Spatial lag features — neighborhood mean/std (local context)
  4. Historical fire frequency — FIRMS-derived fire history (3/5/7 yr windows)
  5. Topographic features — elevation, slope, aspect (from SRTM)
  6. Derived metrics — embedding norm, PCA projections for interpretability
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import BallTree
from typing import List, Dict, Tuple, Optional
from loguru import logger


EMBEDDING_COLS = [f"embedding_{i}" for i in range(64)]
DELTA_COLS = [f"delta_{i}" for i in range(64)]


# ─────────────────────────────────────────────────────────────────
# Core Feature Builder
# ─────────────────────────────────────────────────────────────────

class WildfireFeatureBuilder:
    """
    Transforms raw AEF embeddings + ancillary data into an ML-ready feature matrix.
    
    Usage:
        builder = WildfireFeatureBuilder()
        X_train, feature_names = builder.fit_transform(df_train)
        X_test = builder.transform(df_test)
    """
    
    def __init__(
        self,
        use_temporal_delta: bool = True,
        use_spatial_lag: bool = True,
        use_embedding_norm: bool = True,
        spatial_lag_k: int = 9,          # k nearest neighbors for spatial lag
        pca_components: int = 16,         # PCA for interpretability only, not training
    ):
        self.use_temporal_delta = use_temporal_delta
        self.use_spatial_lag = use_spatial_lag
        self.use_embedding_norm = use_embedding_norm
        self.spatial_lag_k = spatial_lag_k
        self.pca_components = pca_components
        
        self.embedding_scaler = StandardScaler()
        self.pca = PCA(n_components=pca_components, random_state=42)
        self._fitted = False
        self.feature_names_: List[str] = []
    
    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Fit on training data and transform."""
        embeddings = df[EMBEDDING_COLS].values
        
        # Fit embedding scaler
        self.embedding_scaler.fit(embeddings)
        
        # Fit PCA (for visualization only — not used as training features)
        self.pca.fit(embeddings)
        
        self._fitted = True
        return self.transform(df)
    
    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Transform DataFrame to feature matrix."""
        if not self._fitted:
            raise RuntimeError("Call fit_transform() first.")
        
        features = {}
        
        # ── 1. Scaled AEF embeddings ───────────────────────────────
        embeddings = df[EMBEDDING_COLS].values
        embeddings_scaled = self.embedding_scaler.transform(embeddings)
        for i, col in enumerate(EMBEDDING_COLS):
            features[f"emb_{i}"] = embeddings_scaled[:, i]
        
        # ── 2. Temporal delta features ─────────────────────────────
        if self.use_temporal_delta and all(c in df.columns for c in DELTA_COLS):
            deltas = df[DELTA_COLS].values
            for i in range(64):
                features[f"delta_{i}"] = deltas[:, i]
            # Summary delta statistics
            features["delta_norm"] = np.linalg.norm(deltas, axis=1)
            features["delta_mean"] = deltas.mean(axis=1)
            features["delta_pos_frac"] = (deltas > 0).mean(axis=1)  # Fraction of dims increasing (drying)
            logger.debug("Added temporal delta features (64 + 3 summary)")
        
        # ── 3. Spatial lag features ────────────────────────────────
        if self.use_spatial_lag and "lat" in df.columns:
            spatial_feats = self._compute_spatial_lags(df, embeddings_scaled)
            features.update(spatial_feats)
        
        # ── 4. Embedding summary statistics ───────────────────────
        if self.use_embedding_norm:
            features["emb_norm"] = np.linalg.norm(embeddings_scaled, axis=1)
            features["emb_mean"] = embeddings_scaled.mean(axis=1)
            features["emb_std"] = embeddings_scaled.std(axis=1)
            features["emb_max"] = embeddings_scaled.max(axis=1)
            features["emb_min"] = embeddings_scaled.min(axis=1)
            # Percentile features
            features["emb_p25"] = np.percentile(embeddings_scaled, 25, axis=1)
            features["emb_p75"] = np.percentile(embeddings_scaled, 75, axis=1)
        
        # ── 5. Historical fire frequency ───────────────────────────
        for window in [3, 5, 7]:
            col = f"fire_count_{window}yr"
            if col in df.columns:
                features[f"fire_count_{window}yr"] = df[col].fillna(0).values
                features[f"fire_any_{window}yr"] = (df[col].fillna(0) > 0).astype(int).values
        
        if "last_fire_year" in df.columns:
            current_year = df["year"].max() if "year" in df.columns else 2023
            features["years_since_fire"] = (current_year - df["last_fire_year"].fillna(0)).clip(0, 20).values
        
        # ── 6. Topographic features ────────────────────────────────
        for topo_col in ["elevation", "slope", "aspect"]:
            if topo_col in df.columns:
                features[topo_col] = df[topo_col].fillna(df[topo_col].median()).values
        
        # Derived topographic
        if "slope" in df.columns and "aspect" in df.columns:
            # Southward-facing slopes burn more (in Northern Hemisphere)
            aspect_rad = np.deg2rad(df["aspect"].fillna(0))
            features["southward_slope"] = np.cos(aspect_rad) * df["slope"].fillna(0)
        
        # ── 7. Seasonal/temporal ──────────────────────────────────
        if "year" in df.columns:
            features["year"] = df["year"].values
        
        # ── Build matrix ───────────────────────────────────────────
        feature_df = pd.DataFrame(features)
        feature_df = feature_df.fillna(0)
        
        self.feature_names_ = list(feature_df.columns)
        return feature_df.values, self.feature_names_
    
    def _compute_spatial_lags(
        self,
        df: pd.DataFrame,
        embeddings_scaled: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        Compute spatial lag features using k nearest neighbors.
        
        For each pixel, finds its k spatial neighbors and computes:
          - Mean of each embedding dimension across neighbors
          - Std of each embedding dimension across neighbors
          - Mean delta norm in neighborhood
        
        This captures local landscape context — a dry pixel surrounded by
        dry neighbors is much higher risk than an isolated dry pixel.
        """
        logger.debug(f"Computing spatial lags (k={self.spatial_lag_k})...")
        
        # BallTree on lat/lon (haversine metric)
        coords = np.deg2rad(df[["lat", "lon"]].values)
        tree = BallTree(coords, metric="haversine")
        
        _, indices = tree.query(coords, k=self.spatial_lag_k + 1)
        indices = indices[:, 1:]  # Exclude self
        
        # Mean and std of embedding in neighborhood
        neighbor_embeddings = embeddings_scaled[indices]  # (N, k, 64)
        
        spatial_feats = {}
        
        # Full 64-dim neighborhood mean (adds 64 more features)
        neighbor_mean = neighbor_embeddings.mean(axis=1)
        for i in range(64):
            spatial_feats[f"spatial_mean_{i}"] = neighbor_mean[:, i]
        
        # Neighborhood diversity (std across all embedding dims, averaged)
        spatial_feats["spatial_emb_std_mean"] = neighbor_embeddings.std(axis=(1, 2))
        
        # Neighborhood embedding norm
        spatial_feats["spatial_norm_mean"] = np.linalg.norm(
            neighbor_embeddings, axis=2
        ).mean(axis=1)
        
        return spatial_feats
    
    def get_pca_projections(self, df: pd.DataFrame) -> np.ndarray:
        """
        Project embeddings to PCA space for visualization.
        NOT used in model training — used for EDA and cluster plots.
        Returns (N, n_components) array.
        """
        embeddings = df[EMBEDDING_COLS].values
        return self.pca.transform(embeddings)
    
    def explained_variance_summary(self) -> pd.DataFrame:
        """How much variance does each PCA component explain?"""
        evr = self.pca.explained_variance_ratio_
        return pd.DataFrame({
            "component": range(1, len(evr) + 1),
            "explained_var": evr,
            "cumulative": np.cumsum(evr),
        })


# ─────────────────────────────────────────────────────────────────
# Temporal Delta Computation
# ─────────────────────────────────────────────────────────────────

def compute_temporal_deltas(
    df_current: pd.DataFrame,
    df_previous: pd.DataFrame,
    join_on: List[str] = ["lat", "lon"],
) -> pd.DataFrame:
    """
    Compute year-over-year embedding changes for each pixel.
    
    Delta = current_embedding - previous_embedding
    
    A positive delta in most dimensions = pixel moving toward
    "brighter/drier" signature in the foundation model's latent space.
    
    Args:
        df_current: AEF data for year t
        df_previous: AEF data for year t-1 (same spatial locations)
        join_on: Columns to merge on (lat/lon, rounded to grid)
    
    Returns:
        df_current with added delta_0 … delta_63 columns
    """
    # Round coordinates to avoid floating point mismatches
    for df in [df_current, df_previous]:
        df["lat_r"] = df["lat"].round(4)
        df["lon_r"] = df["lon"].round(4)
    
    prev_emb = df_previous[["lat_r", "lon_r"] + EMBEDDING_COLS].copy()
    prev_emb.columns = ["lat_r", "lon_r"] + [f"prev_{c}" for c in EMBEDDING_COLS]
    
    merged = df_current.merge(prev_emb, on=["lat_r", "lon_r"], how="left")
    
    # Compute deltas
    for i in range(64):
        curr_col = f"embedding_{i}"
        prev_col = f"prev_embedding_{i}"
        if prev_col in merged.columns:
            merged[f"delta_{i}"] = merged[curr_col] - merged[prev_col]
        else:
            merged[f"delta_{i}"] = 0.0
    
    merged.drop(columns=["lat_r", "lon_r"] + [c for c in merged.columns if c.startswith("prev_")],
                inplace=True, errors="ignore")
    
    valid_deltas = merged[DELTA_COLS].notna().all(axis=1).sum()
    logger.info(f"Temporal deltas computed: {valid_deltas}/{len(merged)} valid pixels")
    
    return merged


def compute_multi_year_drift(
    df_dict: Dict[int, pd.DataFrame],
    reference_year: int,
    windows: List[int] = [1, 2, 3],
) -> pd.DataFrame:
    """
    Compute embedding drift over multiple time windows.
    
    For each pixel in reference_year, computes deltas vs reference_year-1,
    reference_year-2, and reference_year-3.
    
    Pixels showing consistent directional drift = systematic drying/greening trend.
    """
    df_ref = df_dict[reference_year].copy()
    
    for window in windows:
        compare_year = reference_year - window
        if compare_year not in df_dict:
            logger.warning(f"Year {compare_year} not in df_dict, skipping {window}-yr window")
            continue
        
        df_past = df_dict[compare_year]
        
        # Merge and compute delta
        past_emb = df_past[["lat", "lon"] + EMBEDDING_COLS].copy()
        past_emb.columns = ["lat", "lon"] + [f"e{window}yr_{i}" for i in range(64)]
        
        df_ref["lat_r"] = df_ref["lat"].round(4)
        df_ref["lon_r"] = df_ref["lon"].round(4)
        past_emb["lat_r"] = past_emb["lat"].round(4)
        past_emb["lon_r"] = past_emb["lon"].round(4)
        
        df_ref = df_ref.merge(
            past_emb.drop(columns=["lat", "lon"]),
            on=["lat_r", "lon_r"],
            how="left",
        )
        
        # Summary drift magnitude
        delta_cols_win = [f"e{window}yr_{i}" for i in range(64)]
        current_embs = df_ref[EMBEDDING_COLS].values
        past_embs = df_ref[delta_cols_win].fillna(0).values
        
        drift = current_embs - past_embs
        df_ref[f"drift_norm_{window}yr"] = np.linalg.norm(drift, axis=1)
        df_ref[f"drift_mean_{window}yr"] = drift.mean(axis=1)
        df_ref[f"drift_positive_{window}yr"] = (drift > 0).mean(axis=1)  # Drying indicator
        
        df_ref.drop(columns=delta_cols_win + ["lat_r", "lon_r"], inplace=True, errors="ignore")
    
    logger.info(f"Multi-year drift computed for {len(windows)} windows")
    return df_ref


# ─────────────────────────────────────────────────────────────────
# Label Creation
# ─────────────────────────────────────────────────────────────────

def assign_fire_labels(
    pixel_df: pd.DataFrame,
    fires_gdf,
    label_year: int,
    buffer_m: float = 200,
    fire_free_buffer_m: float = 5000,
) -> pd.DataFrame:
    """
    Assign binary fire labels to each pixel based on FIRMS detections.
    
    Strategy:
      - label=1: pixel is within buffer_m of any fire detection in label_year
      - label=0: pixel is further than fire_free_buffer_m from any fire detection
      - Drop pixels in the ambiguous zone (200m–5km)
    
    This creates a clean margin between fire/non-fire examples, reducing
    label noise from the ~375m MODIS spatial resolution.
    
    Args:
        pixel_df: DataFrame with lat, lon columns
        fires_gdf: GeoDataFrame of fire detections filtered to label_year
        label_year: Year to assign labels from
        buffer_m: Radius around fire to assign label=1
        fire_free_buffer_m: Minimum distance from any fire to assign label=0
    
    Returns:
        pixel_df with 'label' column added (ambiguous pixels removed)
    """
    import geopandas as gpd
    from shapely.geometry import Point
    
    if fires_gdf.empty:
        logger.warning(f"No fire data for year {label_year}")
        pixel_df["label"] = 0
        return pixel_df
    
    year_fires = fires_gdf[fires_gdf["year"] == label_year]
    if year_fires.empty:
        pixel_df["label"] = 0
        return pixel_df
    
    # Convert pixel locations to GeoDataFrame
    pixel_gdf = gpd.GeoDataFrame(
        pixel_df,
        geometry=gpd.points_from_xy(pixel_df["lon"], pixel_df["lat"]),
        crs="EPSG:4326",
    ).to_crs("EPSG:3857")  # Project to meters for distance calcs
    
    year_fires_m = year_fires.to_crs("EPSG:3857")
    fire_union = year_fires_m.geometry.unary_union
    
    # Distance to nearest fire
    distances = pixel_gdf.geometry.distance(fire_union)
    
    # Assign labels
    fire_mask = distances <= buffer_m
    no_fire_mask = distances >= fire_free_buffer_m
    
    pixel_df = pixel_df.copy()
    pixel_df["label"] = -1  # Ambiguous
    pixel_df.loc[pixel_gdf.index[fire_mask], "label"] = 1
    pixel_df.loc[pixel_gdf.index[no_fire_mask & ~fire_mask], "label"] = 0
    
    # Drop ambiguous
    original_len = len(pixel_df)
    pixel_df = pixel_df[pixel_df["label"] != -1].reset_index(drop=True)
    
    n_fire = (pixel_df["label"] == 1).sum()
    n_nofire = (pixel_df["label"] == 0).sum()
    logger.info(
        f"Labels assigned for {label_year}: {n_fire} fire, {n_nofire} non-fire "
        f"({original_len - len(pixel_df)} ambiguous removed)"
    )
    return pixel_df
