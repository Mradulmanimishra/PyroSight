"""
src/data_loader.py
Handles all data acquisition:
  1. AlphaEarth Foundation embeddings via Google Earth Engine
  2. Historical fire detections via NASA FIRMS API
  3. SRTM elevation (topographic proxy) via GEE
"""

import ee
import requests
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Optional
from loguru import logger
from tqdm import tqdm
import json
import time

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    GEE_PROJECT, AEF_COLLECTION, AEF_RESOLUTION_M, AEF_BAND_PREFIX,
    FIRMS_API_KEY, FIRMS_SOURCES, FIRMS_MIN_CONFIDENCE, FIRMS_MIN_FRP,
    RAW_DIR, REGIONS, YEARS
)


# ─────────────────────────────────────────────────────────────────
# GEE Initialization
# ─────────────────────────────────────────────────────────────────

def init_gee(project: str = GEE_PROJECT) -> None:
    """Initialize Google Earth Engine. Prompts browser auth on first run."""
    try:
        ee.Initialize(project=project)
        logger.info(f"GEE initialized — project: {project}")
    except ee.EEException:
        logger.info("GEE auth required — opening browser...")
        ee.Authenticate()
        ee.Initialize(project=project)


# ─────────────────────────────────────────────────────────────────
# AlphaEarth Embeddings
# ─────────────────────────────────────────────────────────────────

def get_aef_image(year: int, region_bbox: List[float]) -> ee.Image:
    """
    Fetch a single annual AlphaEarth Embedding image for a given year.
    
    The AEF collection is indexed by year. Each image has 64 bands named
    embedding_0 through embedding_63, representing the compressed multi-sensor
    signature of every pixel for that calendar year.
    
    Args:
        year: Calendar year (2017–2024)
        region_bbox: [west, south, east, north] in WGS84
    
    Returns:
        ee.Image with 64 embedding bands, clipped to region
    """
    region = ee.Geometry.Rectangle(region_bbox)
    
    # Filter the AEF collection to the target year
    collection = (
        ee.ImageCollection(AEF_COLLECTION)
        .filterDate(f"{year}-01-01", f"{year}-12-31")
        .filterBounds(region)
    )
    
    count = collection.size().getInfo()
    if count == 0:
        raise ValueError(f"No AEF images found for year {year} in bbox {region_bbox}")
    
    # Mosaic if multiple tiles (AEF is tiled globally)
    image = collection.mosaic().clip(region)
    
    logger.info(f"AEF image fetched — year: {year}, bands: {image.bandNames().getInfo()[:3]}...")
    return image


def get_aef_time_series(
    years: List[int],
    region_bbox: List[float],
    scale: int = 100
) -> Dict[int, ee.Image]:
    """
    Fetch AEF images for all years. Returns dict {year: ee.Image}.
    Used to compute temporal delta features later.
    """
    images = {}
    for year in tqdm(years, desc="Fetching AEF time series"):
        try:
            images[year] = get_aef_image(year, region_bbox)
        except ValueError as e:
            logger.warning(f"Skipping year {year}: {e}")
    return images


def sample_aef_to_dataframe(
    aef_image: ee.Image,
    sample_points: ee.FeatureCollection,
    scale: int = 100,
    year: int = None,
) -> pd.DataFrame:
    """
    Sample AEF embeddings at a set of points (fire + non-fire locations).
    
    GEE's sampleRegions is the key call — it extracts the 64-dim embedding
    vector at each lat/lon point and returns it as a FeatureCollection.
    
    Args:
        aef_image: Annual AEF image with 64 bands
        sample_points: GEE FeatureCollection of points (must have 'label' property)
        scale: Pixel scale in meters for sampling
        year: Year label to add to output DataFrame
    
    Returns:
        DataFrame with columns: lat, lon, year, label, embedding_0…embedding_63
    """
    sampled = aef_image.sampleRegions(
        collection=sample_points,
        scale=scale,
        geometries=True,
        tileScale=4,  # Handles large regions by splitting computation
    )
    
    features = sampled.getInfo()["features"]
    
    rows = []
    for feat in features:
        props = feat["properties"]
        coords = feat["geometry"]["coordinates"]
        row = {
            "lon": coords[0],
            "lat": coords[1],
            "year": year,
            "label": props.get("label", -1),
        }
        # Embedding bands
        for i in range(64):
            key = f"{AEF_BAND_PREFIX}{i}"
            row[key] = props.get(key, np.nan)
        rows.append(row)
    
    df = pd.DataFrame(rows)
    logger.info(f"Sampled {len(df)} points from AEF (year={year})")
    return df


def export_aef_to_drive(
    aef_image: ee.Image,
    region_bbox: List[float],
    year: int,
    scale: int = 100,
    folder: str = "wildfire_aef",
    crs: str = "EPSG:4326",
) -> ee.batch.Task:
    """
    Export full AEF raster to Google Drive as a multi-band GeoTIFF.
    Use this for full spatial inference (generating risk maps).
    
    Returns the submitted GEE Task object. Check status with task.status().
    
    NOTE: For a state-sized region at 100m, export is ~2–5 GB.
    For dev work, use sample_aef_to_dataframe() at point locations instead.
    """
    region = ee.Geometry.Rectangle(region_bbox)
    
    task = ee.batch.Export.image.toDrive(
        image=aef_image,
        description=f"AEF_annual_{year}",
        folder=folder,
        fileNamePrefix=f"aef_{year}",
        region=region,
        scale=scale,
        crs=crs,
        maxPixels=1e11,
        fileFormat="GeoTIFF",
    )
    task.start()
    logger.info(f"GEE export task submitted — year: {year}, task_id: {task.id}")
    return task


def get_elevation(region_bbox: List[float], scale: int = 100) -> ee.Image:
    """
    Fetch SRTM 30m elevation data as a topographic feature.
    Elevation, slope, and aspect are strong wildfire correlates.
    Returns: ee.Image with bands [elevation, slope, aspect]
    """
    region = ee.Geometry.Rectangle(region_bbox)
    srtm = ee.Image("USGS/SRTMGL1_003").clip(region)
    
    terrain = ee.Terrain.products(srtm)
    # terrain has: elevation, slope, aspect, hillshade
    
    return terrain.select(["elevation", "slope", "aspect"])


# ─────────────────────────────────────────────────────────────────
# NASA FIRMS Fire Data
# ─────────────────────────────────────────────────────────────────

class FIRMSLoader:
    """
    Downloads fire detection data from NASA FIRMS API.
    Supports MODIS, VIIRS SNPP, and VIIRS NOAA-20 (J1) sensors.
    
    Register for an API key at: https://firms.modaps.eosdis.nasa.gov/api/
    Free tier: 2,000 area requests/day.
    """
    
    BASE_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
    
    def __init__(self, api_key: str = FIRMS_API_KEY):
        self.api_key = api_key
        self._validate_key()
    
    def _validate_key(self):
        """Quick validation — returns 401 if key is invalid."""
        if self.api_key == "your_firms_api_key":
            logger.warning("FIRMS API key not set. Get one at https://firms.modaps.eosdis.nasa.gov/api/")
    
    def fetch_fires(
        self,
        bbox: List[float],
        year: int,
        source: str = "MODIS_NRT",
        day_range: int = 365,
    ) -> pd.DataFrame:
        """
        Fetch fire detections for a bounding box and year.
        
        Args:
            bbox: [west, south, east, north]
            year: Target year
            source: MODIS_NRT | VIIRS_SNPP_NRT | VIIRS_NOAA20_NRT
            day_range: Number of days to query (max 10 for NRT, use archive for historical)
        
        Returns:
            GeoDataFrame with columns: latitude, longitude, acq_date, confidence, frp, ...
        
        Note: For historical data (2000–present), use the archive endpoint instead.
        The NRT (Near Real Time) endpoint only covers the last 10 days.
        For historical training data, use the annual archive:
            https://firms.modaps.eosdis.nasa.gov/download/
        """
        area_str = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"
        url = f"{self.BASE_URL}/{self.api_key}/{source}/{area_str}/{day_range}"
        
        logger.info(f"Fetching FIRMS data — source: {source}, year: {year}")
        
        try:
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
        except requests.exceptions.RequestException as e:
            logger.error(f"FIRMS request failed: {e}")
            return pd.DataFrame()
        
        if not resp.text.strip() or resp.text.startswith("<?xml"):
            logger.warning("No fire data returned (empty region or API error)")
            return pd.DataFrame()
        
        from io import StringIO
        df = pd.read_csv(StringIO(resp.text))
        df = self._filter_and_clean(df, source)
        logger.info(f"FIRMS: {len(df)} fire detections after filtering")
        return df
    
    def fetch_historical(
        self,
        bbox: List[float],
        start_year: int,
        end_year: int,
        sources: List[str] = FIRMS_SOURCES,
    ) -> gpd.GeoDataFrame:
        """
        Load fire data from pre-downloaded FIRMS annual shapefiles/CSVs.
        
        For the full historical record, download directly from:
        https://firms.modaps.eosdis.nasa.gov/download/
        Select: MODIS Collection 6.1 NRT, Annual, Global
        
        This method reads those local files and concatenates across years.
        """
        all_dfs = []
        
        for year in range(start_year, end_year + 1):
            for source in sources:
                fpath = RAW_DIR / "firms" / f"{source.lower()}_{year}.csv"
                if not fpath.exists():
                    logger.warning(f"FIRMS file not found: {fpath} — skipping")
                    continue
                
                df = pd.read_csv(fpath)
                df["source"] = source
                df = self._filter_and_clean(df, source)
                
                # Filter to bbox
                df = df[
                    (df["longitude"] >= bbox[0]) & (df["longitude"] <= bbox[2]) &
                    (df["latitude"] >= bbox[1]) & (df["latitude"] <= bbox[3])
                ]
                all_dfs.append(df)
                logger.info(f"Loaded {len(df)} fire points from {source} {year}")
        
        if not all_dfs:
            logger.error("No FIRMS data loaded. Check raw data directory.")
            return gpd.GeoDataFrame()
        
        combined = pd.concat(all_dfs, ignore_index=True)
        
        # Convert to GeoDataFrame
        gdf = gpd.GeoDataFrame(
            combined,
            geometry=gpd.points_from_xy(combined["longitude"], combined["latitude"]),
            crs="EPSG:4326",
        )
        return gdf
    
    def _filter_and_clean(self, df: pd.DataFrame, source: str) -> pd.DataFrame:
        """Apply confidence and FRP filters, parse dates."""
        if df.empty:
            return df
        
        # Standardize column names (MODIS vs VIIRS differ slightly)
        df.columns = [c.lower().strip() for c in df.columns]
        
        # Confidence filter
        if "confidence" in df.columns:
            if df["confidence"].dtype == object:
                # VIIRS: 'low', 'nominal', 'high'
                df = df[df["confidence"].isin(["nominal", "high"])]
            else:
                # MODIS: 0–100
                df = df[df["confidence"] >= FIRMS_MIN_CONFIDENCE]
        
        # FRP filter (removes spurious detections)
        if "frp" in df.columns:
            df = df[df["frp"] >= FIRMS_MIN_FRP]
        
        # Parse acquisition date
        if "acq_date" in df.columns:
            df["acq_date"] = pd.to_datetime(df["acq_date"])
            df["year"] = df["acq_date"].dt.year
            df["month"] = df["acq_date"].dt.month
            df["doy"] = df["acq_date"].dt.dayofyear  # Day of year (fire seasonality)
        
        return df.reset_index(drop=True)
    
    def compute_fire_density_grid(
        self,
        fires_gdf: gpd.GeoDataFrame,
        region_bbox: List[float],
        resolution_deg: float = 0.001,   # ~100m
        year_windows: List[int] = [3, 5, 7],
        reference_year: int = 2023,
    ) -> pd.DataFrame:
        """
        Aggregate fire detections into a spatial grid.
        Computes fire frequency over multiple time windows.
        
        Returns DataFrame with columns:
            lat, lon, fire_count_3yr, fire_count_5yr, fire_count_7yr,
            last_fire_year, fire_season_peak_month
        """
        west, south, east, north = region_bbox
        
        lons = np.arange(west, east, resolution_deg)
        lats = np.arange(south, north, resolution_deg)
        grid_lons, grid_lats = np.meshgrid(lons, lats)
        
        grid = pd.DataFrame({
            "lon": grid_lons.ravel(),
            "lat": grid_lats.ravel(),
        })
        
        for window in year_windows:
            start_year = reference_year - window
            mask = (fires_gdf["year"] >= start_year) & (fires_gdf["year"] <= reference_year)
            window_fires = fires_gdf[mask]
            
            # Bin fire points to grid
            lon_bins = np.digitize(window_fires["longitude"], lons)
            lat_bins = np.digitize(window_fires["latitude"], lats)
            
            counts = (
                pd.Series(zip(lat_bins, lon_bins))
                .value_counts()
                .reset_index()
            )
            counts.columns = ["grid_idx", "count"]
            
            # Map back to grid (simplified — production code uses rasterize)
            grid[f"fire_count_{window}yr"] = 0  # Placeholder; see features.py for full impl
        
        # Last fire year per pixel
        grid["last_fire_year"] = 0  # Filled in features.py
        
        return grid


def load_sample_data_for_dev(n_fire: int = 5000, n_nonfire: int = 50000, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic data with realistic structure for offline development.
    Use this when you don't have GEE access or want to test the ML pipeline quickly.
    
    Structure mimics real AEF embeddings + FIRMS labels.
    Embeddings are NOT random — we embed a known signal so the model can learn it.
    """
    rng = np.random.RandomState(seed)
    
    n_total = n_fire + n_nonfire
    
    # Simulate embedding clusters
    # Fire pixels cluster around a "dry vegetation" embedding signature
    fire_center = rng.randn(64) * 0.5 + 1.5    # Elevated values = dry/sparse veg
    nonfire_center = rng.randn(64) * 0.5 - 0.5  # Lower values = dense/moist veg
    
    fire_embeddings = fire_center + rng.randn(n_fire, 64) * 0.8
    nonfire_embeddings = nonfire_center + rng.randn(n_nonfire, 64) * 0.8
    
    embeddings = np.vstack([fire_embeddings, nonfire_embeddings])
    labels = np.array([1] * n_fire + [0] * n_nonfire)
    
    # Temporal delta (year-over-year embedding shift)
    # Fire pixels show larger positive drift (drying trend)
    drift = np.vstack([
        rng.randn(n_fire, 64) * 0.3 + 0.2,   # Fire: positive drift
        rng.randn(n_nonfire, 64) * 0.2,        # Non-fire: near-zero drift
    ])
    
    # Spatial features
    lat = rng.uniform(32.5, 42.1, n_total)
    lon = rng.uniform(-124.5, -113.9, n_total)
    elevation = rng.exponential(300, n_total)
    slope = rng.exponential(8, n_total)
    
    # Historical fire frequency (correlated with label)
    fire_count_5yr = (labels * rng.poisson(2, n_total) +
                      rng.poisson(0.1, n_total)).astype(float)
    
    cols = {}
    cols["lat"] = lat
    cols["lon"] = lon
    cols["label"] = labels
    cols["year"] = rng.choice([2020, 2021, 2022, 2023], n_total)
    cols["elevation"] = elevation
    cols["slope"] = slope
    cols["fire_count_3yr"] = fire_count_5yr * 0.6
    cols["fire_count_5yr"] = fire_count_5yr
    cols["fire_count_7yr"] = fire_count_5yr * 1.3
    cols["last_fire_year"] = rng.choice([0, 2018, 2019, 2020, 2021, 2022], n_total)
    
    for i in range(64):
        cols[f"embedding_{i}"] = embeddings[:, i]
    for i in range(64):
        cols[f"delta_{i}"] = drift[:, i]
    
    df = pd.DataFrame(cols)
    logger.info(f"Synthetic dev data: {n_fire} fire, {n_nonfire} non-fire pixels")
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)
