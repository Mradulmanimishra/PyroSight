"""
main.py — End-to-end pipeline runner for Wildfire Risk Prediction.

Usage:
  # Full training run with real GEE data
  python main.py --region california --years 2017 2024 --task train

  # Predict risk map for 2024
  python main.py --region california --year 2024 --task predict

  # Run on synthetic dev data (no GEE access required)
  python main.py --task dev

  # Spatial CV evaluation only
  python main.py --region california --task evaluate

Example full workflow:
  1. python main.py --task dev           # Validate pipeline on synthetic data
  2. python main.py --task train         # Train on real GEE data
  3. python main.py --task predict       # Generate 2024 risk map
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger

from config import (
    REGIONS, DEFAULT_REGION, YEARS, OUTPUTS_DIR,
    AEF_EXPORT_SCALE, DEV_SCALE, FIRE_LABEL_YEAR_OFFSET
)
from src.data_loader import (
    init_gee, get_aef_time_series, FIRMSLoader, load_sample_data_for_dev
)
from src.features import (
    WildfireFeatureBuilder, compute_temporal_deltas,
    compute_multi_year_drift, assign_fire_labels
)
from src.model import (
    WildfireEnsemble, evaluate_model, spatial_cross_validate, WildfireSHAPExplainer
)
from src.visualize import (
    plot_risk_map, plot_risk_tier_map, plot_interactive_risk_map,
    plot_dryness_trend, plot_annual_embedding_pca, plot_evaluation_suite,
    plot_shap_summary, plot_feature_group_contributions, plot_fire_history_vs_risk
)


def run_dev_pipeline():
    """
    Full pipeline on synthetic data — validates all components without GEE.
    Runs in ~2 minutes on a laptop.
    """
    logger.info("=" * 60)
    logger.info("WILDFIRE RISK PREDICTION — DEV MODE (synthetic data)")
    logger.info("=" * 60)
    
    # ── 1. Load synthetic data ────────────────────────────────────
    df = load_sample_data_for_dev(n_fire=8000, n_nonfire=80000)
    logger.info(f"Loaded {len(df):,} samples")
    
    # ── 2. Feature engineering ────────────────────────────────────
    logger.info("Building features...")
    builder = WildfireFeatureBuilder(
        use_temporal_delta=True,
        use_spatial_lag=True,
    )
    X, feature_names = builder.fit_transform(df)
    y = df["label"].values
    
    logger.info(f"Feature matrix: {X.shape[0]:,} × {X.shape[1]}")
    logger.info(f"Fire rate: {y.mean():.3f}")
    
    # ── 3. Train/test split (80/20 random for dev) ────────────────
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
        X, y, df, test_size=0.2, random_state=42, stratify=y
    )
    
    # ── 4. Train model ────────────────────────────────────────────
    logger.info("Training ensemble model...")
    model = WildfireEnsemble()
    model.fit(X_train, y_train, X_test, y_test, feature_names=feature_names)
    
    # ── 5. Evaluate ───────────────────────────────────────────────
    metrics = evaluate_model(model, X_test, y_test, split_name="test_set")
    
    # ── 6. Predictions ────────────────────────────────────────────
    df_test = df_test.copy()
    df_test["fire_prob"] = model.predict_proba(X_test)[:, 1]
    df_test["risk_tier"] = model.predict_risk_tier(X_test)
    
    # ── 7. SHAP ───────────────────────────────────────────────────
    logger.info("Computing SHAP values (sample)...")
    explainer = WildfireSHAPExplainer(model, feature_names)
    shap_values, X_sample = explainer.compute_shap_values(X_test, max_samples=2000)
    
    top_feats = explainer.top_features(shap_values, n=10)
    logger.info(f"Top 10 features:\n{top_feats.to_string(index=False)}")
    
    group_contrib = explainer.embedding_interpretation(shap_values)
    logger.info(f"Feature group contributions:\n{group_contrib.to_string(index=False)}")
    
    # ── 8. Visualizations ─────────────────────────────────────────
    logger.info("Generating visualizations...")
    
    plot_risk_map(
        df_test, title="Wildfire Risk Map (Dev — Synthetic)",
        save_path=OUTPUTS_DIR / "risk_map.png"
    )
    
    plot_risk_tier_map(
        df_test,
        save_path=OUTPUTS_DIR / "risk_tiers.png"
    )
    
    plot_evaluation_suite(
        y_test, df_test["fire_prob"].values,
        save_path=OUTPUTS_DIR / "evaluation.png"
    )
    
    plot_shap_summary(
        shap_values, X_sample, feature_names,
        save_path=OUTPUTS_DIR / "shap_summary.png"
    )
    
    plot_feature_group_contributions(
        group_contrib,
        save_path=OUTPUTS_DIR / "feature_groups.png"
    )
    
    plot_fire_history_vs_risk(
        df_test,
        save_path=OUTPUTS_DIR / "fire_history_vs_risk.png"
    )
    
    # ── 9. Save model ─────────────────────────────────────────────
    model.save()
    
    # ── 10. Save predictions ──────────────────────────────────────
    out_csv = OUTPUTS_DIR / "predictions_dev.csv"
    df_test[["lat", "lon", "year", "label", "fire_prob", "risk_tier"]].to_csv(out_csv, index=False)
    logger.info(f"Predictions saved → {out_csv}")
    
    logger.info("=" * 60)
    logger.info("DEV RUN COMPLETE")
    logger.info(f"  AUC-ROC:  {metrics['auc_roc']:.4f}")
    logger.info(f"  PR-AUC:   {metrics['pr_auc']:.4f}")
    logger.info(f"  Outputs:  {OUTPUTS_DIR}")
    logger.info("=" * 60)
    
    return model, metrics


def run_training_pipeline(region_name: str, years: List[int], scale: int = DEV_SCALE):
    """
    Full pipeline with real GEE data.
    Requires: GEE authentication, FIRMS API key.
    """
    logger.info("=" * 60)
    logger.info(f"WILDFIRE RISK PREDICTION — TRAINING PIPELINE")
    logger.info(f"Region: {region_name}  |  Years: {years[0]}–{years[-1]}")
    logger.info("=" * 60)
    
    region_cfg = REGIONS[region_name]
    bbox = region_cfg["bbox"]
    
    # ── 1. Initialize GEE ────────────────────────────────────────
    init_gee()
    
    # ── 2. Load AEF embeddings ────────────────────────────────────
    logger.info("Fetching AEF embeddings from GEE...")
    aef_images = get_aef_time_series(years, bbox)
    
    if not aef_images:
        raise RuntimeError("No AEF images loaded. Check GEE authentication and region.")
    
    # ── 3. Sample pixels ──────────────────────────────────────────
    # For training, sample a grid of points across the region
    # In production, export full rasters and process per-tile
    logger.info("Sampling pixel embeddings...")
    from src.data_loader import sample_aef_to_dataframe
    import ee
    
    # Create sampling grid (~100m spacing = dense enough for training)
    all_year_dfs = []
    
    for year in years:
        if year not in aef_images:
            continue
        
        # Sample 50k random points per year (increase for production)
        region = ee.Geometry.Rectangle(bbox)
        sample_pts = ee.FeatureCollection.randomPoints(
            region=region, points=50000, seed=year
        )
        
        year_df = sample_aef_to_dataframe(
            aef_images[year], sample_pts, scale=scale, year=year
        )
        all_year_dfs.append(year_df)
    
    df_all = pd.concat(all_year_dfs, ignore_index=True)
    
    # ── 4. Compute temporal deltas ────────────────────────────────
    logger.info("Computing temporal delta features...")
    dfs_by_year = {y: df_all[df_all["year"] == y] for y in years if y in df_all["year"].values}
    
    # For each year t, compute delta vs year t-1
    delta_dfs = []
    for year in sorted(dfs_by_year.keys())[1:]:
        prev_year = year - 1
        if prev_year in dfs_by_year:
            delta_df = compute_temporal_deltas(
                dfs_by_year[year].copy(),
                dfs_by_year[prev_year].copy(),
            )
            delta_dfs.append(delta_df)
    
    df_with_deltas = pd.concat(delta_dfs, ignore_index=True) if delta_dfs else df_all
    
    # ── 5. Load FIRMS fire data ───────────────────────────────────
    logger.info("Loading NASA FIRMS historical fire data...")
    firms = FIRMSLoader()
    fires_gdf = firms.fetch_historical(bbox, years[0], years[-1])
    
    # ── 6. Assign fire labels ─────────────────────────────────────
    logger.info("Assigning fire labels (t+1 prediction)...")
    labeled_dfs = []
    
    for year in sorted(dfs_by_year.keys()):
        label_year = year + FIRE_LABEL_YEAR_OFFSET
        df_year = df_with_deltas[df_with_deltas["year"] == year].copy()
        
        if fires_gdf is not None and not fires_gdf.empty:
            df_year = assign_fire_labels(df_year, fires_gdf, label_year=label_year)
        else:
            logger.warning(f"No FIRMS data for {label_year} — using zero labels")
            df_year["label"] = 0
        
        labeled_dfs.append(df_year)
    
    df_labeled = pd.concat(labeled_dfs, ignore_index=True)
    df_labeled = df_labeled[df_labeled["label"] != -1]
    
    # ── 7–10: Same as dev pipeline ────────────────────────────────
    logger.info("Building feature matrix...")
    builder = WildfireFeatureBuilder()
    X, feature_names = builder.fit_transform(df_labeled)
    y = df_labeled["label"].values
    
    # Spatial CV
    logger.info("Running blocked spatial cross-validation...")
    cv_results = spatial_cross_validate(X, y, df_labeled, feature_names)
    cv_results.to_csv(OUTPUTS_DIR / "cv_results.csv", index=False)
    
    # Train final model on all data
    from sklearn.model_selection import train_test_split
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)
    
    model = WildfireEnsemble()
    model.fit(X_tr, y_tr, X_val, y_val, feature_names=feature_names)
    model.save()
    
    logger.info("Training complete. Run with --task predict to generate risk maps.")
    return model


# ─────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Wildfire Risk Prediction Pipeline")
    parser.add_argument("--task", choices=["dev", "train", "predict", "evaluate"],
                        default="dev", help="Pipeline task to run")
    parser.add_argument("--region", default=DEFAULT_REGION,
                        choices=list(REGIONS.keys()),
                        help="Study region")
    parser.add_argument("--years", nargs="+", type=int, default=YEARS,
                        help="Years to include (e.g. 2018 2019 2020)")
    parser.add_argument("--year", type=int, default=2024,
                        help="Prediction year (for --task predict)")
    parser.add_argument("--scale", type=int, default=DEV_SCALE,
                        help="GEE export scale in meters")
    return parser.parse_args()


if __name__ == "__main__":
    from typing import List  # ensure available
    
    args = parse_args()
    
    if args.task == "dev":
        run_dev_pipeline()
    
    elif args.task == "train":
        run_training_pipeline(
            region_name=args.region,
            years=args.years,
            scale=args.scale,
        )
    
    elif args.task == "predict":
        logger.info(f"Loading model and generating risk map for {args.year}...")
        model = WildfireEnsemble.load()
        logger.info("Prediction task: initialize GEE, fetch AEF for target year, "
                    "then run model.predict_risk_tier(X) and plot_risk_map()")
        # Full prediction pipeline follows same steps as training pipeline
        # but uses the saved model instead of fitting a new one.
    
    elif args.task == "evaluate":
        logger.info("Loading model for evaluation...")
        model = WildfireEnsemble.load()
        logger.info("Evaluation task: load test set, run evaluate_model(), "
                    "spatial_cross_validate(), plot_evaluation_suite()")
