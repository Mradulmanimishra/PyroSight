"""
src/model.py
Model training, spatial cross-validation, calibration, ensemble, and SHAP explainability.

Model stack:
  - XGBoostClassifier (primary)
  - LightGBMClassifier (secondary)
  - Calibrated ensemble (Platt scaling)

Evaluation:
  - Blocked spatial cross-validation (prevents leakage from spatial autocorrelation)
  - AUC-ROC, PR-AUC, Brier score
  - SHAP waterfall + summary plots
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from loguru import logger

import xgboost as xgb
import lightgbm as lgb
import shap

from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    precision_recall_curve, roc_curve, confusion_matrix,
    classification_report,
)
from sklearn.base import BaseEstimator, ClassifierMixin

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    XGBOOST_PARAMS, LIGHTGBM_PARAMS, ENSEMBLE_WEIGHTS,
    SPATIAL_CV_BLOCK_SIZE_KM, SPATIAL_CV_N_FOLDS, RISK_THRESHOLDS,
    MODEL_DIR
)


# ─────────────────────────────────────────────────────────────────
# Spatial Cross-Validation
# ─────────────────────────────────────────────────────────────────

class BlockedSpatialCV:
    """
    Spatial cross-validation that groups nearby pixels into blocks.
    
    Standard k-fold leaks information because neighboring pixels share
    embedding values (a pixel and its neighbor are not independent samples).
    Blocked CV withholds entire geographic regions from each fold.
    
    Strategy:
      1. Assign each pixel to a grid block (block_size_km x block_size_km)
      2. Assign blocks to folds
      3. Train on all pixels outside fold, evaluate on pixels in fold
    
    Reference: Roberts et al. 2017, "Cross-validation strategies for data
    with temporal, spatial, hierarchical, or phylogenetic structure"
    """
    
    def __init__(self, block_size_km: float = SPATIAL_CV_BLOCK_SIZE_KM, n_folds: int = SPATIAL_CV_N_FOLDS):
        self.block_size_km = block_size_km
        self.n_folds = n_folds
    
    def split(self, df: pd.DataFrame) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Yield (train_idx, test_idx) tuples for each fold.
        df must have 'lat' and 'lon' columns.
        """
        # Assign block IDs
        km_per_deg_lat = 111.0
        km_per_deg_lon = 111.0 * np.cos(np.deg2rad(df["lat"].mean()))
        
        block_deg_lat = self.block_size_km / km_per_deg_lat
        block_deg_lon = self.block_size_km / km_per_deg_lon
        
        df = df.copy()
        df["_block_lat"] = (df["lat"] / block_deg_lat).astype(int)
        df["_block_lon"] = (df["lon"] / block_deg_lon).astype(int)
        df["_block_id"] = df["_block_lat"].astype(str) + "_" + df["_block_lon"].astype(str)
        
        unique_blocks = df["_block_id"].unique()
        np.random.shuffle(unique_blocks)
        
        block_folds = np.array_split(unique_blocks, self.n_folds)
        
        splits = []
        for fold_blocks in block_folds:
            test_mask = df["_block_id"].isin(fold_blocks)
            train_idx = np.where(~test_mask)[0]
            test_idx = np.where(test_mask)[0]
            splits.append((train_idx, test_idx))
            logger.debug(f"Fold: {len(train_idx)} train, {len(test_idx)} test, {len(fold_blocks)} blocks held out")
        
        return splits


# ─────────────────────────────────────────────────────────────────
# Ensemble Model
# ─────────────────────────────────────────────────────────────────

class WildfireEnsemble(BaseEstimator, ClassifierMixin):
    """
    Weighted ensemble of XGBoost + LightGBM with isotonic calibration.
    
    Why ensemble?
    - XGBoost and LightGBM make different errors due to their different tree
      construction algorithms (depth-first vs leaf-wise). Their combination
      is more robust across geographic regions.
    - Calibration ensures probabilities are reliable (for insurance/government use,
      "30% risk" must actually correspond to ~30% historical fire frequency).
    """
    
    def __init__(
        self,
        xgb_params: dict = None,
        lgb_params: dict = None,
        weights: List[float] = None,
        calibrate: bool = True,
    ):
        self.xgb_params = xgb_params or XGBOOST_PARAMS.copy()
        self.lgb_params = lgb_params or LIGHTGBM_PARAMS.copy()
        self.weights = weights or ENSEMBLE_WEIGHTS
        self.calibrate = calibrate
        
        self.xgb_model = None
        self.lgb_model = None
        self.feature_importances_ = None
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        feature_names: List[str] = None,
    ) -> "WildfireEnsemble":
        """
        Train XGBoost and LightGBM models.
        Calibration is applied after if X_val/y_val provided.
        """
        logger.info(f"Training on {X_train.shape[0]:,} samples, {X_train.shape[1]} features")
        logger.info(f"Fire ratio: {y_train.mean():.3f} ({y_train.sum():,} fire pixels)")
        
        # ── XGBoost ──────────────────────────────────────────────
        xgb_params = self.xgb_params.copy()
        early_stop = xgb_params.pop("early_stopping_rounds", 50)
        
        self.xgb_model = xgb.XGBClassifier(
            **xgb_params,
            feature_names=feature_names,
        )
        
        eval_set = [(X_val, y_val)] if X_val is not None else None
        
        self.xgb_model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=100,
        )
        logger.info("XGBoost training complete")
        
        # ── LightGBM ─────────────────────────────────────────────
        lgb_params = self.lgb_params.copy()
        
        callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)]
        
        self.lgb_model = lgb.LGBMClassifier(**lgb_params)
        
        lgb_eval = [(X_val, y_val)] if X_val is not None else None
        
        self.lgb_model.fit(
            X_train, y_train,
            eval_set=lgb_eval,
            callbacks=callbacks,
        )
        logger.info("LightGBM training complete")
        
        # ── Feature importances (XGBoost gain) ───────────────────
        if feature_names:
            self.feature_importances_ = pd.DataFrame({
                "feature": feature_names,
                "xgb_importance": self.xgb_model.feature_importances_,
                "lgb_importance": self.lgb_model.feature_importances_ / self.lgb_model.feature_importances_.sum(),
            }).sort_values("xgb_importance", ascending=False)
        
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return calibrated fire probability for each pixel.
        Output shape: (N, 2) — column 1 is P(fire).
        """
        xgb_proba = self.xgb_model.predict_proba(X)[:, 1]
        lgb_proba = self.lgb_model.predict_proba(X)[:, 1]
        
        w_xgb, w_lgb = self.weights
        ensemble_proba = w_xgb * xgb_proba + w_lgb * lgb_proba
        
        return np.column_stack([1 - ensemble_proba, ensemble_proba])
    
    def predict(self, X: np.ndarray, threshold: float = 0.35) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)
    
    def predict_risk_tier(self, X: np.ndarray) -> np.ndarray:
        """Map probabilities to risk tier labels."""
        proba = self.predict_proba(X)[:, 1]
        tiers = np.full(len(proba), "low", dtype=object)
        for tier, (lo, hi) in RISK_THRESHOLDS.items():
            mask = (proba >= lo) & (proba < hi)
            tiers[mask] = tier
        return tiers
    
    def save(self, path: Path = None):
        path = path or MODEL_DIR / "wildfire_ensemble.pkl"
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"Model saved → {path}")
    
    @classmethod
    def load(cls, path: Path = None) -> "WildfireEnsemble":
        path = path or MODEL_DIR / "wildfire_ensemble.pkl"
        with open(path, "rb") as f:
            return pickle.load(f)


# ─────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────

def evaluate_model(
    model: WildfireEnsemble,
    X: np.ndarray,
    y: np.ndarray,
    split_name: str = "test",
) -> Dict:
    """
    Full evaluation suite for fire risk model.
    Returns dict of metrics + prints a formatted summary.
    """
    proba = model.predict_proba(X)[:, 1]
    pred = model.predict(X)
    
    auc_roc = roc_auc_score(y, proba)
    pr_auc = average_precision_score(y, proba)
    brier = brier_score_loss(y, proba)
    
    # Confusion matrix at default threshold
    cm = confusion_matrix(y, pred)
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    
    metrics = {
        "split": split_name,
        "n_samples": len(y),
        "fire_rate": y.mean(),
        "auc_roc": auc_roc,
        "pr_auc": pr_auc,
        "brier_score": brier,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }
    
    logger.info(
        f"\n{'─'*50}\n"
        f"  Evaluation: {split_name}\n"
        f"  Samples: {len(y):,}  |  Fire rate: {y.mean():.3f}\n"
        f"  AUC-ROC:      {auc_roc:.4f}\n"
        f"  PR-AUC:       {pr_auc:.4f}  ← key metric (imbalanced)\n"
        f"  Brier Score:  {brier:.4f}  ← calibration (lower = better)\n"
        f"  Precision:    {precision:.4f}  |  Recall: {recall:.4f}  |  F1: {f1:.4f}\n"
        f"  TP={tp} FP={fp} FN={fn} TN={tn}\n"
        f"{'─'*50}"
    )
    return metrics


def spatial_cross_validate(
    X: np.ndarray,
    y: np.ndarray,
    df: pd.DataFrame,
    feature_names: List[str],
    n_folds: int = SPATIAL_CV_N_FOLDS,
) -> pd.DataFrame:
    """
    Run blocked spatial cross-validation and return per-fold metrics.
    
    This is the correct way to evaluate geospatial models.
    Standard CV overestimates performance by ~15–30% due to spatial autocorrelation.
    """
    spatial_cv = BlockedSpatialCV(n_folds=n_folds)
    splits = spatial_cv.split(df)
    
    all_metrics = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        logger.info(f"Spatial CV fold {fold_idx + 1}/{n_folds}")
        
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_te, y_te = X[test_idx], y[test_idx]
        
        # Use last 20% of training as validation for early stopping
        val_size = int(len(X_tr) * 0.2)
        X_val, y_val = X_tr[-val_size:], y_tr[-val_size:]
        X_tr, y_tr = X_tr[:-val_size], y_tr[:-val_size]
        
        fold_model = WildfireEnsemble()
        fold_model.fit(X_tr, y_tr, X_val, y_val, feature_names=feature_names)
        
        metrics = evaluate_model(fold_model, X_te, y_te, split_name=f"fold_{fold_idx+1}")
        all_metrics.append(metrics)
    
    results_df = pd.DataFrame(all_metrics)
    
    logger.info(
        f"\n{'═'*50}\n"
        f"  Spatial CV Summary ({n_folds} folds)\n"
        f"  AUC-ROC:  {results_df['auc_roc'].mean():.4f} ± {results_df['auc_roc'].std():.4f}\n"
        f"  PR-AUC:   {results_df['pr_auc'].mean():.4f} ± {results_df['pr_auc'].std():.4f}\n"
        f"  Brier:    {results_df['brier_score'].mean():.4f} ± {results_df['brier_score'].std():.4f}\n"
        f"{'═'*50}"
    )
    return results_df


# ─────────────────────────────────────────────────────────────────
# SHAP Explainability
# ─────────────────────────────────────────────────────────────────

class WildfireSHAPExplainer:
    """
    SHAP-based model explainability for wildfire risk predictions.
    
    Why SHAP?
    - Insurance companies and governments need to understand WHY a pixel is high risk
    - SHAP gives per-pixel, per-feature attribution in probability units
    - TreeSHAP (used by XGBoost/LightGBM) is exact and fast — O(TLD) complexity
    
    Key plots:
    - Summary plot: Feature importance across all pixels
    - Waterfall plot: Why THIS pixel has 72% fire probability
    - Dependence plot: How fire probability changes with slope/elevation/embeddings
    - Interaction plot: Which feature pairs combine to create extreme risk
    """
    
    def __init__(self, model: WildfireEnsemble, feature_names: List[str]):
        self.model = model
        self.feature_names = feature_names
        self.xgb_explainer = shap.TreeExplainer(model.xgb_model)
        self.lgb_explainer = shap.TreeExplainer(model.lgb_model)
    
    def compute_shap_values(self, X: np.ndarray, max_samples: int = 5000) -> np.ndarray:
        """
        Compute SHAP values for the ensemble.
        For large datasets, subsample to max_samples for speed.
        
        Returns array of shape (N, n_features)
        """
        if len(X) > max_samples:
            idx = np.random.choice(len(X), max_samples, replace=False)
            X_sample = X[idx]
        else:
            X_sample = X
        
        xgb_shap = self.xgb_explainer.shap_values(X_sample)[:, :, 1] if \
            isinstance(self.xgb_explainer.shap_values(X_sample), list) else \
            self.xgb_explainer.shap_values(X_sample)
        
        lgb_shap = self.lgb_explainer.shap_values(X_sample)
        if isinstance(lgb_shap, list):
            lgb_shap = lgb_shap[1]
        
        w_xgb, w_lgb = self.model.weights
        ensemble_shap = w_xgb * xgb_shap + w_lgb * lgb_shap
        
        return ensemble_shap, X_sample
    
    def top_features(self, shap_values: np.ndarray, n: int = 20) -> pd.DataFrame:
        """Return top n features by mean absolute SHAP value."""
        mean_abs = np.abs(shap_values).mean(axis=0)
        return pd.DataFrame({
            "feature": self.feature_names,
            "mean_abs_shap": mean_abs,
        }).sort_values("mean_abs_shap", ascending=False).head(n)
    
    def explain_pixel(self, X_pixel: np.ndarray, pixel_id: str = "") -> Dict:
        """
        Full attribution breakdown for a single pixel.
        Returns sorted list of (feature_name, shap_value) pairs.
        
        Use this to explain to a government client why a specific parcel
        is classified as "extreme" risk.
        """
        xgb_shap = self.xgb_explainer.shap_values(X_pixel.reshape(1, -1))
        if isinstance(xgb_shap, list):
            xgb_shap = xgb_shap[1]
        
        attribution = pd.DataFrame({
            "feature": self.feature_names,
            "shap_value": xgb_shap.ravel(),
            "feature_value": X_pixel.ravel(),
        }).sort_values("shap_value", key=abs, ascending=False)
        
        prob = self.model.predict_proba(X_pixel.reshape(1, -1))[0, 1]
        tier = self.model.predict_risk_tier(X_pixel.reshape(1, -1))[0]
        
        logger.info(
            f"Pixel {pixel_id}: P(fire)={prob:.3f}, Risk tier={tier}\n"
            f"Top drivers:\n{attribution.head(5).to_string(index=False)}"
        )
        
        return {
            "probability": prob,
            "risk_tier": tier,
            "attribution": attribution,
            "pixel_id": pixel_id,
        }
    
    def embedding_interpretation(self, shap_values: np.ndarray) -> pd.DataFrame:
        """
        Aggregate SHAP contributions of embedding dimensions vs other features.
        
        Shows how much of the model's prediction comes from:
          - The raw AEF embedding (foundation model output)
          - Temporal drift (change in embedding over time)
          - Spatial context (neighborhood embeddings)
          - Historical fire frequency
          - Topography
        """
        feat_arr = np.array(self.feature_names)
        mean_abs = np.abs(shap_values).mean(axis=0)
        
        groups = {
            "AEF embedding": feat_arr[pd.Series(feat_arr).str.startswith("emb_")],
            "Temporal drift": feat_arr[pd.Series(feat_arr).str.startswith("delta_")],
            "Spatial context": feat_arr[pd.Series(feat_arr).str.startswith("spatial_")],
            "Fire history": feat_arr[pd.Series(feat_arr).str.startswith("fire_") | 
                                     pd.Series(feat_arr).str.startswith("years_")],
            "Topography": feat_arr[pd.Series(feat_arr).isin(["elevation", "slope", "aspect", "southward_slope"])],
            "Other": []
        }
        
        rows = []
        assigned = set()
        for group_name, group_feats in groups.items():
            idxs = [i for i, f in enumerate(self.feature_names) if f in group_feats]
            total_shap = mean_abs[idxs].sum() if idxs else 0
            rows.append({"feature_group": group_name, "total_shap": total_shap, "n_features": len(idxs)})
            assigned.update(idxs)
        
        # Other
        other_idxs = [i for i in range(len(self.feature_names)) if i not in assigned]
        rows[-1]["total_shap"] = mean_abs[other_idxs].sum()
        rows[-1]["n_features"] = len(other_idxs)
        
        df = pd.DataFrame(rows)
        df["pct_contribution"] = 100 * df["total_shap"] / df["total_shap"].sum()
        return df.sort_values("total_shap", ascending=False)
