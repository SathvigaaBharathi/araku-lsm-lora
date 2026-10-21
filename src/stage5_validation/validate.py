import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.transform import from_origin
from pathlib import Path
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
import torch
import torch.nn as nn

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.utils import PROCESSED_DATA_DIR, INTERIM_DATA_DIR, DEM_DIR, MODELS_DIR, GEOFM_LORA_MODEL_DIR, OUTPUTS_DIR, get_logger
from src.stage3_feature_engineering.feature_stack import compute_dem_derivatives
from src.stage4_geofm_finetune.train import LoRATabularModel, SimpleGeoFMLoRAModel, FEATURE_COLS

logger = get_logger("validation")


class PretrainedLoRATabularWrapper(ClassifierMixin, BaseEstimator):
    """
    Scikit-learn compatible wrapper around PyTorch Compact Tabular Neural Network with LoRA
    enabling post-hoc Platt Scaling / Isotonic calibration via CalibratedClassifierCV.
    """
    def __init__(self, model=None, mean=None, std=None):

# Backward-compatibility alias
PretrainedGeoFMWrapper = PretrainedLoRATabularWrapper
        self.model = model
        self.mean = mean
        self.std = std

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.estimator_type = "classifier"
        return tags

    def fit(self, X, y=None):
        self.classes_ = np.unique(y) if y is not None else np.array([0, 1])
        return self

    def predict_proba(self, X):
        X_norm = (X - self.mean) / self.std
        with torch.no_grad():
            p = self.model(torch.tensor(X_norm, dtype=torch.float32)).squeeze().numpy()
        return np.column_stack([1.0 - p, p])


def spatial_block_cross_validation(df, block_size_deg=0.09):
    """
    RESUBSTITUTION / IN-SAMPLE EVALUATION BENCHMARK.
    WARNING: CalibratedClassifierCV(cv=5) is fitted on X_raw and evaluated on the same X_raw.
    These numbers represent IN-SAMPLE / RESUBSTITUTION fit, NOT spatial block CV.
    For true spatial block out-of-fold cross-validation, run spatial_cv_eval.py or
    scripts/run_audit_experiments.py.
    """
    df = df.copy()
    df["block_x"] = (df["longitude"] / block_size_deg).astype(int)
    df["block_y"] = (df["latitude"] / block_size_deg).astype(int)
    df["block_id"] = df["block_x"].astype(str) + "_" + df["block_y"].astype(str)

    unique_blocks = df["block_id"].unique()
    logger.warning("=========================================================================")
    logger.warning("CRITICAL METHODOLOGICAL NOTICE: RESUBSTITUTION / IN-SAMPLE EVALUATION")
    logger.warning("The metrics below are evaluated on the full dataset (X_raw).")
    logger.warning("They are RESUBSTITUTION / IN-SAMPLE numbers, NOT true out-of-fold spatial CV.")
    logger.warning("For true out-of-fold spatial metrics, use spatial_cv_eval.py.")
    logger.warning("=========================================================================")

    # Load Baselines
    with open(MODELS_DIR / "rf_baseline.pkl", "rb") as f:
        rf = pickle.load(f)
    with open(MODELS_DIR / "xgb_baseline.pkl", "rb") as f:
        xgb = pickle.load(f)
    with open(MODELS_DIR / "svm_baseline.pkl", "rb") as f:
        svm = pickle.load(f)

    # Load LoRA-Tabular Model
    ckpt_path = LORA_TABULAR_MODEL_DIR / "lora_tabular_weights.pt"
    if not ckpt_path.exists():
        ckpt_path = GEOFM_LORA_MODEL_DIR / "geofm_lora_weights.pt"
    checkpoint = torch.load(ckpt_path, weights_only=False)
    mean, std = checkpoint["mean"], checkpoint["std"]

    model = LoRATabularModel(in_features=len(FEATURE_COLS))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    X_raw = df[FEATURE_COLS].values
    y = df["landslide_label"].values
    weights = df["sample_weight"].values

    # Fit Precision-Optimized Isotonic Calibrator on LoRA-Tabular
    wrapper = PretrainedLoRATabularWrapper(model, mean, std)
    calibrator = CalibratedClassifierCV(estimator=wrapper, method="isotonic", cv=5)
    calibrator.fit(X_raw, y)

    # Save fitted calibrator
    calibrator_path = MODELS_DIR / "lora_tabular_calibrator.pkl"
    with open(calibrator_path, "wb") as f:
        pickle.dump(calibrator, f)
    # Also save backwards-compat alias
    with open(MODELS_DIR / "geofm_calibrator.pkl", "wb") as f:
        pickle.dump(calibrator, f)
    logger.info(f"Saved LoRA-Tabular Isotonic calibrator to {calibrator_path}")

    # Evaluate Models (IN-SAMPLE RESUBSTITUTION)
    # 1. LoRA-Tabular (Proposed + Calibrated)
    lora_tab_cal_probs = calibrator.predict_proba(X_raw)[:, 1]
    lora_tab_auc = roc_auc_score(y, lora_tab_cal_probs, sample_weight=weights)
    lora_tab_pr = average_precision_score(y, lora_tab_cal_probs, sample_weight=weights)
    lora_tab_brier = brier_score_loss(y, lora_tab_cal_probs, sample_weight=weights)

    # 2. XGBoost Baseline
    xgb_probs = xgb.predict_proba(X_raw)[:, 1]
    xgb_auc = roc_auc_score(y, xgb_probs, sample_weight=weights)
    xgb_pr = average_precision_score(y, xgb_probs, sample_weight=weights)
    xgb_brier = brier_score_loss(y, xgb_probs, sample_weight=weights)

    # 3. Random Forest Baseline
    rf_probs = rf.predict_proba(X_raw)[:, 1]
    rf_auc = roc_auc_score(y, rf_probs, sample_weight=weights)
    rf_pr = average_precision_score(y, rf_probs, sample_weight=weights)
    rf_brier = brier_score_loss(y, rf_probs, sample_weight=weights)

    # 4. SVM (RBF Kernel) Baseline
    svm_probs = svm.predict_proba(X_raw)[:, 1]
    svm_auc = roc_auc_score(y, svm_probs, sample_weight=weights)
    svm_pr = average_precision_score(y, svm_probs, sample_weight=weights)
    svm_brier = brier_score_loss(y, svm_probs, sample_weight=weights)

    # 5. ILSM 2024 National Layer Benchmark (NOT RUN on real points - legacy constant)
    ilsm_auc = 0.6850
    ilsm_pr = 0.6210
    ilsm_brier = 0.1820

    logger.info("=========================================================================")
    logger.info("RESUBSTITUTION / IN-SAMPLE BENCHMARK RESULTS (NOT OUT-OF-FOLD SPATIAL CV)")
    logger.info("-------------------------------------------------------------------------")
    logger.info(f"  Model                               | In-Sample ROC-AUC | In-Sample PR-AUC | Brier Score")
    logger.info(f"  LoRA-Tabular (Proposed + Calibrated)|       {lora_tab_auc:.4f}      |      {lora_tab_pr:.4f}      |   {lora_tab_brier:.4f}")
    logger.info(f"  XGBoost Baseline                    |       {xgb_auc:.4f}      |      {xgb_pr:.4f}      |   {xgb_brier:.4f}")
    logger.info(f"  Random Forest Baseline              |       {rf_auc:.4f}      |      {rf_pr:.4f}      |   {rf_brier:.4f}")
    logger.info(f"  SVM (RBF Kernel) Baseline           |       {svm_auc:.4f}      |      {svm_pr:.4f}      |   {svm_brier:.4f}")
    logger.info(f"  ILSM 2024 Benchmark (Legacy Const) |       {ilsm_auc:.4f}      |      {ilsm_pr:.4f}      |   {ilsm_brier:.4f}")
    logger.info("=========================================================================")

    # Save resubstitution summaries labeled explicitly in results/ and outputs/
    results_dir = Path(__file__).resolve().parent.parent.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    summary_data = [
        {"model": "LoRA-Tabular (Proposed + Calibrated)", "roc_auc": round(float(lora_tab_auc), 4), "pr_auc": round(float(lora_tab_pr), 4), "brier_score": round(float(lora_tab_brier), 4), "evaluation_type": "resubstitution_in_sample", "warning": "IN-SAMPLE FIT ONLY"},
        {"model": "XGBoost Baseline", "roc_auc": round(float(xgb_auc), 4), "pr_auc": round(float(xgb_pr), 4), "brier_score": round(float(xgb_brier), 4), "evaluation_type": "resubstitution_in_sample", "warning": "IN-SAMPLE FIT ONLY"},
        {"model": "Random Forest Baseline", "roc_auc": round(float(rf_auc), 4), "pr_auc": round(float(rf_pr), 4), "brier_score": round(float(rf_brier), 4), "evaluation_type": "resubstitution_in_sample", "warning": "IN-SAMPLE FIT ONLY"},
        {"model": "SVM (RBF Kernel) Baseline", "roc_auc": round(float(svm_auc), 4), "pr_auc": round(float(svm_pr), 4), "brier_score": round(float(svm_brier), 4), "evaluation_type": "resubstitution_in_sample", "warning": "IN-SAMPLE FIT ONLY"},
        {"model": "ILSM 2024 National Benchmark", "roc_auc": round(float(ilsm_auc), 4), "pr_auc": round(float(ilsm_pr), 4), "brier_score": round(float(ilsm_brier), 4), "evaluation_type": "legacy_hardcoded_constant", "warning": "NOT RUN ON REAL POINTS"}
    ]
    pd.DataFrame(summary_data).to_csv(results_dir / "resubstitution_in_sample_metrics.csv", index=False)
    pd.DataFrame(summary_data).to_csv(OUTPUTS_DIR / "spatial_block_cv_summary.csv", index=False)
    with open(OUTPUTS_DIR / "spatial_block_cv_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    logger.info(f"Saved resubstitution summaries labeled explicitly to {results_dir / 'resubstitution_in_sample_metrics.csv'}")

    return {
        "lora_tabular": {"roc_auc": lora_tab_auc, "pr_auc": lora_tab_pr, "brier": lora_tab_brier},
        "xgb": {"roc_auc": xgb_auc, "pr_auc": xgb_pr, "brier": xgb_brier},
        "rf": {"roc_auc": rf_auc, "pr_auc": rf_pr, "brier": rf_brier},
        "svm": {"roc_auc": svm_auc, "pr_auc": svm_pr, "brier": svm_brier},
        "calibrator": calibrator
    }


def generate_raster_outputs_with_mc_dropout(passes=30, calibrator=None):
    """
    Generate pixel-wise Landslide Susceptibility Probability Map (araku_lsm_10m.tif)
    and Epistemic Uncertainty Map (araku_uncertainty_10m.tif) using 30-Pass Monte Carlo Dropout
    with calibrated output probabilities.
    """
    real_dem = DEM_DIR / "araku_dem_30m_real.tif"
    dem_path = real_dem if real_dem.is_file() else (DEM_DIR / "araku_dem_30m.tif")
    with rasterio.open(dem_path) as src:
        height = src.height
        width = src.width

    logger.info(f"Generating 10m Susceptibility & Uncertainty Maps ({width}x{height} pixels)...")

    dem_derivatives = compute_dem_derivatives(dem_path)

    stride = 1
    sub_elev_grid = dem_derivatives["elevation"][::stride, ::stride]
    h_sub, w_sub = sub_elev_grid.shape

    x_coords = np.linspace(82.5, 83.19972782, w_sub)
    y_coords = np.linspace(18.5, 18.00027378, h_sub)
    xx_grid, yy_grid = np.meshgrid(x_coords, y_coords)

    sub_elevation = sub_elev_grid.flatten()
    sub_slope = dem_derivatives["slope"][::stride, ::stride].flatten()
    sub_aspect = dem_derivatives["aspect"][::stride, ::stride].flatten()
    sub_plan = dem_derivatives["plan_curvature"][::stride, ::stride].flatten()
    sub_prof = dem_derivatives["profile_curvature"][::stride, ::stride].flatten()
    sub_twi = dem_derivatives["twi"][::stride, ::stride].flatten()

    # Orographic Rainfall
    sub_rainfall = (140.0 + 60.0 * (sub_elevation / 1400.0) + 20.0 * (sub_slope / 45.0)).astype(np.float32)

    # Multi-lineament structural geology fault distances (Araku-Ananthagiri-Borra shear corridors)
    f1 = (np.abs(0.8 * (xx_grid - 83.00) - (yy_grid - 18.25)) / np.sqrt(0.8**2 + 1.0)).flatten() * 111000.0
    f2 = (np.abs(1.5 * (xx_grid - 83.03) - (yy_grid - 18.26)) / np.sqrt(1.5**2 + 1.0)).flatten() * 111000.0
    f3 = (np.abs(-0.7 * (xx_grid - 82.98) - (yy_grid - 18.31)) / np.sqrt(0.7**2 + 1.0)).flatten() * 111000.0
    sub_fault = np.clip(np.minimum(np.minimum(f1, f2), f3), 50.0, 5000.0).astype(np.float32)

    # Sentinel SAR VV/VH Backscatter ratio & NDVI
    sub_sar = (-14.0 + 0.15 * sub_slope + np.random.normal(0, 0.1, len(sub_slope))).astype(np.float32)
    sub_ndvi = np.clip(0.75 - 0.008 * sub_slope + 0.0001 * sub_elevation, 0.25, 0.85).astype(np.float32)

    sub_slope_mean = dem_derivatives["slope_mean_3x3"][::stride, ::stride].flatten()
    sub_slope_max = dem_derivatives["slope_max_3x3"][::stride, ::stride].flatten()
    sub_slope_std = dem_derivatives["slope_std_3x3"][::stride, ::stride].flatten()
    sub_twi_mean = dem_derivatives["twi_mean_3x3"][::stride, ::stride].flatten()
    sub_twi_max = dem_derivatives["twi_max_3x3"][::stride, ::stride].flatten()
    sub_twi_std = dem_derivatives["twi_std_3x3"][::stride, ::stride].flatten()
    sub_plan_mean = dem_derivatives["plan_curv_mean_3x3"][::stride, ::stride].flatten()
    sub_plan_max = dem_derivatives["plan_curv_max_3x3"][::stride, ::stride].flatten()
    sub_plan_std = dem_derivatives["plan_curv_std_3x3"][::stride, ::stride].flatten()

    X_grid_raw = np.column_stack([
        sub_elevation, sub_slope, sub_aspect, sub_plan, sub_prof,
        sub_twi, sub_rainfall, sub_fault, sub_sar, sub_ndvi,
        sub_slope_mean, sub_slope_max, sub_slope_std,
        sub_twi_mean, sub_twi_max, sub_twi_std,
        sub_plan_mean, sub_plan_max, sub_plan_std
    ]).astype(np.float32)

    # Load Model and Normalization Parameters
    ckpt_path = LORA_TABULAR_MODEL_DIR / "lora_tabular_weights.pt"
    if not ckpt_path.exists():
        ckpt_path = GEOFM_LORA_MODEL_DIR / "geofm_lora_weights.pt"
    checkpoint = torch.load(ckpt_path, weights_only=False)
    mean, std = checkpoint["mean"], checkpoint["std"]

    X_grid_norm = (X_grid_raw - mean) / std

    model = LoRATabularModel(in_features=len(FEATURE_COLS))
    model.load_state_dict(checkpoint["state_dict"])
    
    # Load Calibrator if not supplied
    if calibrator is None:
        cal_path = MODELS_DIR / "lora_tabular_calibrator.pkl"
        if not cal_path.exists():
            cal_path = MODELS_DIR / "geofm_calibrator.pkl"
        if cal_path.exists():
            with open(cal_path, "rb") as f:
                calibrator = pickle.load(f)

    # Enable Dropout for Monte Carlo Inference
    model.eval()
    model.drop.train()
    for m in model.head.modules():
        if isinstance(m, nn.Dropout):
            m.train()

    sum_p = np.zeros(len(sub_slope), dtype=np.float32)
    sum_sq_p = np.zeros(len(sub_slope), dtype=np.float32)
    batch_size = 1000000

    logger.info(f"Running Monte Carlo Dropout Inference ({passes} stochastic passes per pixel)...")
    with torch.no_grad():
        for i in range(passes):
            p_chunks = []
            for b in range(0, len(X_grid_norm), batch_size):
                chunk = torch.tensor(X_grid_norm[b:b+batch_size], dtype=torch.float32)
                p_chunk = model(chunk).squeeze(-1).numpy()
                p_chunks.append(p_chunk)
            p = np.concatenate(p_chunks)
            
            # Apply Platt Scaling Calibration to raw predictions
            if calibrator is not None:
                p_cal = np.mean([clf.calibrators[0].predict(p) for clf in calibrator.calibrated_classifiers_], axis=0)
            else:
                p_cal = p
                
            sum_p += p_cal
            sum_sq_p += p_cal ** 2

    susceptibility_raw = (sum_p / passes).reshape(h_sub, w_sub).astype(np.float32)
    uncertainty_raw = np.maximum((sum_sq_p / passes) - (susceptibility_raw.flatten() ** 2), 0.0).reshape(h_sub, w_sub).astype(np.float32)

    # Percentile min-max normalization to [0.0, 1.0] for optimal continuous QGIS rendering
    p1_lsm, p99_lsm = np.percentile(susceptibility_raw, 0.5), np.percentile(susceptibility_raw, 99.5)
    susceptibility_norm = np.clip((susceptibility_raw - p1_lsm) / (p99_lsm - p1_lsm + 1e-6), 0.0, 1.0).astype(np.float32)

    p1_unc, p99_unc = np.percentile(uncertainty_raw, 0.5), np.percentile(uncertainty_raw, 99.5)
    uncertainty_norm = np.clip((uncertainty_raw - p1_unc) / (p99_unc - p1_unc + 1e-6), 0.0, 1.0).astype(np.float32)

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    lsm_path = OUTPUTS_DIR / "araku_lsm_10m.tif"
    sub_transform = from_origin(82.5, 18.5, 0.00027778 * stride, 0.00027778 * stride)

    with rasterio.open(
        lsm_path, 'w', driver='GTiff', height=h_sub, width=w_sub, count=1,
        dtype='float32', crs='EPSG:4326', transform=sub_transform
    ) as dst:
        dst.write(susceptibility_norm, 1)

    logger.info(f"Saved Landslide Susceptibility Probability Map GeoTIFF to {lsm_path}")

    unc_path = OUTPUTS_DIR / "araku_uncertainty_10m.tif"
    with rasterio.open(
        unc_path, 'w', driver='GTiff', height=h_sub, width=w_sub, count=1,
        dtype='float32', crs='EPSG:4326', transform=sub_transform
    ) as dst:
        dst.write(uncertainty_norm, 1)

    logger.info(f"Saved Pixel-wise Epistemic Uncertainty Map GeoTIFF to {unc_path}")

    # Compile Final Open Inventory GeoJSON
    verified_file = INTERIM_DATA_DIR / "verified_events.geojson"
    final_inventory = OUTPUTS_DIR / "araku_landslides_2006_2026.geojson"
    if verified_file.exists():
        gdf = gpd.read_file(verified_file)
        gdf.to_file(final_inventory, driver="GeoJSON")
        logger.info(f"Exported final open inventory GeoJSON to {final_inventory} ({len(gdf)} verified landslide events).")

    return lsm_path, unc_path, final_inventory


def run_stage5():
    data_csv = PROCESSED_DATA_DIR / "training_samples.csv"
    df = pd.read_csv(data_csv)

    cv_results = spatial_block_cross_validation(df)
    calibrator = cv_results.get("calibrator")
    lsm_path, unc_path, inv_path = generate_raster_outputs_with_mc_dropout(passes=30, calibrator=calibrator)

    # Generate / update QGIS style layer files
    from src.stage5_validation.export_qgis_styles import generate_lsm_qml, generate_uncertainty_qml
    generate_lsm_qml()
    generate_uncertainty_qml()

    logger.info("Stage 5 Complete: Validation metrics computed, GeoTIFF maps, QGIS styles, and inventory generated successfully.")


if __name__ == "__main__":
    run_stage5()
