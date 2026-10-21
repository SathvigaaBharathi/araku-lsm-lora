# Automated Landslide Susceptibility Mapping in Data-Sparse Mountainous Terrain

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![QGIS Ready](https://img.shields.io/badge/QGIS-3.x_Styles_Included-589632.svg)](styles/)

An end-to-end framework for automated event inventory generation, bi-temporal satellite verification, and calibrated landslide susceptibility mapping in data-poor tropical mountainous regions, demonstrated across the **Araku Valley** corridor in the Eastern Ghats of India.

---

## 📌 Motivation & Overview

Tropical escarpment environments frequently lack curated, multi-temporal landslide event inventories. Traditional susceptibility modeling in these regions suffers from three severe bottlenecks:
1. **Inventory Scarcity:** No official ground-truth polygon inventory exists for historical slope failures.
2. **Spatial Autocorrelation Leakage:** Standard random cross-validation yields inflated, unreliable performance metrics due to spatial autocorrelation between training and test pixels.
3. **Overfitting on Tabular Baselines:** Applying uncalibrated high-capacity architectures to scarce events produces severe overconfidence and miscalibrated probabilities.

This repository implements an automated, audit-proven pipeline that addresses each challenge:
- **Multilingual NLP Mining:** Mines 20 years (2006–2026) of regional Telugu and English news reports using IndicBERT and localized gazetteer geocoding.
- **Bi-Temporal Satellite Verification:** Automatically queries Planetary Computer STAC to fetch pre- and post-event Sentinel-2 ($\Delta\text{NDVI}$ drop) and Sentinel-1 SAR ($\Delta\sigma^0$ decorrelation) imagery alongside a 30m Copernicus DEM slope gate ($\ge 15^\circ$) to verify scar occurrence.
- **LoRA-Tabular Neural Network:** Adapts tabular representations using low-rank decomposition layers (LoRA) optimized with asymmetric focal loss and post-hoc isotonic calibration.
- **Spatial Block Cross-Validation:** Rigorously evaluated under 5-fold spatial blocking (10 km grid cells) to ensure zero spatial leakage between folds.
- **Epistemic Uncertainty Mapping:** Produces a high-resolution 10m susceptibility probability raster alongside a Monte Carlo dropout uncertainty raster.

---

## 🚀 End-to-End Pipeline

```
  Regional News Snippets (2006-2026)
  [Telugu + English Media Corpus]
                 │
                 ▼
 ┌──────────────────────────────────────┐
 │ Stage 1: IndicBERT NLP Mining        │
 │ • Binary relevance classification    │
 │ • Eastern Ghats gazetteer geocoding  │
 └──────────────────┬───────────────────┘
                    │ Mined Candidate Events
                    ▼
 ┌──────────────────────────────────────┐
 │ Stage 2: Bi-Temporal STAC Validation │
 │ • Sentinel-2 pre/post ΔNDVI drop     │
 │ • Sentinel-1 SAR backscatter change  │
 │ • Copernicus DEM slope gating ≥ 15°  │
 └──────────────────┬───────────────────┘
                    │ 32 Verified Failure Events
                    ▼
 ┌──────────────────────────────────────┐
 │ Stage 3: Feature Engineering         │
 │ • 19 Topo-environmental predictors   │
 │ • Hard negative pseudo-absences      │
 └──────────────────┬───────────────────┘
                    │
                    ▼
 ┌──────────────────────────────────────┐
 │ Stage 4: LoRA-Tabular Architecture   │
 │ • Low-Rank parameter adaptation      │
 │ • Asymmetric focal loss training     │
 │ • Isotonic probability calibration  │
 └──────────────────┬───────────────────┘
                    │
                    ▼
 ┌──────────────────────────────────────┐
 │ Stage 5: Spatial Validation & Output │
 │ • 5-Fold Spatial Block CV            │
 │ • 10m Susceptibility Raster          │
 │ • 10m Monte Carlo Uncertainty Raster │
 └──────────────────────────────────────┘
```

---

## 📊 Rigorous Benchmark Results (5-Fold Spatial Block CV)

All metrics below are computed strictly **out-of-fold** under 5-fold spatial blocking (10 km cells) on real ground observations, guaranteeing that no test event or block was observed during training.

| Model Architecture | Out-of-Fold ROC-AUC | Out-of-Fold PR-AUC | Brier Score | Brier Skill Score (BSS) | 95% Bootstrap PR CI |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **LoRA-Tabular (Calibrated)** | **0.7834** | **0.3273** | **0.0815** | **+0.1083** | **[0.0835, 0.4984]** |
| **LoRA-Tabular (Raw)** | 0.8158 | 0.3277 | 0.0914 | +0.0005 | [0.0784, 0.5399] |
| **SVM (RBF Kernel)** | 0.8013 | 0.3939 | 0.0792 | +0.1337 | [0.0992, 0.5637] |
| **Random Forest** | 0.7863 | 0.2571 | 0.0890 | +0.0261 | [0.0890, 0.4222] |
| **XGBoost** | 0.7295 | 0.1994 | 0.0914 | +0.0008 | [0.0542, 0.3956] |

*Base prevalence: $0.1018$ (334 samples, 34 positives across 34 spatial blocks).*

---

## 📂 Repository Structure

```
.
├── src/
│   ├── stage1_nlp_mining/             # IndicBERT classification, gazetteer retrieval & geocoding
│   ├── stage2_satellite_verification/ # STAC Sentinel-1/2 querying, NDVI drop & slope gating
│   ├── stage3_feature_engineering/    # DEM derivatives, TWI, curvature, neighborhood statistics
│   ├── stage4_geofm_finetune/         # LoRATabularModel architecture & focal loss training
│   ├── stage5_validation/             # Spatial block CV, isotonic calibration & raster export
│   └── utils.py                       # Directory configurations & logging utilities
├── outputs/
│   ├── araku_lsm_10m.tif              # 10m Susceptibility probability raster (DEFLATE compressed)
│   ├── araku_uncertainty_10m.tif      # 10m Epistemic uncertainty raster (DEFLATE compressed)
│   ├── araku_landslides_2006_2026.geojson # 32 satellite-verified historical landslide events
│   ├── spatial_block_cv_summary.csv   # Out-of-fold spatial CV performance summary
│   └── spatial_block_cv_summary.json  # Machine-readable evaluation benchmarks
├── styles/
│   ├── araku_lsm_10m.qml              # QGIS Layer Style for susceptibility (RdYlGn reverse ramp)
│   └── araku_uncertainty_10m.qml      # QGIS Layer Style for epistemic uncertainty (Magma ramp)
├── gazetteer/
│   └── eastern_ghats_toponyms.geojson # Curated regional toponym gazetteer for geocoding
├── data/
│   └── processed/                     # Verified training samples and spatial block partitions
├── spatial_cv_eval.py                 # Standalone script to reproduce the 5-fold spatial block CV
├── main.py                            # Pipeline execution entry point
├── requirements.txt                   # Environment dependencies
└── .gitignore                         # Standard exclusions
```

---

## 🛠️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/SathvigaaBharathi/araku-lsm-lora.git
cd araku-lsm-lora
```

### 2. Set Up Virtual Environment
```bash
python -m venv venv
# Linux / macOS:
source venv/bin/activate
# Windows:
.\venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🔬 Reproduction & Usage

### Reproduce Spatial Block Cross-Validation
To reproduce the 5-fold spatial block cross-validation metrics and bootstrap confidence intervals:
```bash
python spatial_cv_eval.py
```

### Execute Pipeline Stages
To run individual stages or the full pipeline:
```bash
# Run complete pipeline
python main.py

# Or run individual modules:
python -m src.stage1_nlp_mining.classifier
python -m src.stage2_satellite_verification.verify
python -m src.stage3_feature_engineering.negative_sampling
python -m src.stage4_geofm_finetune.train
python -m src.stage5_validation.validate
```

### Visualizing Rasters in QGIS
1. Open **QGIS** (3.x or later).
2. Drag and drop `outputs/araku_lsm_10m.tif` and `outputs/araku_uncertainty_10m.tif`.
3. Right-click each layer $\to$ **Properties** $\to$ **Symbology** $\to$ **Style** $\to$ **Load Style...**
4. Select `styles/araku_lsm_10m.qml` for susceptibility, and `styles/araku_uncertainty_10m.qml` for epistemic uncertainty.

---

## 📜 License
This project is open-source and licensed under the [MIT License](LICENSE).
