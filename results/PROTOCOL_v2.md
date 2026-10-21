# Pre-Registered Evaluation Protocol v2: Empirical Audits, Bias Controls & Ladder Refinements

**Date:** October 21, 2026  
**Repository:** `d:\WLF7`  
**Study Region:** Araku Valley, Alluri Sitharama Raju (ASR) District, Eastern Ghats, Andhra Pradesh, India ($18.0^\circ\text{N} - 18.5^\circ\text{N}, 82.5^\circ\text{E} - 83.5^\circ\text{E}$)  
**Pre-Registration Status:** Immutable specification committed to repository history prior to executing audit experiments.

---

## 1. Core Principles & Claims Under Test

This protocol establishes the rigorous methodology for auditing, strengthening, and evaluating the evidence for three explicit scientific claims:

- **Claim 1 (C1):** The multilingual news-to-satellite pipeline builds a usable, spatially localized landslide event inventory where no official historical inventory exists.
- **Claim 2 (C2):** Transportation reporting bias and naive evaluation habits (such as in-sample evaluation and inclusion of distance-to-infrastructure features) severely inflate apparent model skill.
- **Claim 3 (C3):** A simple, well-calibrated Generalized Additive Model (GAM) incorporating real environmental predictors is fully competitive with complex Multi-Layer Perceptrons (MLPs), tuned tree ensembles, and geospatial foundation model embeddings under spatial block cross-validation.

### 1.1 Non-Negotiable Operational Rules
1. **Zero Invented or Hardcoded Numbers:** Every reported metric, CI bound, p-value, sample count, and test statistic must originate from an executed script with saved stdout logs, seeds, and configs in `results/logs/`.
2. **Immutable Pre-Registration:** This document (`results/PROTOCOL_v2.md`) is committed to git prior to script execution. Model folds, feature sets, metric definitions, and decision thresholds cannot be altered post-hoc.
3. **Documentation of Deviations:** Any unavoidable computational or data constraints must be logged with technical rationale in `results/DEVIATIONS.md`.
4. **Abstract Preservation:** `wlf7_extended_abstract.tex` is strictly preserved without modification.

---

## 2. Task Specifications & Evaluation Methodologies

### 2.1 Task 1: Event Provenance & Scar Footprint Audit (Supports C1)
- **Objective:** Audit the provenance, geolocation accuracy, verification status, and physical scar footprints of all 32 landslide events in `results/events.csv`.
- **Audit Table (`results/events_audit.csv`):**
  - Columns: `event_id`, `date`, `toponym`, `lat`, `lon`, `source_type` (`news`, `NDMA`, `GSI_Bhukosh`, `NASA_GLC`, `UGLC`), `source_url`, `coordinate_method` (`micro_toponym_gazetteer`, `survey_monument`, `infrastructure_marker`), `stated_location_accuracy_m`, `verification_status` (`passed_satellite_verification`, `accepted_catalogue_unverified`), `stac_scene_id_pre`, `stac_scene_id_post`, `dndvi_val`, `dsigma0_val`, `verification_thresholds`, `duplicate_of`, `scar_polygon_mapped`, `scar_pixels_count`.
- **Scar Footprint Mapping:**
  - Replace fixed $3\times3$ pixel squares with real multi-pixel scar footprints derived from Copernicus GLO-30 DEM slope filters ($\ge 15^\circ$) combined with localized optical $\Delta\text{NDVI} \ge 0.15$ or SAR $\Delta\sigma^0 \le -1.5\,\text{dB}$ change clusters.
  - Record the exact number of active pixels per event. For events where clouds, vegetation recovery, or micro-extent preclude polygon extraction, document the count and mark as point-buffered.
- **Spot-Check Sample:**
  - Draw a reproducible sample of 8 events using a fixed seed (`seed=42`) with full URLs, scene IDs, and change metrics for human verification.

### 2.2 Task 2: NLP Pipeline Quality Without Manual Labels
- **Objective:** Benchmark the retrieval, geolocation, and filtering efficiency of the multilingual mining pipeline against simpler baselines without circular manual self-labeling.
- **Corpus Accounting:**
  - Total snippet corpus size, source distribution (Eenadu Telugu, Deccan Chronicle, The Hindu, Times of India, GDELT, Google News RSS), language split (English vs. Telugu), and duplicate rate.
- **Candidate Yield Comparison:**
  - Evaluate three text processing methods on the exact same underlying corpus:
    1. *Keyword-Only Filter:* Regex matching for landslide terms (`landslide`, `mudslide`, `rockfall`, `కొండచరియలు`, `మట్టిచరియలు`).
    2. *Keyword + Gazetteer Filter:* Keyword match followed by micro-toponym gazetteer alignment (without classifier).
    3. *Full Pipeline:* Keyword match + IndicBERT/TF-IDF relevance classifier + Micro-toponym gazetteer alignment.
  - Metrics per method: Candidate count ($N_{cand}$), geolocated count ($N_{geo}$), satellite-verified count ($N_{ver}$), and Verification Yield ($\text{Yield} = N_{ver} / N_{cand}$) with 95% bootstrap CIs.
- **Catalogue Cross-Validation (Study Area 2006–2026):**
  - Cross-reference against external public inventories (NASA GLC, NDMA reports, UGLC).
  - Match criteria: Date window $\pm 7$ days, spatial distance $\le 10\,\text{km}$.
  - Report recall of known catalogue events and the number of novel events uncovered by the news pipeline that exist in no official catalogue.
  - If a catalogue is offline or inaccessible via script, mark `NOT RUN` with reason.
- **External Evaluation Export & Scoring:**
  - Export `results/nlp_sample_150.csv` containing 150 stratified snippets (seed 42) with pipeline flags removed and blank annotation columns (`manual_relevance_binary`, `manual_toponym_accurate`, `notes`).
  - Provide `scripts/score_nlp_sample.py` to calculate precision, recall, F1, and bootstrap CIs when external annotations are supplied.

### 2.3 Task 3: Temporal Leakage & Date Audit (Supports C2 & C3)
- **Objective:** Ensure no feature uses data acquired after the event date or proxies linear infrastructure.
- **Feature Metadata Stack (`results/features_table.csv`):**
  - Add acquisition dates/composite windows, product versions, and pre-event status per event.
  - Explicitly flag features containing post-event temporal data (e.g., ESA WorldCover 2021 applied to pre-2021 events, or static post-event satellite composites).
  - Pre-2015 sensor tracking: Document Landsat-7 ETM+ / Landsat-8 OLI composite windows for pre-2015 events.
- **Ablation Experiments Under Spatial Block CV:**
  - Evaluate the impact on PR-AUC when removing potentially leaked or static layers:
    - (a) Baseline L2 GAM (All features).
    - (b) L2 GAM without Land Cover (`land_cover_class` removed).
    - (c) L2 GAM without NDVI (`pre_event_ndvi` removed).
    - (d) L2 GAM without any temporally post-dated or flagged features.
    - (e) Strictly Pre-Event Valid Stack: Only features guaranteed valid prior to every event (Copernicus DEM topography, 30-year climatological monsoon rainfall, bedrock geology).
  - Report paired block-bootstrap differences ($\Delta\text{PR-AUC}$) with 95% CIs against the full L2 GAM.
- **Collinearity with Infrastructure:**
  - Compute Pearson ($r$) and Spearman ($\rho$) correlation coefficients between every environmental feature and `dist_to_road_rail_m`.
  - Flag any feature with $|\rho| > 0.60$ and evaluate model degradation upon removal.

### 2.4 Task 4: Decision Rules, Unified Calibration & Event-Level Metrics (Supports C3)
- **Objective:** Compare all model rungs against the preceding verified best rung using a pre-registered calibration standard and event-level aggregation.
- **Preceding Best Rung:**
  - As established in `PROTOCOL.md` and `RESULTS.md`, Rung L2 (LogisticGAM on environmental splines) is the preceding best rung ($\text{PR-AUC} = 0.2167$), as L3 MLP ($\text{PR-AUC} = 0.2333$) did not achieve a statistically significant improvement ($\Delta\text{PR-AUC} = +0.0156$, 95% CI: `[-0.0094, +0.0469]`).
  - All comparisons for L3 models (RF, XGBoost, LightGBM, SVM, Plain MLP, Stacking), L4 (Prithvi ViT embeddings), and L5 (LoRA adaptation) are computed directly against L2 LogisticGAM.
- **Unified Headline Calibration:**
  - Every model reported in the headline ladder is calibrated using **Platt scaling** (univariate logistic regression fitted on out-of-fold logits strictly on the training blocks).
  - Raw and Isotonic calibrations are reported in secondary tables.
- **Event-Level Aggregation:**
  - To prevent over-representation of large scars, aggregate pixel predictions to the event level using both maximum ($\max_{i \in \text{scar}} P_i$) and mean ($\bar{P}_{\text{scar}}$) predicted probability.
  - Calculate event-level PR-AUC, ROC-AUC, and the percentile rank of each held-out event relative to background cells.
  - Compute both block-level bootstrap (1,000 resamples) and event-level bootstrap (1,000 resamples) 95% CIs.
- **Effective Sample Size Audit:**
  - Explicitly report the effective sample size: total verified events ($N=32$), positive scar pixels, total 10 km blocks ($N=57$), and positive-bearing blocks.
- **Road-Adjacent Exclusion Deep Dive:**
  - Re-examine the subset of events located $>200\,\text{m}$ from roads/railways ($N=110$ pixels). Report the number of unique events and blocks, and compute the Gini coefficient / contribution of the top 2 events to PR-AUC.
- **Rung L0 Re-Evaluation:**
  - Attempt programmatic ingestion of the Geological Survey of India (GSI) National Landslide Susceptibility Mapping (NLSM) layer. If only interactive web viewing is provided with no open GeoTIFF download API, mark `NOT RUN (Data Unavailable)` with exact technical reason.

### 2.5 Task 5: Systematic Reporting Bias Experiments (Supports C2)
- **Objective:** Quantify the exact margin of performance inflation caused by reporting bias, evaluation design, and infrastructure proximity.
- **Comparative Experimental Grid (Identical Folds & Platt Calibration):**
  1. *In-Sample (Resubstitution) vs. Out-of-Fold Spatial Block CV:* Train and test on the same data vs. 10 km spatial block CV for L2 GAM and L3 MLP.
  2. *Feature Inclusion:* Headline model (without road/rail distance) vs. Bias Control model (with `dist_to_road_rail_m`).
  3. *Background Negative Selection:* Unbiased uniform background negatives vs. Bias-matched negatives (matching positive distance deciles).
  4. *Corridor Exclusion:* Full inventory vs. Road-adjacent events excluded ($>200\,\text{m}$).
  5. *Distance-Only Baselines:*
     - Model A: Distance to road/rail only ($P \sim \text{dist\_to\_road\_rail\_m}$).
     - Model B: Distance to road/rail + Elevation ($P \sim \text{dist\_to\_road\_rail\_m} + \text{elevation}$).
- All comparisons reported with 95% block-bootstrap CIs.

### 2.6 Task 6: Raster Generation Debugging & Verification Fix
- **Objective:** Diagnose and rectify the 78.5% "Very High" regional raster artifact and enforce strict numerical consistency.
- **Root Cause Diagnosis:**
  - Investigate nodata masking and spatial bounds in auxiliary rasters (`araku_s2_pre_event_ndvi_30m.tif` and `araku_chirps_monsoon_30m.tif`). Values outside the downloaded bounding box default to 0.0 or nodata (-9999.0), causing spline terms in `LogisticGAM` to extrapolate to positive infinity logits.
- **Corrective Protocol:**
  - Restrict regional map generation strictly to the valid common spatial envelope of all input environmental layers (the active Araku Valley regional domain).
  - Verify that feature scaling, column order, and transformation parameters match the training pipeline.
  - **Numerical Verification Check:** Sample the output GeoTIFF raster at the exact coordinates of all training points. The raster-sampled probability must match the in-memory model prediction to within numerical tolerance:
    $$\max |P_{\text{raster}}(x_i, y_i) - P_{\text{model}}(x_i)| < 10^{-4}$$
  - **Probability Histograms:** Plot and tabulate the predicted probability distributions for training samples, active slope pixels, and corridor buffers, demonstrating concordance with regional geomorphic expectations.
  - Remove any unwarranted assertions that the raster "proves independence from reporting bias."

### 2.7 Task 7: Methodological Limits & Claim Verdicts
- **Shortcut Check Audit:**
  - Report the statistical power and sample size of the pre-2015 vs. post-2015 temporal classifier (positive events: 3 pre-2015, 29 post-2015). Acknowledge that chance-level ROC-AUC (0.5025) demonstrates absence of overt temporal leakage but does not constitute mathematical proof.
- **Foundation Model & Transfer Limits:**
  - Explicitly document the checkpoint weights (`ibm-nasa-geospatial/Prithvi-EO-1.0-100M/Prithvi_100M.pt`, SHA-256: `7fac0c8a8693198e32a055e0c5a967f8b005f382182b63df1b29fdcd5c880731`) and `Landslide4Sense` pretraining details.
  - Clarify the domain mismatch: `Landslide4Sense` is a post-event rapid scar segmentation dataset with optical/SAR changes, whereas the current task is static/pre-event landslide susceptibility mapping. Conclude that performance drops reflect task and covariate shift rather than a general failure of transfer learning.
- **Formal Claim Evaluation:**
  - Grade C1, C2, and C3 as **SUPPORTED**, **PARTLY SUPPORTED**, or **NOT SUPPORTED**, citing paired bootstrap CIs, effect sizes, and primary scientific caveats.

---

## 3. Pre-Registered Decision Criteria

1. **Statistical Superiority:** A model $A$ is deemed statistically **"Better"** than model $B$ if and only if the 95% paired block-bootstrap CI of $\Delta\text{PR-AUC} = \text{PR-AUC}_A - \text{PR-AUC}_B$ strictly excludes zero ($\text{CI}_{lower} > 0.0$).
2. **Equivalence / Competitiveness:** If the 95% CI contains zero, model $A$ is deemed **"Comparable"** to model $B$. If model $A$ is simpler (fewer parameters, no black-box pretraining) and comparable, it is selected as the preferred scientific model under the principle of parsimony.
3. **Inferiority:** If the 95% CI is strictly below zero ($\text{CI}_{upper} < 0.0$), model $A$ is deemed **"Worse"** than model $B$.

---

## 4. Deliverables & Artifact Manifest

1. `results/PROTOCOL_v2.md`: This pre-registered specification document.
2. `results/DEVIATIONS.md`: Log of all technical and computational deviations.
3. `results/RESULTS_v2.md`: Comprehensive final report detailing every number, script, log file, NOT RUN rationale, and claim verdict.
4. `results/events_audit.csv`: Complete provenance and verification audit of all 32 events.
5. `results/nlp_sample_150.csv` & `scripts/score_nlp_sample.py`: Stratified annotation sample (unlabeled) and automated bootstrap evaluation script.
6. `results/features_table.csv`: Updated feature table with temporal windows and infrastructure correlations.
7. `results/cv_metrics.csv` & `results/bootstrap_results.csv`: Updated cross-validation tables with unified Platt calibration and paired bootstrap CIs against Rung L2.
8. `outputs/ladder_best_lsm_30m.tif` & `outputs/ladder_best_uncertainty_30m.tif`: Corrected regional GeoTIFF rasters with verified sampling tolerance.
9. `reproduce_ladder_v2.bat` & `reproduce_ladder_v2.ps1`: One-command end-to-end execution scripts.
