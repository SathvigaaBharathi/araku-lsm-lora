"""
Spatial block cross-validation for the Araku LSM pilot (out-of-fold metrics).

Why: calibrator.predict_proba(X_raw) after fitting on X_raw scores the model on
its own training data, so those AUC values are in-sample. Here every sample is
scored by a model that never saw its spatial block.

EDIT the CONFIG block, then run:  python spatial_cv_eval.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.calibration import CalibratedClassifierCV, IsotonicRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
import torch
import torch.nn as nn

# ---------------- CONFIG ----------------
BASE_DIR = Path(__file__).resolve().parent
CSV = BASE_DIR / "data" / "processed" / "training_samples.csv"
LABEL_COL = "landslide_label"   # 1 = positive, 0 = negative
BLOCK_COL = "block_id"          # the 10 km block id (e.g. "921_202")
N_FOLDS, N_BOOT, SEED = 5, 1000, 0
BLOCK_SIZE_DEG = 0.09
# ----------------------------------------

sys.path.append(str(BASE_DIR))
from src.stage4_geofm_finetune.train import LoRATabularModel, SimpleGeoFMLoRAModel, FocalLoss, FEATURE_COLS


def fit_predict_lora_tabular(Xtr, ytr, Xte, wtr=None, epochs=60, lr=0.005, wd=1e-3,
                             alpha=0.85, gamma=3.0, class_scale=3.0, lora_r=8, seed=SEED):
    """Real LoRATabularModel training from train.py: 60 epochs, lr 0.005, AdamW, focal loss."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    mean_tr = np.mean(Xtr, axis=0)
    std_tr = np.std(Xtr, axis=0) + 1e-6
    Xtr_norm = (Xtr - mean_tr) / std_tr
    Xte_norm = (Xte - mean_tr) / std_tr

    X_t = torch.tensor(Xtr_norm, dtype=torch.float32)
    y_t = torch.tensor(ytr, dtype=torch.float32).unsqueeze(1)
    Xte_t = torch.tensor(Xte_norm, dtype=torch.float32)

    if wtr is None:
        raw_w = torch.ones_like(y_t)
    else:
        raw_w = torch.tensor(wtr, dtype=torch.float32).unsqueeze(1)

    c_scale = torch.where(y_t == 1, torch.tensor(class_scale), torch.tensor(1.0))
    loss_weights = raw_w * c_scale

    in_features = Xtr.shape[1]
    model = LoRATabularModel(in_features=in_features, hidden_dim=64, lora_r=lora_r, dropout_p=0.25)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    criterion = FocalLoss(alpha=alpha, gamma=gamma)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        preds = model(X_t)
        loss = criterion(preds, y_t, sample_weight=loss_weights)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        p_te = model(Xte_t).squeeze(-1).numpy()
    return p_te


def fit_predict_lora_tabular_calibrated(Xtr, ytr, gtr, Xte, wtr=None):
    """Isotonic calibration for LoRA-Tabular fitted on TRAINING blocks only (grouped inner CV)."""
    inner = list(GroupKFold(n_splits=3).split(Xtr, ytr, gtr))
    p_inner_oof = np.zeros(len(ytr), dtype=np.float32)
    for in_tr, in_val in inner:
        w_in = wtr[in_tr] if wtr is not None else None
        p_val = fit_predict_lora_tabular(Xtr[in_tr], ytr[in_tr], Xtr[in_val], wtr=w_in)
        p_inner_oof[in_val] = p_val
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_inner_oof, ytr)
    p_raw = fit_predict_lora_tabular(Xtr, ytr, Xte, wtr=wtr)
    return iso.predict(p_raw)

# Aliases
fit_predict_geofm = fit_predict_lora_tabular
fit_predict_geofm_calibrated = fit_predict_lora_tabular_calibrated


def baselines():
    from sklearn.base import clone
    from xgboost import XGBClassifier
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=SEED),
        "SVM": make_pipeline(StandardScaler(), SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=SEED)),
        "XGBoost": XGBClassifier(n_estimators=300, eval_metric="logloss", random_state=SEED)
    }
    return {k: clone(v) for k, v in models.items()}


def make_folds(blocks, y):
    """Group folds by block; spread the positive-bearing blocks across folds."""
    rng = np.random.default_rng(SEED)
    ub = pd.Series(blocks).unique()
    pos_b = [b for b in ub if y[blocks == b].sum() > 0]
    neg_b = [b for b in ub if b not in pos_b]
    rng.shuffle(pos_b)
    rng.shuffle(neg_b)
    fold_of = {}
    for i, b in enumerate(pos_b):
        fold_of[b] = i % N_FOLDS
    for i, b in enumerate(neg_b):
        fold_of[b] = (i + len(pos_b)) % N_FOLDS
    return np.array([fold_of[b] for b in blocks])


def calibrated(model, Xtr, ytr, gtr, Xte, wtr=None):
    """Isotonic calibration fitted on the TRAINING blocks only (grouped inner CV)."""
    inner = list(GroupKFold(n_splits=3).split(Xtr, ytr, gtr))
    c = CalibratedClassifierCV(model, method="isotonic", cv=inner)
    c.fit(Xtr, ytr, sample_weight=wtr)
    return c.predict_proba(Xte)[:, 1]


def metrics(y, p, weights=None):
    prev = np.average(y, weights=weights) if weights is not None else y.mean()
    brier = brier_score_loss(y, p, sample_weight=weights)
    bss = 1.0 - brier / (prev * (1.0 - prev) + 1e-12)
    roc = roc_auc_score(y, p, sample_weight=weights)
    pr = average_precision_score(y, p, sample_weight=weights)
    return {
        "ROC_AUC": float(roc),
        "PR_AUC": float(pr),
        "Brier": float(brier),
        "BSS": float(bss),
        "no_skill_PR": float(prev)
    }


def block_bootstrap(y, p, blocks):
    rng = np.random.default_rng(SEED)
    ub = pd.Series(blocks).unique()
    idx = {b: np.where(blocks == b)[0] for b in ub}
    out, skipped = [], 0
    for _ in range(N_BOOT):
        pick = np.concatenate([idx[b] for b in rng.choice(ub, len(ub))])
        if y[pick].sum() == 0 or y[pick].sum() == len(pick):
            skipped += 1
            continue
        out.append((roc_auc_score(y[pick], p[pick]), average_precision_score(y[pick], p[pick])))
    a = np.array(out)
    return np.percentile(a, [2.5, 97.5], axis=0), skipped


def main():
    df = pd.read_csv(CSV)
    df["block_x"] = (df["longitude"] / BLOCK_SIZE_DEG).astype(int)
    df["block_y"] = (df["latitude"] / BLOCK_SIZE_DEG).astype(int)
    df["block_id"] = df["block_x"].astype(str) + "_" + df["block_y"].astype(str)

    y = df[LABEL_COL].to_numpy()
    blocks = df[BLOCK_COL].astype(str).to_numpy()
    w = df["sample_weight"].to_numpy()
    X = df[FEATURE_COLS].to_numpy(float)

    fold = make_folds(blocks, y)
    print(f"{len(y)} samples, {int(y.sum())} positives, prevalence {y.mean():.4f}, "
          f"{len(set(blocks))} blocks, {len(set(blocks[y == 1]))} with positives")

    models = {"LoRA-Tabular (Calibrated)": None, "LoRA-Tabular (Raw)": None, **baselines()}
    oof_preds = {name: np.zeros(len(y)) for name in models.keys()}

    for f in range(N_FOLDS):
        tr, te = fold != f, fold == f
        Xtr, ytr, gtr = X[tr], y[tr], blocks[tr]
        Xte, yte = X[te], y[te]
        wtr = w[tr]

        for name, m in models.items():
            if name == "LoRA-Tabular (Raw)":
                oof_preds[name][te] = fit_predict_lora_tabular(Xtr, ytr, Xte, wtr=wtr)
            elif name == "LoRA-Tabular (Calibrated)":
                oof_preds[name][te] = fit_predict_lora_tabular_calibrated(Xtr, ytr, gtr, Xte, wtr=wtr)
            else:
                oof_preds[name][te] = calibrated(m, Xtr, ytr, gtr, Xte, wtr=wtr)

    print("\n--- UNWEIGHTED OUT-OF-FOLD SPATIAL BLOCK CV ---")
    rows = {}
    for name in models.keys():
        oof = oof_preds[name]
        r = metrics(y, oof)
        (lo, hi), skipped = block_bootstrap(y, oof, blocks)
        r.update(ROC_CI=f"{lo[0]:.4f}-{hi[0]:.4f}", PR_CI=f"{lo[1]:.4f}-{hi[1]:.4f}", boot_skipped=skipped)
        rows[name] = r
    print(pd.DataFrame(rows).T.round(4).to_string())

    print("\n--- SAMPLE-WEIGHTED OUT-OF-FOLD SPATIAL BLOCK CV ---")
    rows_w = {}
    for name in models.keys():
        oof = oof_preds[name]
        r_w = metrics(y, oof, weights=w)
        rows_w[name] = r_w
    print(pd.DataFrame(rows_w).T.round(4).to_string())


if __name__ == "__main__":
    main()
