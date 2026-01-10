"""
-- Graph-Assisted Botnet Detection via Cascade Learning with Ensemble Feature Ranking --

Inspired by SB-Net, this study adopts an ensemble feature selection strategy
with rank aggregation. To ensure scalability on large-scale flow datasets,
wrapper-based methods are replaced with model-based importance measures
while preserving the ensemble ranking principle.

This adaptation maintains methodological consistency with SB-Net while
enabling efficient integration with graph-based C&C inference.
"""

import sys
import os
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

import gc
import numpy as np
import polars as pl

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)

from libInternal.dFHelper import (
    fast_label_as_bot_to_binary,
    setFileLocation,
    export_confusion_matrix_html,
    export_evaluation_table_html,
    export_cnc_graph_3d_edgeweighted,
)

RANDOM_STATE = 42
SAFE_THREADS = "1"

os.environ.update({
    "OMP_NUM_THREADS": SAFE_THREADS,
    "OPENBLAS_NUM_THREADS": SAFE_THREADS,
    "MKL_NUM_THREADS": SAFE_THREADS,
})

DATA_DIR = "dataset/src"
ALGO_BASE = "cascade_rf_rf_sbnet_fs"

BASE_FEATURES = [
    "DirEnc", "Dur", "ProtoEnc",
    "TotBytes", "TotPkts", "SrcBytes",
]

REQUIRED_COLS = [
    "SrcAddr", "DstAddr",
    "Dir", "Proto",
    "Dur", "TotBytes", "TotPkts", "SrcBytes",
    "Label",
]

def export_stage(df_stage, probs, y_true, y_pred, algo_name, stage_dir):
    export_confusion_matrix_html(y_true, y_pred, algo_name, stage_dir)
    export_evaluation_table_html(y_true, y_pred, probs, algo_name, stage_dir)

    edge = (
        df_stage.group_by(["SrcAddr", "DstAddr"])
        .len()
        .rename({"len": "EdgeWeight"})
    )

    df_stage = (
        df_stage.join(edge, on=["SrcAddr", "DstAddr"], how="left")
        .with_columns(pl.col("EdgeWeight").fill_null(1))
    )

    cc_nodes = (
        df_stage.group_by("SrcAddr")
        .agg(pl.mean("PredictedProb").alias("score"))
        .sort("score", descending=True)
        .head(5)
        .select("SrcAddr")
        .to_series()
        .to_list()
    )

    with open(os.path.join(stage_dir, "cnc_candidates.txt"), "w") as f:
        for ip in cc_nodes:
            f.write(ip + "\n")

    export_cnc_graph_3d_edgeweighted(
        df_all=df_stage,
        cc_nodes=cc_nodes,
        algo_name=algo_name,
        output_dir=stage_dir,
    )

dfs = []
files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".binetflow"))
if not files:
    raise RuntimeError("tidak ada file binetflow")

for fname in files:
    df = pl.read_csv(
        os.path.join(DATA_DIR, fname),
        separator=",",
        infer_schema_length=10000,
        ignore_errors=True,
    )

    if any(c not in df.columns for c in REQUIRED_COLS):
        continue

    df = df.select(REQUIRED_COLS)
    df = fast_label_as_bot_to_binary(df)

    df = df.filter(
        pl.all_horizontal([
            pl.col("SrcAddr").is_not_null(),
            pl.col("DstAddr").is_not_null(),
            pl.col("Dur").is_not_null(),
            pl.col("TotBytes").is_not_null(),
            pl.col("TotPkts").is_not_null(),
        ])
    )

    if df.height > 0:
        dfs.append(df)

df = pl.concat(dfs, how="vertical")

df = df.with_columns([
    pl.when(pl.col("Dir") == "->").then(1)
      .when(pl.col("Dir") == "<-").then(-1)
      .otherwise(0).alias("DirEnc"),
    pl.col("Proto").cast(pl.Categorical).to_physical().alias("ProtoEnc"),
    pl.col("label_as_bot").cast(pl.Int8),
])

X = df.select(BASE_FEATURES).to_numpy()
y = df["label_as_bot"].to_numpy()
idx = np.arange(len(y))

X_tr, X_te, y_tr, y_te, idx_tr, idx_te = train_test_split(
    X, y, idx,
    test_size=0.3,
    stratify=y,
    random_state=RANDOM_STATE,
)

rank_scores = {f: [] for f in BASE_FEATURES}

chi_vals, _ = chi2(np.abs(X_tr), y_tr)
f_vals, _ = f_classif(X_tr, y_tr)
mi_vals = mutual_info_classif(X_tr, y_tr, random_state=RANDOM_STATE)

dt = DecisionTreeClassifier(random_state=RANDOM_STATE)
dt.fit(X_tr, y_tr)

rf_fs = RandomForestClassifier(
    n_estimators=200,
    random_state=RANDOM_STATE,
    n_jobs=1,
)
rf_fs.fit(X_tr, y_tr)

perm = permutation_importance(
    rf_fs,
    X_tr,
    y_tr,
    n_repeats=5,
    random_state=RANDOM_STATE,
    n_jobs=1,
)

for i, f in enumerate(BASE_FEATURES):
    rank_scores[f].extend([
        chi_vals[i],
        f_vals[i],
        mi_vals[i],
        dt.feature_importances_[i],
        rf_fs.feature_importances_[i],
        perm.importances_mean[i],
    ])

borda = {
    f: np.sum(np.argsort(np.argsort(v)))
    for f, v in rank_scores.items()
}

final_features = sorted(borda, key=lambda x: borda[x], reverse=True)

ts, out_dir = setFileLocation()
results = {}

for k in [7, 5, 3]:
    feats = final_features[:k]
    algo_name = f"{ALGO_BASE}_top{k}"

    root = os.path.join(out_dir["html"], algo_name)
    s1 = os.path.join(root, "stage1")
    s2 = os.path.join(root, "stage2")
    os.makedirs(s1, exist_ok=True)
    os.makedirs(s2, exist_ok=True)

    X_tr_k = df[idx_tr].select(feats).to_numpy()
    X_te_k = df[idx_te].select(feats).to_numpy()

    rf1 = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        class_weight="balanced",
        n_jobs=1,
        random_state=RANDOM_STATE,
    )
    rf1.fit(X_tr_k, y_tr)

    p1 = rf1.predict_proba(X_te_k)[:, 1]
    fpr, tpr, thr = roc_curve(y_te, p1)
    thr1 = thr[np.argmax(np.sqrt(tpr * (1 - fpr)))]
    y1 = (p1 >= thr1).astype(int)

    export_stage(
        df[idx_te].with_columns(pl.Series("PredictedProb", p1)),
        p1, y_te, y1,
        f"{algo_name}_stage1", s1
    )

    mask_tr = rf1.predict_proba(X_tr_k)[:, 1] >= thr1
    rf2 = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        class_weight="balanced",
        n_jobs=1,
        random_state=RANDOM_STATE,
    )
    rf2.fit(X_tr_k[mask_tr], y_tr[mask_tr])

    p2 = np.zeros(len(X_te_k))
    mask_te = p1 >= thr1
    if mask_te.any():
        p2[mask_te] = rf2.predict_proba(X_te_k[mask_te])[:, 1]

    y2 = (p2 >= 0.5).astype(int)

    export_stage(
        df[idx_te].with_columns(pl.Series("PredictedProb", p2)),
        p2, y_te, y2,
        f"{algo_name}_stage2", s2
    )

    results[k] = {
        "accuracy": accuracy_score(y_te, y2),
        "precision": precision_score(y_te, y2, zero_division=0),
        "recall": recall_score(y_te, y2, zero_division=0),
        "f1": f1_score(y_te, y2, zero_division=0),
        "roc_auc": roc_auc_score(y_te, p2),
    }

rows = [{"num_features": k, **results[k]} for k in results]
pl.DataFrame(rows).write_csv(
    os.path.join(out_dir["csv"], f"{ALGO_BASE}_feature_comparison.csv")
)

gc.collect()
print("\nselesai - sb-net inspired cascade (per-stage export lengkap)")
