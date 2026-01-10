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

os.environ.update({
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
})

DATA_DIR = "dataset/src"
ALGO_NAME = "cascade_rf_rf_merged"

FEATURES = [
    "DirEnc", "Dur", "ProtoEnc",
    "TotBytes", "TotPkts", "SrcBytes",
]

REQUIRED_COLS = [
    "SrcAddr", "DstAddr", "Dir", "Proto",
    "Dur", "TotBytes", "TotPkts", "SrcBytes", "Label",
]

files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".binetflow"))
dfs = []

for fname in files:
    print(f"[load] {fname}")
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
print("[info] total flow:", df.height)

df = df.with_columns([
    pl.when(pl.col("Dir") == "->").then(1)
      .when(pl.col("Dir") == "<-").then(-1)
      .otherwise(0)
      .alias("DirEnc"),

    pl.col("Proto").cast(pl.Categorical).to_physical().alias("ProtoEnc"),
    pl.col("label_as_bot").cast(pl.Int8),
])

X = df.select(FEATURES).to_numpy()
y = df["label_as_bot"].to_numpy()
idx = np.arange(len(y))

X_tr, X_te, y_tr, y_te, idx_tr, idx_te = train_test_split(
    X, y, idx,
    test_size=0.3,
    stratify=y,
    random_state=RANDOM_STATE,
)

ts, out_dir = setFileLocation()
base_dir = os.path.join(out_dir["html"], ALGO_NAME)
os.makedirs(base_dir, exist_ok=True)

# STAGE 1 : NORMAL vs BOT
stage1_dir = os.path.join(base_dir, "stage1")
os.makedirs(stage1_dir, exist_ok=True)

rf1 = RandomForestClassifier(
    n_estimators=300,
    max_depth=20,
    class_weight="balanced",
    n_jobs=1,
    random_state=RANDOM_STATE,
)
rf1.fit(X_tr, y_tr)

p1_te = rf1.predict_proba(X_te)[:, 1]
fpr, tpr, thr = roc_curve(y_te, p1_te)
thr1 = thr[np.argmax(np.sqrt(tpr * (1 - fpr)))]

y_pred1 = (p1_te >= thr1).astype(int)

print("\n# evaluasi stage-1")
print("accuracy :", round(accuracy_score(y_te, y_pred1) * 100, 3), "%")
print("precision:", precision_score(y_te, y_pred1, zero_division=0))
print("recall   :", recall_score(y_te, y_pred1, zero_division=0))
print("f1       :", f1_score(y_te, y_pred1, zero_division=0))
print("roc-auc  :", roc_auc_score(y_te, p1_te))

export_confusion_matrix_html(
    y_te, y_pred1, f"{ALGO_NAME}_stage1", stage1_dir
)
export_evaluation_table_html(
    y_te, y_pred1, p1_te, f"{ALGO_NAME}_stage1", stage1_dir
)

df_stage1 = df[idx_te].with_columns(
    pl.Series("PredictedProb", p1_te)
)

edge1 = (
    df_stage1.group_by(["SrcAddr", "DstAddr"])
    .len()
    .rename({"len": "EdgeWeight"})
)

df_stage1 = df_stage1.join(
    edge1, on=["SrcAddr", "DstAddr"], how="left"
).with_columns(
    pl.col("EdgeWeight").fill_null(1)
)

cc_stage1 = (
    df_stage1.group_by("SrcAddr")
    .agg(pl.mean("PredictedProb").alias("score"))
    .sort("score", descending=True)
    .head(5)
    .select("SrcAddr")
    .to_series()
    .to_list()
)

with open(os.path.join(stage1_dir, "cnc_candidates.txt"), "w") as f:
    for ip in cc_stage1:
        f.write(ip + "\n")

export_cnc_graph_3d_edgeweighted(
    df_all=df_stage1,
    cc_nodes=cc_stage1,
    algo_name=f"{ALGO_NAME}_stage1",
    output_dir=stage1_dir,
)

# STAGE 2 : CASCADE + C&C INFERENCE
stage2_dir = os.path.join(base_dir, "stage2")
os.makedirs(stage2_dir, exist_ok=True)

mask_tr = rf1.predict_proba(X_tr)[:, 1] >= thr1
X_tr2 = X_tr[mask_tr]
y_tr2 = y_tr[mask_tr]

rf2 = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    class_weight="balanced",
    n_jobs=1,
    random_state=RANDOM_STATE,
)
rf2.fit(X_tr2, y_tr2)

p_final = np.zeros(len(X_te))
mask_te = p1_te >= thr1
if mask_te.any():
    p_final[mask_te] = rf2.predict_proba(X_te[mask_te])[:, 1]

y_pred2 = (p_final >= 0.5).astype(int)

print("\n# evaluasi stage-2")
print("accuracy :", round(accuracy_score(y_te, y_pred2) * 100, 2), "%")
print("precision:", precision_score(y_te, y_pred2, zero_division=0))
print("recall   :", recall_score(y_te, y_pred2, zero_division=0))
print("f1       :", f1_score(y_te, y_pred2, zero_division=0))
print("roc-auc  :", roc_auc_score(y_te, p_final))

export_confusion_matrix_html(
    y_te, y_pred2, f"{ALGO_NAME}_stage2", stage2_dir
)
export_evaluation_table_html(
    y_te, y_pred2, p_final, f"{ALGO_NAME}_stage2", stage2_dir
)

df_stage2 = df[idx_te].with_columns(
    pl.Series("PredictedProb", p_final)
)

edge2 = (
    df_stage2.group_by(["SrcAddr", "DstAddr"])
    .len()
    .rename({"len": "EdgeWeight"})
)

df_stage2 = df_stage2.join(
    edge2, on=["SrcAddr", "DstAddr"], how="left"
).with_columns(
    pl.col("EdgeWeight").fill_null(1)
)

stats_out = (
    df_stage2.group_by("SrcAddr")
    .agg([
        pl.len().alias("out_ct"),
        pl.mean("PredictedProb").alias("out_prob"),
        pl.sum("EdgeWeight").alias("src_weight"),
    ])
)

stats_in = (
    df_stage2.group_by("DstAddr")
    .agg([
        pl.len().alias("in_ct"),
        pl.mean("PredictedProb").alias("in_prob"),
        pl.sum("EdgeWeight").alias("dst_weight"),
    ])
)

stats = (
    stats_in.join(
        stats_out,
        left_on="DstAddr",
        right_on="SrcAddr",
        how="full"
    )
    .with_columns(
        pl.coalesce(["DstAddr", "SrcAddr"]).alias("Node")
    )
    .drop(["DstAddr", "SrcAddr"])
    .fill_null(0)
)

stats = stats.with_columns([
    (pl.col("in_ct") + pl.col("out_ct")).alias("degree"),
    ((pl.col("in_prob") + pl.col("out_prob")) / 2).alias("avg_prob"),
])

stats = stats.with_columns(
    (
        pl.col("avg_prob")
        * (pl.col("degree") + 1).log()
        * (pl.col("src_weight") + pl.col("dst_weight") + 1).log()
    ).alias("cnc_score")
)

cc_stage2 = (
    stats.sort("cnc_score", descending=True)
    .head(5)
    .select("Node")
    .to_series()
    .to_list()
)

with open(os.path.join(stage2_dir, "cnc_candidates.txt"), "w") as f:
    for ip in cc_stage2:
        f.write(ip + "\n")

export_cnc_graph_3d_edgeweighted(
    df_all=df_stage2,
    cc_nodes=cc_stage2,
    algo_name=f"{ALGO_NAME}_stage2",
    output_dir=stage2_dir,
)

gc.collect()
print("\nselesai - pipeline per-stage lengkap")
