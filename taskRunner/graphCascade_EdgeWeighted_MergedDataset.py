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

# config umum
RANDOM_STATE = 42

os.environ.update({
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
})

DATA_DIR = "dataset/src"
ALGO_NAME = "cascade_rf_rf_merged"

FEATURES = [
    "DirEnc",
    "Dur",
    "ProtoEnc",
    "TotBytes",
    "TotPkts",
    "SrcBytes",
]

REQUIRED_COLS = [
    "SrcAddr",
    "DstAddr",
    "Dir",
    "Proto",
    "Dur",
    "TotBytes",
    "TotPkts",
    "SrcBytes",
    "Label",
]

# kumpulkan semua file binetflow
files = sorted(
    f for f in os.listdir(DATA_DIR)
    if f.endswith(".binetflow")
)

if not files:
    raise RuntimeError("tidak ada file binetflow")

dfs = []

# load dan samakan schema
for fname in files:
    print(f"[load] {fname}")
    path = os.path.join(DATA_DIR, fname)

    df = pl.read_csv(
        path,
        separator=",",
        infer_schema_length=10000,
        ignore_errors=True,
    )

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        print(f"[skip] {fname}, missing kolom: {missing}")
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

if not dfs:
    raise RuntimeError("dataset kosong setelah filtering")

df = pl.concat(dfs, how="vertical")
print("[info] total flow:", df.height)

# feature engineering dasar
df = df.with_columns([
    pl.when(pl.col("Dir") == "->").then(1)
      .when(pl.col("Dir") == "<-").then(-1)
      .otherwise(0)
      .alias("DirEnc"),

    pl.col("Proto")
      .cast(pl.Categorical)
      .to_physical()
      .alias("ProtoEnc"),

    pl.col("label_as_bot").cast(pl.Int8),
])

# split train test global
X = df.select(FEATURES).to_numpy()
y = df["label_as_bot"].to_numpy()
idx = np.arange(len(y))

X_tr, X_te, y_tr, y_te, idx_tr, idx_te = train_test_split(
    X,
    y,
    idx,
    test_size=0.3,
    stratify=y,
    random_state=RANDOM_STATE,
)

# stage 1: normal vs bot
rf_stage1 = RandomForestClassifier(
    n_estimators=300,
    max_depth=20,
    class_weight="balanced",
    n_jobs=1,
    random_state=RANDOM_STATE,
)

rf_stage1.fit(X_tr, y_tr)

p1_tr = rf_stage1.predict_proba(X_tr)[:, 1]
p1_te = rf_stage1.predict_proba(X_te)[:, 1]

fpr, tpr, thr = roc_curve(y_te, p1_te)
thr_stage1 = thr[np.argmax(np.sqrt(tpr * (1 - fpr)))]

# stage 2: bot hasil stage 1
mask_bot_tr = p1_tr >= thr_stage1
X_tr_2 = X_tr[mask_bot_tr]
y_tr_2 = y_tr[mask_bot_tr]

if X_tr_2.shape[0] == 0:
    raise RuntimeError("stage 2 kosong, threshold terlalu tinggi")

rf_stage2 = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    class_weight="balanced",
    n_jobs=1,
    random_state=RANDOM_STATE,
)

rf_stage2.fit(X_tr_2, y_tr_2)

# inferensi cascade
p_final = np.zeros(len(X_te))
mask_bot_te = p1_te >= thr_stage1

if mask_bot_te.any():
    p_final[mask_bot_te] = rf_stage2.predict_proba(
        X_te[mask_bot_te]
    )[:, 1]

y_pred = (p_final >= 0.5).astype(int)

# evaluasi global
print("\n# evaluasi global merged dataset")
print("accuracy :", round(accuracy_score(y_te, y_pred) * 100, 2), "%")
print("precision:", precision_score(y_te, y_pred, zero_division=0))
print("recall   :", recall_score(y_te, y_pred, zero_division=0))
print("f1       :", f1_score(y_te, y_pred, zero_division=0))
print("roc-auc  :", roc_auc_score(y_te, p_final))

# setup output
ts, out_dir = setFileLocation()
algo_dir = os.path.join(out_dir["html"], ALGO_NAME)
os.makedirs(algo_dir, exist_ok=True)

export_confusion_matrix_html(
    y_te, y_pred, ALGO_NAME, algo_dir
)

export_evaluation_table_html(
    y_te, y_pred, p_final, ALGO_NAME, algo_dir
)

# dataframe test dengan probabilitas
df_test = df[idx_te].with_columns(
    pl.Series("PredictedProb", p_final)
)

# hitung edge weight
edge_w = (
    df_test
    .groupby(["SrcAddr", "DstAddr"])
    .count()
    .rename({"count": "EdgeWeight"})
)

df_test = df_test.join(
    edge_w,
    on=["SrcAddr", "DstAddr"],
    how="left"
).with_columns(
    pl.col("EdgeWeight").fill_null(1)
)

# agregasi node source
stats_out = (
    df_test.groupby("SrcAddr")
    .agg([
        pl.count().alias("out_ct"),
        pl.mean("PredictedProb").alias("out_prob"),
        pl.sum("EdgeWeight").alias("src_weight"),
    ])
)

# agregasi node destination
stats_in = (
    df_test.groupby("DstAddr")
    .agg([
        pl.count().alias("in_ct"),
        pl.mean("PredictedProb").alias("in_prob"),
        pl.sum("EdgeWeight").alias("dst_weight"),
    ])
)

stats = stats_in.join(stats_out, how="outer").fill_null(0)

# hitung cnc score
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

# top kandidat c&c
top_nodes = (
    stats.sort("cnc_score", descending=True)
    .head(10)
    .select("SrcAddr")
    .to_series()
    .to_list()
)

print("\n# top kandidat c&c (merged)")
for i, ip in enumerate(top_nodes, 1):
    score = stats.filter(pl.col("SrcAddr") == ip)["cnc_score"][0]
    print(f"{i}. {ip} | cnc_score={score:.4f}")

# render graph cnc 3d
export_cnc_graph_3d_edgeweighted(
    df_all=df_test,
    cc_nodes=top_nodes,
    algo_name=ALGO_NAME,
    output_dir=algo_dir,
)

gc.collect()
print("\nselesai - merged dataset")
