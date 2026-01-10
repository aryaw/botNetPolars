import os
import gc
import numpy as np
import polars as pl
import psutil
import networkx as nx
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    StackingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve
)

from libInternal.dFHelper import (
    fast_label_as_bot_to_binary,
    setFileLocation,
    export_confusion_matrix_html,
    export_evaluation_table_html,
    export_cnc_graph_3d_edgeweighted,
)

# =========================
# HARD SAFETY (NATIVE CRASH)
# =========================
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

RANDOM_STATE = 42
DATA_DIR = "dataset/src"
ALGO_NAME = "stacking_edgeweighted_polars"

MAX_TRAIN_FLOWS = 1_500_000

REQUIRED_COLS = [
    "SrcAddr","DstAddr",
    "Dir","Proto",
    "Dur","TotBytes","TotPkts","SrcBytes",
    "Label"
]

BASE_FEATURES = [
    "DirEnc","Dur","ProtoEnc",
    "TotBytes","TotPkts","SrcBytes"
]

# =========================
# OUTPUT DIR
# =========================
ts, out_dir = setFileLocation()
base_dir = os.path.join(out_dir["html"], ALGO_NAME)
os.makedirs(base_dir, exist_ok=True)

# =========================
# LOAD & NORMALIZE DATA
# =========================
dfs = []
files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".binetflow"))
if not files:
    raise RuntimeError("no binetflow files")

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

    df = df.with_columns(
        pl.col("Dir")
        .cast(pl.Utf8)
        .str.replace(r"^\s+|\s+$", "")
        .alias("Dir")
    )

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

# =========================
# CAST (MEMORY SAFETY)
# =========================
df = df.with_columns([
    pl.col("Dur").cast(pl.Float32),
    pl.col("TotBytes").cast(pl.Float32),
    pl.col("TotPkts").cast(pl.Float32),
    pl.col("SrcBytes").cast(pl.Float32),
])

# =========================
# FEATURE ENCODING
# =========================
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

# =========================
# TRAIN SAMPLE (ANTI SEGFAULT)
# =========================
df_train_pool = df
if df.height > MAX_TRAIN_FLOWS:
    df_train_pool = df.sample(
        n=MAX_TRAIN_FLOWS,
        seed=RANDOM_STATE,
        with_replacement=False
    )
    print("[warn] training sampled to", df_train_pool.height)

y_pool = df_train_pool["label_as_bot"].to_numpy()
idx_pool = np.arange(len(df_train_pool))

idx_tr, idx_te = train_test_split(
    idx_pool,
    test_size=0.3,
    stratify=y_pool,
    random_state=RANDOM_STATE,
)

df_train = df_train_pool[idx_tr]
df_test  = df_train_pool[idx_te]

# =========================
# GRAPH FEATURES (TRAIN ONLY)
# =========================
edge_w = (
    df_train
    .group_by(["SrcAddr","DstAddr"])
    .len()
    .rename({"len":"EdgeWeight"})
)

def attach_edge(d):
    return (
        d.join(edge_w, on=["SrcAddr","DstAddr"], how="left")
         .with_columns(pl.col("EdgeWeight").fill_null(1))
    )

df_train = attach_edge(df_train)
df_test  = attach_edge(df_test)
df       = attach_edge(df)

src_total = (
    df_train.group_by("SrcAddr")
    .agg(pl.col("EdgeWeight").sum().alias("SrcTotalWeight"))
)

dst_total = (
    df_train.group_by("DstAddr")
    .agg(pl.col("EdgeWeight").sum().alias("DstTotalWeight"))
)

def attach_totals(d):
    return (
        d.join(src_total, on="SrcAddr", how="left")
         .join(dst_total, on="DstAddr", how="left")
         .fill_null(0)
    )

df_train = attach_totals(df_train)
df_test  = attach_totals(df_test)
df       = attach_totals(df)

FEATURES = BASE_FEATURES + [
    "EdgeWeight","SrcTotalWeight","DstTotalWeight"
]

# =========================
# MODEL (STACKING SAFE)
# =========================
model = StackingClassifier(
    estimators=[
        ("rf", RandomForestClassifier(
            n_estimators=50,
            max_depth=12,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=1
        )),
        ("et", ExtraTreesClassifier(
            n_estimators=50,
            random_state=RANDOM_STATE,
            n_jobs=1
        )),
        ("hgb", HistGradientBoostingClassifier(
            max_iter=100,
            max_depth=8,
            learning_rate=0.05,
            random_state=RANDOM_STATE
        )),
    ],
    final_estimator=LogisticRegression(max_iter=1000),
    stack_method="predict_proba",
    cv=2,
    n_jobs=1
)

X_tr = df_train.select(FEATURES).to_numpy()
y_tr = df_train["label_as_bot"].to_numpy()
model.fit(X_tr, y_tr)

# =========================
# EVALUATION (SAMPLE TEST)
# =========================
X_te = df_test.select(FEATURES).to_numpy()
y_te = df_test["label_as_bot"].to_numpy()

p_te = model.predict_proba(X_te)[:,1]
fpr, tpr, thr = roc_curve(y_te, p_te)
best_thr = thr[np.argmax(np.sqrt(tpr * (1 - fpr)))]
y_pred = (p_te >= best_thr).astype(int)

print("\n# Global Model Evaluation (sample)")
print("Accuracy:", round(accuracy_score(y_te,y_pred)*100,2),"%")
print("Precision:", precision_score(y_te,y_pred))
print("Recall:", recall_score(y_te,y_pred))
print("F1:", f1_score(y_te,y_pred))
print("ROC-AUC:", roc_auc_score(y_te,p_te))

export_confusion_matrix_html(
    y_te, y_pred, ALGO_NAME, base_dir
)
export_evaluation_table_html(
    y_te, y_pred, p_te, ALGO_NAME, base_dir
)

# =========================
# INFERENCE (FULL DATA)
# =========================
df = df.with_columns(
    pl.Series(
        "PredictedProb",
        model.predict_proba(
            df.select(FEATURES).to_numpy()
        )[:,1]
    )
)

# =========================
# C&C INFERENCE (FULL GRAPH)
# =========================
stats_out = (
    df.group_by("SrcAddr")
    .agg([
        pl.len().alias("out_ct"),
        pl.mean("PredictedProb").alias("out_prob"),
        pl.mean("SrcTotalWeight").alias("src_weight"),
    ])
)

stats_in = (
    df.group_by("DstAddr")
    .agg([
        pl.len().alias("in_ct"),
        pl.mean("PredictedProb").alias("in_prob"),
        pl.mean("DstTotalWeight").alias("dst_weight"),
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
        pl.coalesce(["DstAddr","SrcAddr"]).alias("Node")
    )
    .drop(["DstAddr","SrcAddr"])
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

top5 = stats.sort("cnc_score", descending=True).head(5)
cc_nodes = top5["Node"].to_list()

with open(os.path.join(base_dir,"cnc_candidates.txt"),"w") as f:
    for ip in cc_nodes:
        f.write(ip+"\n")

export_cnc_graph_3d_edgeweighted(
    df_all=df,
    cc_nodes=cc_nodes,
    algo_name=ALGO_NAME,
    output_dir=base_dir,
)

gc.collect()
print("\nDONE - stacking (train-sample, infer-full) - NO SEGFAULT.")
