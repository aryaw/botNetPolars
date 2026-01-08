"""
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
from sklearn.feature_selection import (
    chi2,
    f_classif,
    mutual_info_classif,
)
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

# config umum
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
    "DirEnc",
    "Dur",
    "ProtoEnc",
    "TotBytes",
    "TotPkts",
    "SrcBytes",
]

# load dan gabungkan semua dataset binetflow
dfs = []
files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".binetflow"))

if not files:
    raise RuntimeError("tidak ada file binetflow")

for fname in files:
    print(f"[load] {fname}")
    df = pl.read_csv(
        os.path.join(DATA_DIR, fname),
        separator=",",
        infer_schema_length=10000,
        ignore_errors=True,
    )

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

# encoding fitur
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

X_all = df.select(BASE_FEATURES).to_numpy()
y_all = df["label_as_bot"].to_numpy()

idx = np.arange(len(y_all))
X_tr, X_te, y_tr, y_te, idx_tr, idx_te = train_test_split(
    X_all,
    y_all,
    idx,
    test_size=0.3,
    stratify=y_all,
    random_state=RANDOM_STATE,
)

# seleksi feature ensemble terinspirasi sb-net
rank_scores = {f: [] for f in BASE_FEATURES}

chi_vals, _ = chi2(np.abs(X_tr), y_tr)
for f, v in zip(BASE_FEATURES, chi_vals):
    rank_scores[f].append(v)

f_vals, _ = f_classif(X_tr, y_tr)
for f, v in zip(BASE_FEATURES, f_vals):
    rank_scores[f].append(v)

mi_vals = mutual_info_classif(X_tr, y_tr, random_state=RANDOM_STATE)
for f, v in zip(BASE_FEATURES, mi_vals):
    rank_scores[f].append(v)

dt = DecisionTreeClassifier(random_state=RANDOM_STATE)
dt.fit(X_tr, y_tr)
for f, v in zip(BASE_FEATURES, dt.feature_importances_):
    rank_scores[f].append(v)

rf_fs = RandomForestClassifier(
    n_estimators=200,
    random_state=RANDOM_STATE,
    n_jobs=1,
)
rf_fs.fit(X_tr, y_tr)
for f, v in zip(BASE_FEATURES, rf_fs.feature_importances_):
    rank_scores[f].append(v)

perm = permutation_importance(
    rf_fs,
    X_tr,
    y_tr,
    n_repeats=5,
    random_state=RANDOM_STATE,
    n_jobs=1,
)
for f, v in zip(BASE_FEATURES, perm.importances_mean):
    rank_scores[f].append(v)

borda_score = {
    f: np.sum(np.argsort(np.argsort(vals)))
    for f, vals in rank_scores.items()
}

final_features = sorted(
    borda_score.keys(),
    key=lambda x: borda_score[x],
    reverse=True,
)

print("\n[ranking feature akhir]")
for i, f in enumerate(final_features, 1):
    print(f"{i}. {f}")

# evaluasi top-7, top-5, dan top-3
ts, out_dir = setFileLocation()
results = {}

for k in [7, 5, 3]:
    feats = final_features[:k]
    algo_name = f"{ALGO_BASE}_top{k}"
    print(f"\n[evaluasi] {algo_name}")

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

    p1_te = rf1.predict_proba(X_te_k)[:, 1]
    fpr, tpr, thr = roc_curve(y_te, p1_te)
    thr1 = thr[np.argmax(np.sqrt(tpr * (1 - fpr)))]

    mask_tr = rf1.predict_proba(X_tr_k)[:, 1] >= thr1
    X_tr2 = X_tr_k[mask_tr]
    y_tr2 = y_tr[mask_tr]

    rf2 = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        class_weight="balanced",
        n_jobs=1,
        random_state=RANDOM_STATE,
    )
    rf2.fit(X_tr2, y_tr2)

    p_final = np.zeros(len(X_te_k))
    mask_te = p1_te >= thr1
    if mask_te.any():
        p_final[mask_te] = rf2.predict_proba(X_te_k[mask_te])[:, 1]

    y_pred = (p_final >= 0.5).astype(int)

    results[k] = {
        "accuracy": accuracy_score(y_te, y_pred),
        "precision": precision_score(y_te, y_pred, zero_division=0),
        "recall": recall_score(y_te, y_pred, zero_division=0),
        "f1": f1_score(y_te, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_te, p_final),
    }

    algo_dir = os.path.join(out_dir["html"], algo_name)
    os.makedirs(algo_dir, exist_ok=True)

    export_confusion_matrix_html(y_te, y_pred, algo_name, algo_dir)
    export_evaluation_table_html(y_te, y_pred, p_final, algo_name, algo_dir)

# tabel perbandingan jumlah feature
rows = []
for k, m in results.items():
    rows.append({
        "num_features": k,
        "accuracy": round(m["accuracy"], 4),
        "precision": round(m["precision"], 4),
        "recall": round(m["recall"], 4),
        "f1": round(m["f1"], 4),
        "roc_auc": round(m["roc_auc"], 4),
    })

df_compare = pl.DataFrame(rows).sort("num_features", descending=True)
print("\n[tabel perbandingan feature]")
print(df_compare)

df_compare.write_csv(
    os.path.join(out_dir["csv"], f"{ALGO_BASE}_feature_comparison.csv")
)
df_compare.to_pandas().to_html(
    os.path.join(out_dir["html"], f"{ALGO_BASE}_feature_comparison.html"),
    index=False,
    float_format="%.4f"
)

# pilih konfigurasi terbaik berdasarkan f1
best_k = max(results.keys(), key=lambda k: results[k]["f1"])
print(f"\n[graph] konfigurasi terbaik berdasarkan f1 -> top-{best_k} feature")

feats = final_features[:best_k]
X_tr_k = df[idx_tr].select(feats).to_numpy()
X_te_k = df[idx_te].select(feats).to_numpy()

rf1.fit(X_tr_k, y_tr)
p1_te = rf1.predict_proba(X_te_k)[:, 1]

mask_te = p1_te >= thr1
p_final = np.zeros(len(X_te_k))
if mask_te.any():
    p_final[mask_te] = rf2.predict_proba(X_te_k[mask_te])[:, 1]

df_test = df[idx_te].with_columns(
    pl.Series("PredictedProb", p_final)
)

stats_out = (
    df_test.groupby("SrcAddr")
    .agg([
        pl.count().alias("out_ct"),
        pl.mean("PredictedProb").alias("out_prob"),
    ])
)

stats_in = (
    df_test.groupby("DstAddr")
    .agg([
        pl.count().alias("in_ct"),
        pl.mean("PredictedProb").alias("in_prob"),
    ])
)

stats = stats_in.join(stats_out, how="outer").fill_null(0)

stats = stats.with_columns([
    (pl.col("in_ct") + pl.col("out_ct")).alias("degree"),
    ((pl.col("in_prob") + pl.col("out_prob")) / 2).alias("avg_prob"),
])

stats = stats.with_columns(
    (pl.col("avg_prob") * (pl.col("degree") + 1).log()).alias("cnc_score")
)

cc_nodes = (
    stats.sort("cnc_score", descending=True)
    .head(5)
    .select("SrcAddr")
    .to_series()
    .to_list()
)

algo_dir = os.path.join(out_dir["html"], f"{ALGO_BASE}_best")
os.makedirs(algo_dir, exist_ok=True)

export_cnc_graph_3d_edgeweighted(
    df_all=df_test,
    cc_nodes=cc_nodes,
    algo_name=f"{ALGO_BASE}_best",
    output_dir=algo_dir,
)

gc.collect()
print("\nselesai - sb-net inspired cascade + feature selection")
