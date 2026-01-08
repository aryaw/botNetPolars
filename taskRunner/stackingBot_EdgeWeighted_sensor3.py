import os
import gc
import duckdb
import psutil
import numpy as np
import polars as pl
import networkx as nx
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
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

from libInternal import (
    setFileLocation,
    setExportDataLocation,
    fast_label_as_bot_to_binary
)

RANDOM_STATE = 42
SAFE_THREADS = "1"

os.environ.update({
    "OMP_NUM_THREADS": SAFE_THREADS,
    "OPENBLAS_NUM_THREADS": SAFE_THREADS,
    "MKL_NUM_THREADS": SAFE_THREADS,
})

fileTimeStamp, output_dir = setFileLocation()
_, _ = setExportDataLocation()

csv_path = "assets/dataset/NCC2AllSensors_clean.csv"

con = duckdb.connect()
df = pl.from_arrow(
    con.sql(f"""
    SELECT SrcAddr, DstAddr, Proto, Dir, State,
           Dur, TotBytes, TotPkts, sTos, dTos, SrcBytes,
           Label, SensorId
    FROM read_csv_auto('{csv_path}')
    WHERE Label IS NOT NULL
      AND REGEXP_MATCHES(SrcAddr,'^[0-9.]+$')
      AND SensorId = 3
    """).arrow()
)

df = fast_label_as_bot_to_binary(df)

df = df.drop_nulls([
    "SrcAddr","DstAddr","Dir","Proto",
    "Dur","TotBytes","TotPkts"
])

df = df.with_columns(
    pl.col("Dir")
    .cast(pl.Utf8)
    .replace({"->":1,"<-":-1,"<->":0}, default=0)
    .cast(pl.Int32)
)

for c in ["Proto","State"]:
    df = df.with_columns(
        pl.Series(c, LabelEncoder().fit_transform(df[c].to_list()))
    )

features_base = [
    "Dir","Dur","Proto","TotBytes",
    "TotPkts","sTos","dTos","SrcBytes"
]

y = df["label_as_bot"].to_numpy()

idx_train, idx_test = train_test_split(
    np.arange(len(df)),
    test_size=0.3,
    stratify=y,
    random_state=RANDOM_STATE
)

df_train = df[idx_train]
df_test  = df[idx_test]

edge_w = (
    df_train
    .group_by(["SrcAddr","DstAddr"])
    .count()
    .rename({"count":"EdgeWeight"})
)

def attach_edge(d):
    return (
        d.join(edge_w, on=["SrcAddr","DstAddr"], how="left")
         .with_columns(pl.col("EdgeWeight").fill_null(1))
    )

df_train = attach_edge(df_train)
df_test  = attach_edge(df_test)

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

features = features_base + [
    "EdgeWeight","SrcTotalWeight","DstTotalWeight"
]

X_train = df_train.select(features).to_numpy()
X_test  = df_test.select(features).to_numpy()
y_train = df_train["label_as_bot"].to_numpy()
y_test  = df_test["label_as_bot"].to_numpy()

model = StackingClassifier(
    estimators=[
        ("rf", RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=1
        )),
        ("et", ExtraTreesClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
            n_jobs=1
        )),
        ("hgb", HistGradientBoostingClassifier(
            max_iter=100,
            max_depth=8,
            learning_rate=0.05,
            random_state=RANDOM_STATE
        ))
    ],
    final_estimator=LogisticRegression(max_iter=1000),
    stack_method="predict_proba",
    cv=2,
    n_jobs=1
)

model.fit(X_train, y_train)

p_test = model.predict_proba(X_test)[:,1]
fpr, tpr, thr = roc_curve(y_test, p_test)
best_thr = thr[np.argmax(np.sqrt(tpr * (1 - fpr)))]
y_pred = (p_test >= best_thr).astype(int)

print("\n# Global Model Evaluation")
print("Best threshold:", round(float(best_thr),4))
print("Accuracy:", round(accuracy_score(y_test,y_pred)*100,2),"%")
print("Precision:", precision_score(y_test,y_pred))
print("Recall:", recall_score(y_test,y_pred))
print("F1:", f1_score(y_test,y_pred))
print("ROC-AUC:", roc_auc_score(y_test,p_test))

df_all = pl.concat([df_train, df_test])

df_all = df_all.with_columns(
    pl.Series(
        "PredictedProb",
        model.predict_proba(
            df_all.select(features).to_numpy()
        )[:,1]
    )
)

stats_out = (
    df_all.group_by("SrcAddr")
    .agg(
        out_ct=pl.count(),
        out_prob=pl.col("PredictedProb").mean(),
        src_weight=pl.col("SrcTotalWeight").mean()
    )
)

stats_in = (
    df_all.group_by("DstAddr")
    .agg(
        in_ct=pl.count(),
        in_prob=pl.col("PredictedProb").mean(),
        dst_weight=pl.col("DstTotalWeight").mean()
    )
)

stats = (
    stats_in.join(stats_out, left_on="DstAddr", right_on="SrcAddr", how="outer")
    .fill_null(0)
    .with_columns([
        (pl.col("in_ct")+pl.col("out_ct")).alias("degree"),
        ((pl.col("in_prob")+pl.col("out_prob"))/2).alias("avg_prob")
    ])
    .with_columns(
        (
            pl.col("avg_prob")
            * pl.col("degree").log1p()
            * (pl.col("src_weight")+pl.col("dst_weight")).log1p()
        ).alias("cnc_score")
    )
)

top5 = stats.sort("cnc_score", descending=True).head(5)

print("\n# Top-5 C&C Candidates")
for i, r in enumerate(top5.iter_rows(named=True),1):
    print(
        f"{i}. IP={r['DstAddr']} | "
        f"cnc_score={r['cnc_score']:.4f} | "
        f"avg_prob={r['avg_prob']:.3f} | "
        f"degree={int(r['degree'])}"
    )

cc_nodes = top5["DstAddr"].to_list()

avail_gb = psutil.virtual_memory().available / (1024**3)
MAX_NODES, MAX_EDGES = (
    (200,400) if avail_gb < 4 else
    (300,600) if avail_gb < 8 else
    (500,800)
)

G_full = nx.DiGraph()
for r in df_all.select(["SrcAddr","DstAddr"]).iter_rows():
    G_full.add_edge(r[0],r[1])

neighbors = set()
for n in cc_nodes:
    if n in G_full:
        neighbors |= set(G_full.predecessors(n))
        neighbors |= set(G_full.successors(n))

nodes_keep = set(cc_nodes) | neighbors
if len(nodes_keep) > MAX_NODES:
    others = list(nodes_keep - set(cc_nodes))
    sampled = np.random.choice(
        others,
        size=min(len(others), MAX_NODES-len(cc_nodes)),
        replace=False
    )
    nodes_keep = set(cc_nodes) | set(sampled)

G = G_full.subgraph(nodes_keep).copy()

edges = list(G.edges())
if len(edges) > MAX_EDGES:
    idx = np.random.choice(len(edges), MAX_EDGES, replace=False)
    G2 = nx.DiGraph()
    G2.add_nodes_from(G.nodes())
    G2.add_edges_from([edges[i] for i in idx])
    G = G2

pos = nx.spring_layout(G, dim=3, seed=RANDOM_STATE, iterations=60)

edge_x, edge_y, edge_z = [], [], []
for u,v in G.edges():
    x0,y0,z0 = pos[u]
    x1,y1,z1 = pos[v]
    edge_x += [x0,x1,None]
    edge_y += [y0,y1,None]
    edge_z += [z0,z1,None]

node_x,node_y,node_z,colors,sizes,labels = [],[],[],[],[],[]
for n in G.nodes():
    x,y,z = pos[n]
    node_x.append(x)
    node_y.append(y)
    node_z.append(z)
    if n in cc_nodes:
        colors.append("red")
        sizes.append(10)
        labels.append(n)
    else:
        colors.append("blue")
        sizes.append(4)
        labels.append("")

fig = go.Figure([
    go.Scatter3d(x=edge_x,y=edge_y,z=edge_z,mode="lines",
                 line=dict(color="gray",width=1)),
    go.Scatter3d(
        x=node_x,y=node_y,z=node_z,
        mode="markers+text",
        marker=dict(size=sizes,color=colors),
        text=labels,
        textposition="top center",
        textfont=dict(color="orange",size=10)
    )
])

fig.write_html(
    os.path.join(
        output_dir,
        f"EdgeWeighted_SAFE_3DGraph_{fileTimeStamp}.html"
    )
)

gc.collect()
print("\nDONE - Polars-based edge-weighted stacking C&C detection.")
