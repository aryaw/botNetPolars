import os
import gc
import psutil
import numpy as np
import polars as pl
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve
)
import networkx as nx
import plotly.graph_objects as go
from libInternal.db import get_mysql_engine

RANDOM_STATE = 42
TEST_SIZE = 0.30
MAX_ROWS = 13_000_000


def log_ram(tag=""):
    p = psutil.Process()
    rss = p.memory_info().rss / (1024 ** 2)
    print(f"[RAM] {tag:<30} {rss:8.2f} MB")


def render_3d_graph(df_edges: pl.DataFrame, cnc_nodes: set, title: str, out_html: str):
    if df_edges.height == 0 or not cnc_nodes:
        return

    G = nx.DiGraph()
    for r in df_edges.iter_rows(named=True):
        G.add_edge(r["SrcAddr"], r["DstAddr"], weight=r["edge_weight"])

    if G.number_of_nodes() == 0:
        return

    pos = nx.spring_layout(G, dim=3, seed=RANDOM_STATE, iterations=60)

    edge_x, edge_y, edge_z = [], [], []
    for u, v in G.edges():
        x0, y0, z0 = pos[u]
        x1, y1, z1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        edge_z += [z0, z1, None]

    edge_trace = go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode="lines",
        line=dict(color="#B0B0B0", width=1.2),
        hoverinfo="none"
    )

    node_x, node_y, node_z = [], [], []
    node_color, node_size, labels = [], [], []

    for n in G.nodes():
        x, y, z = pos[n]
        node_x.append(x)
        node_y.append(y)
        node_z.append(z)
        if n in cnc_nodes:
            node_color.append("#E41A1C")
            node_size.append(20)
            labels.append(n)
        else:
            node_color.append("#377EB8")
            node_size.append(6)
            labels.append("")

    node_trace = go.Scatter3d(
        x=node_x, y=node_y, z=node_z,
        mode="markers+text",
        marker=dict(size=node_size, color=node_color, opacity=0.9),
        text=labels,
        textposition="top center",
        hoverinfo="text"
    )

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=title,
            showlegend=False,
            margin=dict(l=0, r=0, t=40, b=0),
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
            )
        )
    )

    fig.write_html(out_html)
    print(f"[Plot] Graph saved -> {out_html}")


def main():
    engine = get_mysql_engine()

    query = """
    SELECT
        SrcAddr,
        DstAddr,
        Proto,
        dir_clean AS Dir,
        Dur,
        TotBytes,
        TotPkts,
        SrcBytes,
        sTos,
        dTos,

        edge_weight,
        src_total_weight,
        dst_total_weight,

        featureEng_ByteRatio,
        featureEng_DurationRate,
        featureEng_FlowIntensity,
        featureEng_PktByteRatio,
        featureEng_SrcByteRatio,
        featureEng_TrafficBalance,
        featureEng_DurationPerPkt,
        featureEng_Intensity,

        label_as_bot
    FROM sensor3
    WHERE will_be_drop = 0
    """

    print("[Load] Reading data from MySQL")
    df = pl.read_database(query, engine)
    log_ram("After Load")

    features = [
        "Dir", "Dur", "Proto", "TotBytes", "TotPkts",
        "sTos", "dTos", "SrcBytes",
        "featureEng_ByteRatio",
        "featureEng_DurationRate",
        "featureEng_FlowIntensity",
        "featureEng_PktByteRatio",
        "featureEng_SrcByteRatio",
        "featureEng_TrafficBalance",
        "featureEng_DurationPerPkt",
        "featureEng_Intensity",
        "edge_weight",
        "src_total_weight",
        "dst_total_weight",
    ]

    X = df.select(features).to_numpy()
    y = df.select("label_as_bot").to_numpy().ravel().astype(int)

    X = np.nan_to_num(X, posinf=0.0, neginf=0.0)

    if len(y) > MAX_ROWS:
        idx = np.random.RandomState(RANDOM_STATE).choice(len(y), MAX_ROWS, replace=False)
        X = X[idx]
        y = y[idx]
        df = df.take(idx)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    base_learners = [
        ("rf", RandomForestClassifier(
            n_estimators=120, max_depth=12,
            class_weight="balanced", random_state=RANDOM_STATE, n_jobs=1
        )),
        ("et", ExtraTreesClassifier(
            n_estimators=120, random_state=RANDOM_STATE, n_jobs=1
        )),
        ("hgb", HistGradientBoostingClassifier(
            max_iter=120, max_depth=8, learning_rate=0.05,
            random_state=RANDOM_STATE
        )),
    ]

    model = StackingClassifier(
        estimators=base_learners,
        final_estimator=LogisticRegression(max_iter=1000),
        stack_method="predict_proba",
        cv=2,
        n_jobs=1
    )

    print("[Train] Training stacking model (ENGINEERING FEATURES)")
    model.fit(X_train, y_train)
    log_ram("After Train")

    p_test = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thr = roc_curve(y_test, p_test)
    best_thr = thr[np.argmax(np.sqrt(tpr * (1 - fpr)))]

    y_pred = (p_test >= best_thr).astype(int)

    print("\n# Evaluation (ENGINEERING FEATURES)")
    print("Threshold:", round(float(best_thr), 4))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall   :", recall_score(y_test, y_pred))
    print("F1       :", f1_score(y_test, y_pred))
    print("ROC-AUC  :", roc_auc_score(y_test, p_test))

    print("[Infer] Running full inference")
    X_all = scaler.transform(np.nan_to_num(df.select(features).to_numpy()))
    p_all = model.predict_proba(X_all)[:, 1]
    pred_all = (p_all >= best_thr).astype(int)

    df = df.with_columns([
        pl.Series("PredProb", p_all),
        pl.Series("PredLabel", pred_all),
    ])

    bot_df = df.filter(pl.col("PredLabel") == 1)

    cnc_nodes = (
        bot_df
        .groupby("SrcAddr")
        .agg([
            pl.count().alias("flows"),
            pl.mean("src_total_weight").alias("out_weight"),
            pl.mean("dst_total_weight").alias("in_weight"),
        ])
        .with_columns(
            (pl.col("out_weight") / (pl.col("in_weight") + 1)).alias("out_in_ratio")
        )
        .sort("out_in_ratio", descending=True)
        .head(5)
    )

    cnc_set = set(cnc_nodes["SrcAddr"].to_list())

    print("\nDetected C&C nodes:")
    print(cnc_nodes)

    edges_df = bot_df.filter(pl.col("SrcAddr").is_in(cnc_set)) \
        .select(["SrcAddr", "DstAddr", "edge_weight"])

    out_html = f"Sensor3_Stacking_Engineering_CNC_3D_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    render_3d_graph(
        edges_df,
        cnc_set,
        title="3D C&C Communication Graph (Stacking ML - Engineering Features)",
        out_html=out_html
    )

    gc.collect()
    log_ram("Script End")


if __name__ == "__main__":
    main()
