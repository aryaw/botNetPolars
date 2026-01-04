import polars as pl
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import networkx as nx
import plotly.graph_objects as go
from libInternal.db import get_mysql_engine

TEST_SIZE = 0.2
RANDOM_STATE = 42
TOP_K_CNC = 3

def render_cc_graph_3d(df_cnc: pl.DataFrame, highlight_nodes: set):
    G = nx.DiGraph()
    for r in df_cnc.iter_rows(named=True):
        G.add_edge(r["SrcAddr"], r["DstAddr"], weight=r["edge_weight"])
    if G.number_of_nodes() == 0:
        return
    pos = nx.spring_layout(G, dim=3, seed=42)
    edge_x, edge_y, edge_z = [], [], []
    for u, v in G.edges():
        x0, y0, z0 = pos[u]
        x1, y1, z1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        edge_z += [z0, z1, None]
    edge_trace = go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z, mode="lines",
        line=dict(width=1, color="gray"), hoverinfo="none"
    )
    node_x, node_y, node_z = [], [], []
    node_color, node_size, labels = [], [], []
    for n in G.nodes():
        x, y, z = pos[n]
        node_x.append(x)
        node_y.append(y)
        node_z.append(z)
        if n in highlight_nodes:
            node_color.append("red")
            node_size.append(18)
        else:
            node_color.append("blue")
            node_size.append(6)
        labels.append(n)
    node_trace = go.Scatter3d(
        x=node_x, y=node_y, z=node_z, mode="markers",
        marker=dict(size=node_size, color=node_color, opacity=0.85),
        text=labels, hoverinfo="text"
    )
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title="3D C&C Communication Graph (Raw + Graph Features)",
            showlegend=False,
            margin=dict(l=0, r=0, t=40, b=0)
        )
    )
    fig.show()

def main():
    engine = get_mysql_engine()
    query = """
    SELECT
        dataset_source,
        SrcAddr,
        DstAddr,
        Proto,
        dir_clean,
        TotBytes,
        TotPkts,
        Dur,
        edge_weight,
        src_total_weight,
        dst_total_weight,
        label_as_bot
    FROM v_all_flows
    WHERE will_be_drop = 0
    """
    df = pl.read_database(query, engine)
    feature_cols = [
        "Proto",
        "dir_clean",
        "TotBytes",
        "TotPkts",
        "Dur",
        "edge_weight",
        "src_total_weight",
        "dst_total_weight",
    ]
    X = df.select(feature_cols)
    y = df.select("label_as_bot").to_numpy().ravel()
    cat_features = [feature_cols.index("Proto"), feature_cols.index("dir_clean")]
    X_np = X.to_numpy()
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X_np, y, np.arange(len(y)),
        test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    model = CatBoostClassifier(
        iterations=300,
        depth=8,
        learning_rate=0.1,
        loss_function="Logloss",
        auto_class_weights="Balanced",
        random_seed=RANDOM_STATE,
        verbose=100
    )
    model.fit(X_train, y_train, cat_features=cat_features)
    p_test = model.predict_proba(X_test)[:, 1]
    y_pred = (p_test >= 0.5).astype(int)
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall   :", recall_score(y_test, y_pred))
    print("F1-score :", f1_score(y_test, y_pred))
    print("ROC-AUC  :", roc_auc_score(y_test, p_test))
    test_df = (
        df.select(["SrcAddr", "DstAddr", "edge_weight", "src_total_weight", "dst_total_weight"])
        .with_row_count("row_id")
        .filter(pl.col("row_id").is_in(idx_test))
        .with_columns(pl.Series("y_pred", y_pred))
    )
    bot_flows = test_df.filter(pl.col("y_pred") == 1)
    cnc_nodes = (
        bot_flows
        .groupby("SrcAddr")
        .agg([
            pl.count().alias("flows"),
            pl.mean("src_total_weight").alias("out_weight"),
            pl.mean("dst_total_weight").alias("in_weight"),
        ])
        .with_columns((pl.col("out_weight") / (pl.col("in_weight") + 1)).alias("out_in_ratio"))
        .sort("out_in_ratio", descending=True)
        .head(TOP_K_CNC)
    )
    print(cnc_nodes)
    detected_cnc_ips = set(cnc_nodes["SrcAddr"].to_list())
    if detected_cnc_ips:
        df_graph = bot_flows.filter(pl.col("SrcAddr").is_in(detected_cnc_ips))
        render_cc_graph_3d(df_graph, detected_cnc_ips)

if __name__ == "__main__":
    main()
