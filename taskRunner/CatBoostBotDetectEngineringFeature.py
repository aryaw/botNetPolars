import polars as pl
import numpy as np
from datetime import datetime
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import networkx as nx
import plotly.graph_objects as go
from libInternal.db import get_mysql_engine

TEST_SIZE = 0.2
RANDOM_STATE = 42
TOP_K_CNC = 3

def render_3d_graph(df_edges: pl.DataFrame, cnc_nodes: set, title: str, out_html: str):
    if df_edges.height == 0 or not cnc_nodes:
        return

    G = nx.DiGraph()
    for r in df_edges.iter_rows(named=True):
        G.add_edge(r["SrcAddr"], r["DstAddr"], weight=r["EdgeWeight"])

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
        dataset_source,
        SrcAddr,
        DstAddr,
        Proto,
        dir_clean,

        featureEng_ByteRatio,
        featureEng_DurationRate,
        featureEng_FlowIntensity,
        featureEng_PktByteRatio,
        featureEng_SrcByteRatio,
        featureEng_TrafficBalance,
        featureEng_DurationPerPkt,
        featureEng_Intensity,

        EdgeWeight,
        SrcTotalWeight,
        DstTotalWeight,

        label_as_bot
    FROM v_all_flows
    WHERE will_be_drop = 0
    """

    df = pl.read_database(query, engine)

    feature_cols = [
        "Proto",
        "dir_clean",
        "featureEng_ByteRatio",
        "featureEng_DurationRate",
        "featureEng_FlowIntensity",
        "featureEng_PktByteRatio",
        "featureEng_SrcByteRatio",
        "featureEng_TrafficBalance",
        "featureEng_DurationPerPkt",
        "featureEng_Intensity",
        "EdgeWeight",
        "SrcTotalWeight",
        "DstTotalWeight",
    ]

    X = df.select(feature_cols)
    y = df.select("label_as_bot").to_numpy().ravel()

    cat_features = [
        feature_cols.index("Proto"),
        feature_cols.index("dir_clean"),
    ]

    X_np = X.to_numpy()

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X_np,
        y,
        np.arange(len(y)),
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
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
        df
        .select(["SrcAddr", "DstAddr", "EdgeWeight", "SrcTotalWeight", "DstTotalWeight"])
        .with_row_index(name="row_id")
        .filter(pl.col("row_id").is_in(idx_test))
        .with_columns(pl.Series("PredLabel", y_pred))
    )

    bot_flows = test_df.filter(pl.col("PredLabel") == 1)

    cnc_nodes = (
        bot_flows
        .groupby("SrcAddr")
        .agg([
            pl.count().alias("flows"),
            pl.mean("SrcTotalWeight").alias("out_weight"),
            pl.mean("DstTotalWeight").alias("in_weight"),
        ])
        .with_columns(
            (pl.col("out_weight") / (pl.col("in_weight") + 1))
            .alias("out_in_ratio")
        )
        .sort("out_in_ratio", descending=True)
        .head(TOP_K_CNC)
    )

    print("\nDetected C&C nodes:")
    print(cnc_nodes)

    cnc_set = set(cnc_nodes["SrcAddr"].to_list())

    if cnc_set:
        edges_df = (
            bot_flows
            .filter(pl.col("SrcAddr").is_in(cnc_set))
            .select(["SrcAddr", "DstAddr", "EdgeWeight"])
        )

        out_html = f"CatBoost_FeatureEng_CNC_3D_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"

        render_3d_graph(
            edges_df,
            cnc_set,
            title="3D C&C Communication Graph (CatBoostFeature Engineered)",
            out_html=out_html
        )

if __name__ == "__main__":
    main()
