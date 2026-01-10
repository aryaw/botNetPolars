import polars as pl
import pandas as pd

from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

import networkx as nx
import plotly.graph_objects as go
from libInternal.db import get_mysql_engine


def build_cc_graph(df_bot: pd.DataFrame):
    G = nx.DiGraph()

    for _, row in df_bot.iterrows():
        src = row["SrcAddr"]
        dst = row["DstAddr"]
        weight = row.get("EdgeWeight", 1)

        if G.has_edge(src, dst):
            G[src][dst]["weight"] += weight
        else:
            G.add_edge(src, dst, weight=weight)

    pos = nx.spring_layout(G, k=0.8, seed=42)

    # edges -----
    edge_x, edge_y, edge_width = [], [], []
    for src, dst, data in G.edges(data=True):
        x0, y0 = pos[src]
        x1, y1 = pos[dst]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        edge_width.append(min(data["weight"] / 5, 6))

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=1, color="#888"),
        hoverinfo="none",
        mode="lines"
    )

    # nodes -----
    node_x, node_y, node_text, node_size = [], [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

        deg = G.degree(node)
        node_size.append(8 + deg * 3)
        node_text.append(f"{node}<br>Degree: {deg}")

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers",
        hoverinfo="text",
        text=node_text,
        marker=dict(
            size=node_size,
            color=node_size,
            colorscale="Reds",
            showscale=True,
            colorbar=dict(title="Node Degree"),
            line_width=1
        )
    )

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title="C&C Graph",
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40)
        )
    )

    fig.show()


def main():
    engine = get_mysql_engine()

    query = """
    SELECT
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
    FROM scenario9
    WHERE will_be_drop = 0
    """

    df = pl.read_database(query, engine)
    print("Data shape:", df.shape)
    print(df.head())
    pdf = df.to_pandas()

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
    ]

    X = pdf[feature_cols]
    y = pdf["label_as_bot"]

    cat_features = ["Proto", "dir_clean"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    model = CatBoostClassifier(
        iterations=300,
        depth=8,
        learning_rate=0.1,
        loss_function="Logloss",
        eval_metric="AUC",
        auto_class_weights="Balanced",
        verbose=100
    )

    model.fit(X_train, y_train, cat_features=cat_features)

    p_test = model.predict_proba(X_test)[:, 1]
    best_threshold = 0.5
    y_pred_test = (p_test >= best_threshold).astype(int)

    print("\n# Global Model Evaluation")
    print("Best threshold:", round(float(best_threshold), 4))
    print("Accuracy:", round(float((y_pred_test == y_test).mean() * 100), 2), "%")
    print("Precision:", precision_score(y_test, y_pred_test))
    print("Recall:", recall_score(y_test, y_pred_test))
    print("F1:", f1_score(y_test, y_pred_test))
    print("ROC-AUC:", roc_auc_score(y_test, p_test))

    pdf_test = pdf.loc[X_test.index].copy()
    pdf_test["pred_bot"] = y_pred_test

    df_bot = pdf_test[pdf_test["pred_bot"] == 1]
    print(f"\nDetected bot flows: {len(df_bot)}")

    if len(df_bot) > 0:
        build_cc_graph(df_bot)
    else:
        print("No bot flows detected, graph not generated.")

if __name__ == "__main__":
    main()
