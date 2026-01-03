import polars as pl
import pandas as pd

from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

import networkx as nx
import plotly.graph_objects as go
from libInternal.db import get_mysql_engine


def build_cc_graph(df_bot: pd.DataFrame):
    G = nx.DiGraph()

    for _, row in df_bot.iterrows():
        src = row["SrcAddr"]
        dst = row["DstAddr"]
        weight = row.get("EdgeWeight", 1)

        G.add_edge(src, dst, weight=weight)

    pos = nx.spring_layout(G, k=0.8, seed=42)

    edge_x, edge_y = [], []
    for src, dst in G.edges():
        x0, y0 = pos[src]
        x1, y1 = pos[dst]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    fig = go.Figure(
        data=[go.Scatter(x=edge_x, y=edge_y, mode="lines")],
        layout=go.Layout(title="C&C Graph (Raw Feature)")
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
        TotBytes,
        TotPkts,
        Dur,
        EdgeWeight,
        SrcTotalWeight,
        DstTotalWeight,
        label_as_bot
    FROM scenario9
    WHERE will_be_drop = 0
    """

    df = pl.read_database(query, engine)
    pdf = df.to_pandas()

    feature_cols = [
        "Proto",
        "dir_clean",
        "TotBytes",
        "TotPkts",
        "Dur",
        "EdgeWeight",
        "SrcTotalWeight",
        "DstTotalWeight",
    ]

    X = pdf[feature_cols]
    y = pdf["label_as_bot"]

    cat_features = ["Proto", "dir_clean"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model = CatBoostClassifier(
        iterations=300,
        depth=8,
        learning_rate=0.1,
        loss_function="Logloss",
        auto_class_weights="Balanced",
        verbose=100
    )

    model.fit(X_train, y_train, cat_features=cat_features)

    p_test = model.predict_proba(X_test)[:, 1]
    y_pred = (p_test >= 0.5).astype(int)

    print("\n# RAW FEATURE MODEL")
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall   :", recall_score(y_test, y_pred))
    print("F1       :", f1_score(y_test, y_pred))
    print("ROC-AUC  :", roc_auc_score(y_test, p_test))

    df_bot = pdf.loc[X_test.index][y_pred == 1]
    if len(df_bot) > 0:
        build_cc_graph(df_bot)


if __name__ == "__main__":
    main()
