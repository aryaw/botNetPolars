import os
from datetime import datetime
import re
import numpy as np
import polars as pl
import networkx as nx
import plotly.graph_objects as go
import plotly.figure_factory as ff

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

BOT_REGEX = (
    r"\b(bot|botnet|cnc|c&c|malware|infected|attack|spam|ddos|"
    r"trojan|worm|zombie|backdoor)\b"
)


def fast_label_as_bot_to_binary(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(
        pl.col("Label")
        .cast(pl.Utf8)
        .fill_null("")
        .str.to_lowercase()
        .alias("_label_str")
    )

    df = df.with_columns(
        pl.when(pl.col("_label_str").str.contains(BOT_REGEX))
        .then(1)
        .otherwise(0)
        .alias("_label_text")
    )

    df = df.with_columns(
        pl.col("Label")
        .cast(pl.Float64, strict=False)
        .alias("_label_num")
    )

    df = df.with_columns(
        pl.when(pl.col("_label_num").is_not_null())
        .then((pl.col("_label_num") >= 0.5).cast(pl.Int8))
        .otherwise(pl.col("_label_text"))
        .alias("label_as_bot")
    )

    print("[label_as_bot] value counts:")
    print(df.select(pl.col("label_as_bot").value_counts(sort=True)))

    return df.drop(["_label_str", "_label_text", "_label_num"])

def setFileLocation():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(base_dir, ".."))

    output_root = os.path.join(project_root, "outputs")
    html_dir = os.path.join(output_root, "html")
    csv_dir = os.path.join(output_root, "csv")
    img_dir = os.path.join(output_root, "img")

    for d in [html_dir, csv_dir, img_dir]:
        os.makedirs(d, exist_ok=True)

    return ts, {
        "root": output_root,
        "html": html_dir,
        "csv": csv_dir,
        "img": img_dir,
    }

def setExportDataLocation():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(base_dir, ".."))

    output_datadir = os.path.join(project_root, "outputsData")
    os.makedirs(output_datadir, exist_ok=True)

    return ts, output_datadir

def get_algo_output_dir(base_output_dir, algo_name):
    algo_dir = os.path.join(base_output_dir["html"], algo_name.lower())
    os.makedirs(algo_dir, exist_ok=True)
    return algo_dir

def export_confusion_matrix_html(y_true, y_pred, algo_name, output_dir):
    cm = confusion_matrix(y_true, y_pred)

    fig = ff.create_annotated_heatmap(
        z=cm,
        x=["Normal", "Bot"],
        y=["Normal", "Bot"],
        colorscale="Blues"
    )

    fig.update_layout(title=f"Confusion Matrix - {algo_name}")
    fig.write_html(os.path.join(output_dir, "confusion_matrix.html"))

def export_evaluation_table_html(
    y_true, y_pred, y_prob, algo_name, output_dir
):
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1-score": f1_score(y_true, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, y_prob),
    }

    df_metrics = pl.DataFrame(
        {
            "Metric": list(metrics.keys()),
            "Value": [round(v, 4) for v in metrics.values()]
        }
    )

    df_metrics.write_csv(
        os.path.join(output_dir, "evaluation_metrics.csv")
    )

    df_metrics.to_pandas().to_html(
        os.path.join(output_dir, "evaluation_table.html"),
        index=False,
        float_format="%.4f"
    )

    print(
        f"[EXPORT] Evaluation table saved to "
        f"{os.path.join(output_dir, 'evaluation_table.html')}"
    )

def export_cnc_candidate_table_html(
    stats_df: pl.DataFrame,
    output_dir,
    top_k=5
):
    top_df = (
        stats_df
        .sort("cnc_ddos_score", descending=True)
        .head(top_k)
    )

    top_df.to_pandas().to_html(
        os.path.join(output_dir, "cnc_candidates.html"),
        float_format="%.4f",
        index=False
    )

def export_cnc_graph_3d_edgeweighted(
    df_all: pl.DataFrame,
    cc_nodes: list,
    algo_name: str,
    output_dir: str,
    max_nodes=400,
    max_edges=800,
    random_state=42
):
    if not cc_nodes:
        print("[WARN] No C&C nodes, skip 3D graph")
        return

    G_full = nx.DiGraph()

    for r in df_all.select(
        ["SrcAddr", "DstAddr", "EdgeWeight"]
    ).iter_rows():
        G_full.add_edge(
            r[0], r[1],
            weight=float(r[2])
        )

    nodes_keep = set(cc_nodes)
    for n in cc_nodes:
        if n in G_full:
            nodes_keep |= set(G_full.predecessors(n))
            nodes_keep |= set(G_full.successors(n))

    if len(nodes_keep) > max_nodes:
        normal_nodes = [n for n in nodes_keep if n not in cc_nodes]
        sampled = np.random.choice(
            normal_nodes,
            size=min(len(normal_nodes), max_nodes - len(cc_nodes)),
            replace=False
        )
        nodes_keep = set(cc_nodes) | set(sampled)

    G = G_full.subgraph(nodes_keep).copy()

    edges = list(G.edges())
    if len(edges) > max_edges:
        idx = np.random.choice(len(edges), max_edges, replace=False)
        G2 = nx.DiGraph()
        G2.add_nodes_from(G.nodes())
        G2.add_edges_from([edges[i] for i in idx])
        G = G2

    if G.number_of_nodes() == 0:
        print("[WARN] Graph empty after filtering")
        return

    pos = nx.spring_layout(
        G,
        dim=3,
        seed=random_state,
        iterations=60
    )

    edge_x, edge_y, edge_z = [], [], []
    for u, v in G.edges():
        x0, y0, z0 = pos[u]
        x1, y1, z1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        edge_z += [z0, z1, None]

    node_x, node_y, node_z = [], [], []
    colors, sizes, labels = [], [], []

    for n in G.nodes():
        x, y, z = pos[n]
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
        go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z,
            mode="lines",
            line=dict(color="gray", width=1),
            hoverinfo="none"
        ),
        go.Scatter3d(
            x=node_x, y=node_y, z=node_z,
            mode="markers+text",
            marker=dict(size=sizes, color=colors),
            text=labels,
            textposition="top center",
            textfont=dict(size=10)
        )
    ])

    fig.update_layout(
        title=f"Edge-Weighted 3D C&C Graph - {algo_name}",
        showlegend=False,
        margin=dict(l=0, r=0, t=40, b=0),
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
        )
    )

    out_path = os.path.join(output_dir, "graph_cnc_3d.html")
    fig.write_html(out_path)

    print(f"[EXPORT] 3D C&C graph saved -> {out_path}")

