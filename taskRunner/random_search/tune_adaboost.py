import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import gc
import random
import duckdb
import numpy as np
import polars as pl

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import f1_score, roc_curve

from libInternal.dFHelper import (
    fast_label_as_bot_to_binary,
    setFileLocation,
    get_algo_output_dir,
    export_confusion_matrix_html,
    export_evaluation_table_html,
    export_cnc_candidate_table_html,
    export_cnc_graph_3d_edgeweighted
)
from libInternal.slideHelper import generate_algo_slide


# =========================
# CONFIG
# =========================
RANDOM_STATE = 42
MAX_TUNE_SAMPLES = 200_000
N_TRIALS = 20
TOP_K_INFECTED = 5
TOP_K_CNC = 5


def tune_adaboost(
    csv_path="dataset/ncc/NCC2AllSensors_clean.csv"
):
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

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

    # ===== LABEL =====
    df = fast_label_as_bot_to_binary(df)

    df = df.drop_nulls([
        "SrcAddr","DstAddr","Dir","Proto",
        "Dur","TotBytes","TotPkts"
    ])

    df = df.with_columns(
        pl.col("Dir")
        .cast(pl.Utf8)
        .replace({"->": 1, "<-": -1, "<->": 0}, default=0)
        .cast(pl.Int32)
    )

    for c in ["Proto", "State"]:
        df = df.with_columns(
            pl.Series(c, LabelEncoder().fit_transform(df[c].to_list()))
        )

    # ===== SPLIT =====
    y = df["label_as_bot"].to_numpy(copy=True)

    idx_train, idx_test = train_test_split(
        np.arange(len(df)),
        test_size=0.3,
        stratify=y,
        random_state=RANDOM_STATE
    )

    df_train = df[idx_train]
    df_test = df[idx_test]

    # ===== EDGE FEATURES =====
    edge_w = (
        df_train.group_by(["SrcAddr", "DstAddr"])
        .count()
        .rename({"count": "EdgeWeight"})
    )

    df_train = df_train.join(edge_w, on=["SrcAddr","DstAddr"], how="left") \
                       .with_columns(pl.col("EdgeWeight").fill_null(1))
    df_test = df_test.join(edge_w, on=["SrcAddr","DstAddr"], how="left") \
                     .with_columns(pl.col("EdgeWeight").fill_null(1))

    src_total = df_train.group_by("SrcAddr").agg(
        pl.col("EdgeWeight").sum().alias("SrcTotalWeight")
    )
    dst_total = df_train.group_by("DstAddr").agg(
        pl.col("EdgeWeight").sum().alias("DstTotalWeight")
    )

    df_train = df_train.join(src_total, on="SrcAddr", how="left") \
                       .join(dst_total, on="DstAddr", how="left") \
                       .fill_null(0)
    df_test = df_test.join(src_total, on="SrcAddr", how="left") \
                      .join(dst_total, on="DstAddr", how="left") \
                      .fill_null(0)

    features = [
        "Dir","Dur","Proto","TotBytes","TotPkts",
        "sTos","dTos","SrcBytes",
        "EdgeWeight","SrcTotalWeight","DstTotalWeight"
    ]

    # ===== TUNING SUBSET =====
    if len(df_train) > MAX_TUNE_SAMPLES:
        idx = np.random.choice(len(df_train), MAX_TUNE_SAMPLES, replace=False)
        df_train_tune = df_train[idx]
    else:
        df_train_tune = df_train

    X_train_tune = df_train_tune.select(features).to_numpy(copy=True)
    y_train_tune = df_train_tune["label_as_bot"].to_numpy(copy=True)
    X_test = df_test.select(features).to_numpy(copy=True)
    y_test = df_test["label_as_bot"].to_numpy(copy=True)

    # ===== RANDOM SEARCH =====
    best_score = -1.0
    best_params = None

    for _ in range(N_TRIALS):
        params = {
            "n_estimators": random.randint(100, 200),
            "learning_rate": 10 ** random.uniform(-2, -0.3)
        }

        model = AdaBoostClassifier(
            **params,
            random_state=RANDOM_STATE
        )

        model.fit(X_train_tune, y_train_tune)

        p = model.predict_proba(X_test)[:, 1]
        fpr, tpr, thr = roc_curve(y_test, p)

        score = f1_score(
            y_test,
            (p >= thr[np.argmax(tpr - fpr)]).astype(int)
        )

        if score > best_score:
            best_score = score
            best_params = params

        del model
        gc.collect()

    if best_params is None:
        raise RuntimeError("Random search failed to find valid parameters.")

    # ===== FINAL MODEL =====
    model = AdaBoostClassifier(
        **best_params,
        random_state=RANDOM_STATE
    )

    model.fit(
        df_train.select(features).to_numpy(copy=True),
        df_train["label_as_bot"].to_numpy(copy=True)
    )

    p = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thr = roc_curve(y_test, p)
    best_thr = thr[np.argmax(tpr - fpr)]
    y_pred = (p >= best_thr).astype(int)

    # ===== APPLY TO ALL =====
    df_all = pl.concat([df_train, df_test])
    X_all = df_all.select(features).to_numpy(copy=True)

    df_all = df_all.with_columns(
        pl.Series("PredictedProb", model.predict_proba(X_all)[:, 1])
    ).with_columns(
        (pl.col("PredictedProb") >= best_thr)
        .cast(pl.Int8)
        .alias("PredictedLabel")
    )

    # ===== CNC / INFECTION =====
    stats_out = df_all.group_by("SrcAddr").agg(
        out_ct=pl.count(),
        bot_out=pl.col("PredictedLabel").sum(),
        out_prob=pl.col("PredictedProb").mean(),
        src_weight=pl.col("SrcTotalWeight").mean(),
        uniq_dst=pl.col("DstAddr").n_unique()
    )

    infected = stats_out.with_columns(
        (
            (pl.col("bot_out") / (pl.col("out_ct") + 1e-9))
            * pl.col("out_prob")
            * pl.col("uniq_dst").log1p()
            * pl.col("src_weight").log1p()
        ).alias("infection_score")
    ).sort("infection_score", descending=True)

    stats_in = df_all.group_by("DstAddr").agg(
        in_ct=pl.count(),
        in_prob=pl.col("PredictedProb").mean(),
        dst_weight=pl.col("DstTotalWeight").mean()
    )

    cnc = stats_in.join(
        stats_out,
        left_on="DstAddr",
        right_on="SrcAddr",
        how="outer"
    ).fill_null(0).with_columns(
        degree=pl.col("in_ct") + pl.col("out_ct"),
        out_ratio=pl.col("out_ct") / (pl.col("in_ct") + pl.col("out_ct") + 1e-9),
        avg_prob=(pl.col("in_prob") + pl.col("out_prob")) / 2
    ).with_columns(
        (
            pl.col("avg_prob")
            * pl.col("degree").log1p()
            * pl.col("out_ratio")
            * (pl.col("src_weight") + pl.col("dst_weight")).log1p()
        ).alias("cnc_ddos_score")
    ).sort("cnc_ddos_score", descending=True)

    infected_nodes = infected.head(TOP_K_INFECTED)["SrcAddr"].to_list()
    cnc_nodes = cnc.head(TOP_K_CNC)["DstAddr"].to_list()

    # ===== EXPORT =====
    ts, out = setFileLocation()
    algo_dir = get_algo_output_dir(out, "AdaBoost", "random_search")

    export_confusion_matrix_html(y_test, y_pred, "AdaBoost (Random Search)", algo_dir)
    export_evaluation_table_html(y_test, y_pred, p, "AdaBoost (Random Search)", algo_dir)
    export_cnc_candidate_table_html(cnc, algo_dir, TOP_K_CNC)
    export_cnc_graph_3d_edgeweighted(
        df_all, cnc_nodes, "AdaBoost (Random Search)", algo_dir
    )

    generate_algo_slide("AdaBoost (Random Search)", algo_dir)

    return {
        "best_params": best_params,
        "infected_hosts": infected_nodes,
        "cnc_hosts": cnc_nodes
    }
