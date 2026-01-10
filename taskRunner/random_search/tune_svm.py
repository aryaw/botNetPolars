import duckdb
import numpy as np
import polars as pl

from sklearn.model_selection import train_test_split, ParameterSampler
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
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

RANDOM_STATE = 42
MAX_TUNE_SAMPLES = 120_000
TOP_K_INFECTED = 5
TOP_K_CNC = 5


def tune_svm(
    csv_path="dataset/ncc/NCC2AllSensors_clean.csv",
    n_iter=10
):
    con = duckdb.connect()
    df = pl.from_arrow(con.sql(f"""
        SELECT SrcAddr, DstAddr, Proto, Dir, State,
               Dur, TotBytes, TotPkts, sTos, dTos, SrcBytes,
               Label, SensorId
        FROM read_csv_auto('{csv_path}')
        WHERE Label IS NOT NULL
          AND REGEXP_MATCHES(SrcAddr,'^[0-9.]+$')
          AND SensorId = 3
    """).arrow())

    df = fast_label_as_bot_to_binary(df)
    df = df.drop_nulls([
        "SrcAddr","DstAddr","Dir","Proto","Dur","TotBytes","TotPkts"
    ])

    df = df.with_columns(
        pl.col("Dir").cast(pl.Utf8)
        .replace({"->":1,"<-":-1,"<->":0}, default=0)
        .cast(pl.Int32)
    )

    for c in ["Proto","State"]:
        df = df.with_columns(
            pl.Series(c, LabelEncoder().fit_transform(df[c].to_list()))
        )

    y = df["label_as_bot"].to_numpy()
    idx_train, idx_test = train_test_split(
        np.arange(len(df)), test_size=0.30, stratify=y, random_state=RANDOM_STATE
    )

    df_train, df_test = df[idx_train], df[idx_test]

    edge_w = df_train.group_by(["SrcAddr","DstAddr"]).count().rename({"count":"EdgeWeight"})
    for name in ["df_train","df_test"]:
        locals()[name] = locals()[name].join(
            edge_w, on=["SrcAddr","DstAddr"], how="left"
        ).with_columns(pl.col("EdgeWeight").fill_null(1))

    src_total = df_train.group_by("SrcAddr").agg(
        pl.col("EdgeWeight").sum().alias("SrcTotalWeight")
    )
    dst_total = df_train.group_by("DstAddr").agg(
        pl.col("EdgeWeight").sum().alias("DstTotalWeight")
    )

    for name in ["df_train","df_test"]:
        locals()[name] = locals()[name] \
            .join(src_total, on="SrcAddr", how="left") \
            .join(dst_total, on="DstAddr", how="left") \
            .fill_null(0)

    features = [
        "Dir","Dur","Proto","TotBytes","TotPkts",
        "sTos","dTos","SrcBytes",
        "EdgeWeight","SrcTotalWeight","DstTotalWeight"
    ]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(df_train.select(features).to_numpy())
    X_test = scaler.transform(df_test.select(features).to_numpy())
    y_train = df_train["label_as_bot"].to_numpy()
    y_test = df_test["label_as_bot"].to_numpy()

    if len(X_train) > MAX_TUNE_SAMPLES:
        idx = np.random.choice(len(X_train), MAX_TUNE_SAMPLES, replace=False)
        X_train_tune, y_train_tune = X_train[idx], y_train[idx]
    else:
        X_train_tune, y_train_tune = X_train, y_train

    param_dist = {
        "C": np.logspace(-2, np.log10(5.0), 30),
        "gamma": np.logspace(-4, np.log10(5e-2), 30)
    }

    best_score, best_params = -1, None

    for params in ParameterSampler(
        param_dist, n_iter=n_iter, random_state=RANDOM_STATE
    ):
        model = SVC(
            **params,
            kernel="rbf",
            probability=True,
            class_weight="balanced",
            random_state=RANDOM_STATE
        )
        model.fit(X_train_tune, y_train_tune)
        p = model.predict_proba(X_test)[:,1]
        fpr,tpr,thr = roc_curve(y_test,p)
        score = f1_score(
            y_test, (p >= thr[np.argmax(tpr-fpr)]).astype(int)
        )
        if score > best_score:
            best_score, best_params = score, params

    model = SVC(
        **best_params,
        kernel="rbf",
        probability=True,
        class_weight="balanced",
        random_state=RANDOM_STATE
    )
    model.fit(X_train, y_train)

    return {"best_params": best_params}
