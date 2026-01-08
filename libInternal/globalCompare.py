import os
import polars as pl


def export_global_comparison_table(output_dir):
    rows = []

    base_html = output_dir["html"]

    for algo in os.listdir(base_html):
        algo_dir = os.path.join(base_html, algo)
        metrics_file = os.path.join(algo_dir, "evaluation_metrics.csv")

        if not os.path.isfile(metrics_file):
            continue

        df = pl.read_csv(metrics_file)

        metrics = {
            row["Metric"]: row["Value"]
            for row in df.iter_rows(named=True)
        }

        rows.append({
            "Algorithm": algo,
            "Accuracy": metrics.get("Accuracy"),
            "Precision": metrics.get("Precision"),
            "Recall": metrics.get("Recall"),
            "F1-score": metrics.get("F1-score"),
            "ROC-AUC": metrics.get("ROC-AUC"),
        })

    if not rows:
        print("[GLOBAL] No evaluation metrics found")
        return

    df_all = pl.DataFrame(rows).sort(
        "F1-score",
        descending=True
    )

    out_html = os.path.join(
        output_dir["html"],
        "GLOBAL_MODEL_COMPARISON.html"
    )

    out_csv = os.path.join(
        output_dir["html"],
        "GLOBAL_MODEL_COMPARISON.csv"
    )

    df_all.write_html(
        out_html,
        float_precision=4
    )

    df_all.write_csv(out_csv)

    print("[GLOBAL] Comparison table exported")
    print(" - HTML:", out_html)
    print(" - CSV :", out_csv)
