import duckdb
from polars.selectors import numeric
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
import polars as pl
import plotly.figure_factory as ff

csv_path = "dataset/NCC2AllSensors_clean_dir_encoded.csv"
df_pl = pl.read_csv(csv_path)
print("[debug --] Columns:", df_pl.columns)

con = duckdb.connect()
con.register("flows", df_pl)
query = """
SELECT 
    SrcAddr, DstAddr, Proto, Dir, Dir_encode, State, Dur, TotBytes, TotPkts,
    sTos, dTos, SrcBytes, Label, label_as_bot, SensorId
FROM flows
WHERE label_as_bot IS NOT NULL
  AND REGEXP_MATCHES(SrcAddr, '^[0-9.]+$')
  AND SensorId = 3
"""

df_filtered = con.execute(query).pl()
print("[debug --] Filtered DF shape:", df_filtered.shape)
print(df_filtered.head())

LABEL_COL = "label_as_bot"
features_to_drop = [
    "Label",
    "label_as_bot",
    "Dir",
    "SrcAddr",
    "DstAddr",
    "SensorId",
    "Proto",
    "State"
]

df_filtered = df_filtered.drop_nulls()

X_pl = df_filtered.drop(features_to_drop).select(numeric())
y_pl = df_filtered.select(LABEL_COL)

X = X_pl.to_numpy()
y = y_pl.to_numpy().ravel()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

clf = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(
        max_depth=1,
        random_state=42
    ),
    n_estimators=200,
    learning_rate=0.5,
    random_state=42
)

print("[debug --] training adaboost ========================")
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("\n accuracy:", accuracy_score(y_test, y_pred))
print("\n report:")
print(classification_report(y_test, y_pred))

labels = clf.classes_.astype(str).tolist()
cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)

cm_pl = pl.DataFrame(
    cm,
    schema=labels
).with_columns([
    pl.Series("actual", labels)
]).select(["actual", *labels])

print("\n confusion matrix (polars df):")
print(cm_pl)

# Plotly Confusion Matrix (HTML)
z = cm.tolist()

fig = ff.create_annotated_heatmap(
    z,
    x=labels,
    y=labels,
    colorscale='Blues',
    showscale=True
)

fig.update_layout(
    title="confusion matrix (AdaBoost)",
    xaxis_title="predicted",
    yaxis_title="actual"
)

fig.write_html("confusion_matrix_adaboost.html")
print("\n saved: confusion_matrix_adaboost.html")
