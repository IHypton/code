import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
)

df = pd.read_csv("training_data.csv")

print("Anzahl Zeilen (Spalten über alle Instanzen):", len(df))
print("Anzahl Instanzen:", df["instance_id"].nunique())
print("Anteil 'chosen=1':", df["chosen"].mean())

instance_ids = df["instance_id"].unique()
train_ids, test_ids = train_test_split(instance_ids, test_size=0.2, random_state=42)

train_df = df[df["instance_id"].isin(train_ids)]
test_df = df[df["instance_id"].isin(test_ids)]

print(f"\nTrain: {len(train_df)} Zeilen aus {len(train_ids)} Instanzen")
print(f"Test:  {len(test_df)} Zeilen aus {len(test_ids)} Instanzen")

feature_cols = [
    "n_items", "n_locations", "Q",
    "n_pods", "n_items_covered",
    "workload", "distance",
    "distance_per_item",
    "workload_rank_in_instance",
    "distance_rank_in_instance",
    "is_lpt_solution",
]

X_train = train_df[feature_cols]
y_train = train_df["chosen"]
X_test = test_df[feature_cols]
y_test = test_df["chosen"]

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("\nKlassifikationsbericht (Schwelle 0.5):")
print(classification_report(y_test, y_pred, digits=3))

print("ROC-AUC:", round(roc_auc_score(y_test, y_proba), 4))
print("PR-AUC (Average Precision):", round(average_precision_score(y_test, y_proba), 4))

importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\n Feature Importance:")
print(importances)

test_df = test_df.copy()
test_df["model_score"] = y_proba

print("\n Beispiel: Top-5 Spalten nach Modell-Score für eine Test-Instanz:")
example_instance = test_ids[0]
example = test_df[test_df["instance_id"] == example_instance].sort_values("model_score", ascending=False)
print(example[["workload", "distance", "n_items_covered", "model_score", "chosen"]].head(10))
