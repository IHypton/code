import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score

df = pd.read_csv("training_data.csv")

# ==================================================
# ANALYSE 1: Wie oft schlägt eine Zufalls-Spalte die LPT-Heuristik?
# ==================================================

chosen = df[df["chosen"] == 1]
chosen_random = chosen[chosen["is_lpt_solution"] == 0]

print(f"Gewählte Spalten insgesamt: {len(chosen)}")
print(f"Davon Zufalls-Spalten (schlagen die Heuristik): {len(chosen_random)} "
      f"({100*len(chosen_random)/len(chosen):.1f}%)")

print("\n--- Wie viele Instanzen hatten mind. 1 Zufalls-Spalte in der Lösung? ---")
instances_with_random_win = chosen_random["instance_id"].nunique()
print(f"{instances_with_random_win} von {df['instance_id'].nunique()} Instanzen")

print("\n--- Eigenschaften der Zufalls-Spalten, die die Heuristik schlagen ---")
print(chosen_random[["workload", "distance", "n_items_covered",
                      "workload_rank_in_instance", "distance_rank_in_instance"]].describe())

print("\n--- Im Vergleich: Eigenschaften der gewählten LPT-Spalten ---")
chosen_lpt = chosen[chosen["is_lpt_solution"] == 1]
print(chosen_lpt[["workload", "distance", "n_items_covered",
                   "workload_rank_in_instance", "distance_rank_in_instance"]].describe())

instance_ids = df["instance_id"].unique()
train_ids, test_ids = train_test_split(instance_ids, test_size=0.2, random_state=42)

train_df = df[df["instance_id"].isin(train_ids)]
test_df = df[df["instance_id"].isin(test_ids)]

feature_cols_no_lpt = [
    "n_items", "n_locations", "Q",
    "n_pods", "n_items_covered",
    "workload", "distance",
    "distance_per_item",
    "workload_rank_in_instance",
    "distance_rank_in_instance",
]

X_train = train_df[feature_cols_no_lpt]
y_train = train_df["chosen"]
X_test = test_df[feature_cols_no_lpt]
y_test = test_df["chosen"]

model = RandomForestClassifier(
    n_estimators=300, max_depth=12,
    class_weight="balanced", random_state=42, n_jobs=-1,
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("\n\n=== MODELL OHNE is_lpt_solution ===")
print(classification_report(y_test, y_pred, digits=3))
print("ROC-AUC:", round(roc_auc_score(y_test, y_proba), 4))
print("PR-AUC:", round(average_precision_score(y_test, y_proba), 4))

importances = pd.Series(model.feature_importances_, index=feature_cols_no_lpt).sort_values(ascending=False)
print("\nFeature Importance (ohne LPT-Feature):")
print(importances)

df_random_only = df[df["is_lpt_solution"] == 0]
print(f"\n\n=== NUR ZUFALLS-SPALTEN (ohne jede LPT-Spalte) ===")
print(f"Zeilen: {len(df_random_only)}, davon chosen=1: {df_random_only['chosen'].sum()} "
      f"({100*df_random_only['chosen'].mean():.3f}%)")

train_df2 = df_random_only[df_random_only["instance_id"].isin(train_ids)]
test_df2 = df_random_only[df_random_only["instance_id"].isin(test_ids)]

X_train2 = train_df2[feature_cols_no_lpt]
y_train2 = train_df2["chosen"]
X_test2 = test_df2[feature_cols_no_lpt]
y_test2 = test_df2["chosen"]

model2 = RandomForestClassifier(
    n_estimators=300, max_depth=12,
    class_weight="balanced", random_state=42, n_jobs=-1,
)
model2.fit(X_train2, y_train2)

y_pred2 = model2.predict(X_test2)
y_proba2 = model2.predict_proba(X_test2)[:, 1]

print(classification_report(y_test2, y_pred2, digits=3))
print("ROC-AUC:", round(roc_auc_score(y_test2, y_proba2), 4))
print("PR-AUC:", round(average_precision_score(y_test2, y_proba2), 4))
