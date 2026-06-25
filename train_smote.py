import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score

# Benötigt: pip install imbalanced-learn
from imblearn.over_sampling import SMOTE

CSV_FILE = "training_data_large.csv"  # ggf. anpassen, falls du training_data.csv weiter nutzt

df = pd.read_csv(CSV_FILE)
print("Zeilen gesamt:", len(df))
print("Instanzen:", df["instance_id"].nunique())
print("Anteil chosen=1:", df["chosen"].mean())

# Fokus: NUR reine Zufalls-Spalten (der schwere Fall von vorhin)
df_random_only = df[df["is_lpt_solution"] == 0].copy()
print(f"\nNur Zufalls-Spalten: {len(df_random_only)} Zeilen, "
      f"chosen=1: {df_random_only['chosen'].sum()} ({100*df_random_only['chosen'].mean():.4f}%)")

feature_cols = [
    "n_items", "n_locations", "Q",
    "n_pods", "n_items_covered",
    "workload", "distance",
    "distance_per_item",
    "workload_rank_in_instance",
    "distance_rank_in_instance",
]

instance_ids = df_random_only["instance_id"].unique()
train_ids, test_ids = train_test_split(instance_ids, test_size=0.2, random_state=42)

train_df = df_random_only[df_random_only["instance_id"].isin(train_ids)]
test_df = df_random_only[df_random_only["instance_id"].isin(test_ids)]

X_train = train_df[feature_cols]
y_train = train_df["chosen"]
X_test = test_df[feature_cols]
y_test = test_df["chosen"]

print(f"\nTrain vor SMOTE: {len(X_train)} Zeilen, {y_train.sum()} positiv")

smote = SMOTE(sampling_strategy=0.1, random_state=42, k_neighbors=5)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print(f"Train nach SMOTE: {len(X_train_res)} Zeilen, {y_train_res.sum()} positiv")

model = RandomForestClassifier(
    n_estimators=300, max_depth=12,
    class_weight="balanced", random_state=42, n_jobs=-1,
)
model.fit(X_train_res, y_train_res)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("\n=== Modell MIT SMOTE-Oversampling (nur Zufalls-Spalten) ===")
print(classification_report(y_test, y_pred, digits=3))
print("ROC-AUC:", round(roc_auc_score(y_test, y_proba), 4))
print("PR-AUC:", round(average_precision_score(y_test, y_proba), 4))

importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\nFeature Importance:")
print(importances)

model_baseline = RandomForestClassifier(
    n_estimators=300, max_depth=12,
    class_weight="balanced", random_state=42, n_jobs=-1,
)
model_baseline.fit(X_train, y_train)
y_proba_baseline = model_baseline.predict_proba(X_test)[:, 1]

print("\n=== Referenz OHNE SMOTE (nur class_weight='balanced') ===")
print("ROC-AUC:", round(roc_auc_score(y_test, y_proba_baseline), 4))
print("PR-AUC:", round(average_precision_score(y_test, y_proba_baseline), 4))
