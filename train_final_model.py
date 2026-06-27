import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
from feature_engineering import FEATURE_COLS

CSV_FILE = "training_data_context.csv"
MODEL_FILE = "model.pkl"

df = pd.read_csv(CSV_FILE)
print("Zeilen gesamt:", len(df))
print("Instanzen:", df["instance_id"].nunique())

instance_ids = df["instance_id"].unique()
train_ids, test_ids = train_test_split(instance_ids, test_size=0.2, random_state=42)

train_df = df[df["instance_id"].isin(train_ids)]
test_df = df[df["instance_id"].isin(test_ids)]

X_train = train_df[FEATURE_COLS]
y_train = train_df["chosen"]
X_test = test_df[FEATURE_COLS]
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

print("\nFinales Modell (mit is_lpt_solution + Kontext-Features):")
print(classification_report(y_test, y_pred, digits=3))
print("ROC-AUC:", round(roc_auc_score(y_test, y_proba), 4))
print("PR-AUC:", round(average_precision_score(y_test, y_proba), 4))

joblib.dump(model, MODEL_FILE)
print(f"\nModell gespeichert als {MODEL_FILE}")