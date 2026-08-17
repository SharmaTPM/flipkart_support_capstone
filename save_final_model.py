import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score

# 1. Load Data
df = pd.read_csv("orders_dataset.csv")
X = df.drop(columns=["order_id", "returned"])
y = df["returned"]

num_cols = ["price_inr", "discount_pct", "delivery_days", "customer_tenure_days", 
            "num_previous_orders", "num_previous_returns", "rating_given", "delivery_distance_km"]
cat_cols = ["product_category", "payment_method", "is_weekend_order"]

# 2. Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# 3. Fit Final Pipeline (Tuned Random Forest from Task 5)
preprocessor = ColumnTransformer([
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), num_cols),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore"))
    ]), cat_cols)
])

final_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=200, max_depth=10, class_weight="balanced", random_state=42))
])

final_pipeline.fit(X_train, y_train)

# 4. Perform Threshold Sweep on RF Probabilities
rf_probs = final_pipeline.predict_proba(X_test)[:, 1]
thresholds = np.arange(0.1, 0.92, 0.02)
sweep_results = []

for t in thresholds:
    preds = (rf_probs >= t).astype(int)
    p = precision_score(y_test, preds, pos_label=1, zero_division=0)
    r = recall_score(y_test, preds, pos_label=1, zero_division=0)
    f = f1_score(y_test, preds, pos_label=1, zero_division=0)
    sweep_results.append({"threshold": round(t, 2), "precision": p, "recall": r, "f1": f})

sweep_df = pd.DataFrame(sweep_results)
best_row = sweep_df.loc[sweep_df["f1"].idxmax()]
t_star_rf = best_row["threshold"]

print("=" * 60)
print("--- TASK 8: RANDOM FOREST THRESHOLD CALIBRATION (t*_rf) ---")
print("=" * 60)
print(f"Optimal RF Threshold (t*_rf) : {t_star_rf:.2f}")
print(f"RF F1-Score at t*_rf         : {best_row['f1']:.4f}")
print(f"RF Precision at t*_rf        : {best_row['precision']:.4f}")
print(f"RF Recall at t*_rf           : {best_row['recall']:.4f}")
print("=" * 60)

# 5. Persist Final Artifact
os.makedirs("models", exist_ok=True)
joblib.dump(final_pipeline, "models/return_risk_model.pkl")
print("Saved final pipeline artifact to 'models/return_risk_model.pkl'")