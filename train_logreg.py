import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# 1. Load Data
df = pd.read_csv("orders_dataset.csv")

X = df.drop(columns=["order_id", "returned"])
y = df["returned"]

num_cols = ["price_inr", "discount_pct", "delivery_days", "customer_tenure_days", 
            "num_previous_orders", "num_previous_returns", "rating_given", "delivery_distance_km"]
cat_cols = ["product_category", "payment_method", "is_weekend_order"]

# 2. Train-Test Split (80/20 Stratified, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# 3. Preprocessing Pipeline without data leakage
num_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", num_transformer, num_cols),
    ("cat", cat_transformer, cat_cols)
])

# 4. Train Balanced Logistic Regression
model_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000))
])

model_pipeline.fit(X_train, y_train)

# 5. Evaluate at Default 0.5 Threshold
y_probs = model_pipeline.predict_proba(X_test)[:, 1]
y_pred_default = (y_probs >= 0.5).astype(int)

acc_def = accuracy_score(y_test, y_pred_default)
prec_def = precision_score(y_test, y_pred_default, pos_label=1)
rec_def = recall_score(y_test, y_pred_default, pos_label=1)
f1_def = f1_score(y_test, y_pred_default, pos_label=1)
auc_def = roc_auc_score(y_test, y_probs)

print("=" * 60)
print("--- TASK 4: LOGISTIC REGRESSION (DEFAULT 0.5 THRESHOLD) ---")
print("=" * 60)
print(f"Accuracy : {acc_def:.4f}")
print(f"Precision: {prec_def:.4f}")
print(f"Recall   : {rec_def:.4f}")
print(f"F1-Score : {f1_def:.4f}")
print(f"ROC-AUC  : {auc_def:.4f}\n")

# 6. Decision Threshold Sweep (0.1 to 0.9 in steps of 0.02)
thresholds = np.arange(0.1, 0.92, 0.02)
sweep_results = []

for t in thresholds:
    preds = (y_probs >= t).astype(int)
    p = precision_score(y_test, preds, pos_label=1, zero_division=0)
    r = recall_score(y_test, preds, pos_label=1, zero_division=0)
    f = f1_score(y_test, preds, pos_label=1, zero_division=0)
    acc = accuracy_score(y_test, preds)
    sweep_results.append({"threshold": round(t, 2), "accuracy": acc, "precision": p, "recall": r, "f1": f})

sweep_df = pd.DataFrame(sweep_results)

# Find optimal threshold that maximizes F1
best_row = sweep_df.loc[sweep_df["f1"].idxmax()]
t_star = best_row["threshold"]

print("=" * 60)
print("--- F1-THRESHOLD SWEEP SUMMARY (SAMPLE STEPS) ---")
print("=" * 60)
print(sweep_df.to_string(index=False))

print("\n" + "=" * 60)
print(f"--- OPTIMAL F1 THRESHOLD (t*) ---")
print("=" * 60)
print(f"Optimal Threshold (t*) : {t_star:.2f}")
print(f"F1-Score at t*         : {best_row['f1']:.4f}")
print(f"Precision at t*        : {best_row['precision']:.4f}")
print(f"Recall at t*           : {best_row['recall']:.4f}")
print("=" * 60)