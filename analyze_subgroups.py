import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score

# 1. Load & Split Data
df = pd.read_csv("orders_dataset.csv")
X = df.drop(columns=["order_id", "returned"])
y = df["returned"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# 2. Fit RF Model
num_cols = ["price_inr", "discount_pct", "delivery_days", "customer_tenure_days", 
            "num_previous_orders", "num_previous_returns", "rating_given", "delivery_distance_km"]
cat_cols = ["product_category", "payment_method", "is_weekend_order"]

preprocessor = ColumnTransformer([
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), num_cols),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore"))
    ]), cat_cols)
])

rf_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=200, max_depth=10, class_weight="balanced", random_state=42))
])

rf_pipeline.fit(X_train, y_train)

# 3. Test Set Predictions
test_df = X_test.copy()
test_df["actual_returned"] = y_test
test_df["pred_returned"] = rf_pipeline.predict(X_test)

# Helper function for subgroup breakout
def get_subgroup_metrics(group_col):
    results = []
    for val, group in test_df.groupby(group_col):
        p = precision_score(group["actual_returned"], group["pred_returned"], pos_label=1, zero_division=0)
        r = recall_score(group["actual_returned"], group["pred_returned"], pos_label=1, zero_division=0)
        count = len(group)
        returns = group["actual_returned"].sum()
        results.append({group_col: val, "count": count, "returns": returns, "precision": round(p, 4), "recall": round(r, 4)})
    return pd.DataFrame(results)

print("=" * 60)
print("--- SUBGROUP METRICS BY PRODUCT CATEGORY ---")
print("=" * 60)
print(get_subgroup_metrics("product_category").to_string(index=False))

print("\n" + "=" * 60)
print("--- SUBGROUP METRICS BY PAYMENT METHOD ---")
print("=" * 60)
print(get_subgroup_metrics("payment_method").to_string(index=False))