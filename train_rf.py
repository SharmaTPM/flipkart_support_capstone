import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# 1. Load Data
df = pd.read_csv("orders_dataset.csv")

X = df.drop(columns=["order_id", "returned"])
y = df["returned"]

num_cols = ["price_inr", "discount_pct", "delivery_days", "customer_tenure_days", 
            "num_previous_orders", "num_previous_returns", "rating_given", "delivery_distance_km"]
cat_cols = ["product_category", "payment_method", "is_weekend_order"]

# 2. Stratified 80/20 Train-Test Split (random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# 3. Preprocessing Pipeline
num_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

cat_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", num_transformer, num_cols),
    ("cat", cat_transformer, cat_cols)
])

# 4. Pipeline with RandomForestClassifier(class_weight="balanced")
rf_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(class_weight="balanced", random_state=42))
])

# 5. Define Grid Search Parameters per spec
param_grid = {
    "classifier__n_estimators": [100, 200],
    "classifier__max_depth": [6, 10, None]
}

# 6. Stratified 5-Fold Cross-Validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    estimator=rf_pipeline,
    param_grid=param_grid,
    cv=cv,
    scoring="roc_auc",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

# 7. Evaluate on Held-Out Test Set
best_rf_model = grid_search.best_estimator_
y_test_probs = best_rf_model.predict_proba(X_test)[:, 1]
test_auc = roc_auc_score(y_test, y_test_probs)

print("=" * 60)
print("--- TASK 5: RANDOM FOREST GRIDSEARCHCV RESULTS ---")
print("=" * 60)
print(f"Best Parameters          : {grid_search.best_params_}")
print(f"Best Cross-Val ROC-AUC   : {grid_search.best_score_:.4f}")
print(f"Held-Out Test ROC-AUC    : {test_auc:.4f}")
print("=" * 60)