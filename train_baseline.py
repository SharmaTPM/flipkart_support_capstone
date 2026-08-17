import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report, ConfusionMatrixDisplay

# 1. Load Data
df = pd.read_csv("orders_dataset.csv")

# 2. Define Features & Target
X = df.drop(columns=["order_id", "returned"])
y = df["returned"]

# Identify column types
num_cols = ["price_inr", "discount_pct", "delivery_days", "customer_tenure_days", 
            "num_previous_orders", "num_previous_returns", "rating_given"]
cat_cols = ["product_category", "payment_method", "is_weekend_order"]

# 3. Train-Test Split (80/20 Stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# 4. Build Preprocessing Pipelines
num_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_transformer = Pipeline([
    ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", num_transformer, num_cols),
    ("cat", cat_transformer, cat_cols)
])

# 5. Build & Fit Full Pipeline
model_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(random_state=42))
])

model_pipeline.fit(X_train, y_train)

# 6. Evaluation
y_pred_proba = model_pipeline.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"Baseline Logistic Regression ROC-AUC: {roc_auc:.4f}")

# Save the fitted baseline pipeline
joblib.dump(model_pipeline, "models/baseline_logreg.joblib")
print("Baseline model saved to models/baseline_logreg.joblib")