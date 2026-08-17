import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

# 1. Load Data
df = pd.read_csv("orders_dataset.csv")

X = df.drop(columns=["order_id", "returned"])
y = df["returned"]

num_cols = ["price_inr", "discount_pct", "delivery_days", "customer_tenure_days", 
            "num_previous_orders", "num_previous_returns", "rating_given"]
cat_cols = ["product_category", "payment_method", "is_weekend_order"]

# 2. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Tree models don't require feature scaling, only imputation & encoding
num_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

cat_transformer = Pipeline([
    ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", num_transformer, num_cols),
    ("cat", cat_transformer, cat_cols)
])

# 3. Train & Evaluate Random Forest
rf_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42))
])

rf_pipeline.fit(X_train, y_train)
rf_auc = roc_auc_score(y_test, rf_pipeline.predict_proba(X_test)[:, 1])
print(f"Random Forest ROC-AUC: {rf_auc:.4f}")

# 4. Train & Evaluate XGBoost
xgb_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.05, random_state=42))
])

xgb_pipeline.fit(X_train, y_train)
xgb_auc = roc_auc_score(y_test, xgb_pipeline.predict_proba(X_test)[:, 1])
print(f"XGBoost ROC-AUC: {xgb_auc:.4f}")

# Save the better performing tree model
if xgb_auc >= rf_auc:
    joblib.dump(xgb_pipeline, "models/best_tree_model.joblib")
    print("Saved XGBoost to models/best_tree_model.joblib")
else:
    joblib.dump(rf_pipeline, "models/best_tree_model.joblib")
    print("Saved Random Forest to models/best_tree_model.joblib")