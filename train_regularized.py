import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
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

# 3. Preprocessing Pipeline
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

# 4. Define Hyperparameter Grids
param_grid = [
    {
        "classifier__l1_ratio": [0.0, 0.2, 0.5, 0.8, 1.0], # 0.0 = Ridge (L2), 1.0 = Lasso (L1)
        "classifier__C": [0.01, 0.1, 1.0, 10.0],
        "classifier__solver": ["saga"]
    }
]

# Set penalty='elasticnet' directly on the LogisticRegression object
full_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(penalty="elasticnet", random_state=42, max_iter=1000))
])

# 5. Pipeline with GridSearchCV
full_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(random_state=42, max_iter=1000))
])

grid_search = GridSearchCV(
    full_pipeline, param_grid, cv=5, scoring="roc_auc", n_jobs=-1
)

grid_search.fit(X_train, y_train)

# 6. Evaluate Best Model
best_model = grid_search.best_estimator_
y_pred_proba = best_model.predict_proba(X_test)[:, 1]
test_auc = roc_auc_score(y_test, y_pred_proba)

print(f"Best Parameters: {grid_search.best_params_}")
print(f"Tuned Regularized ROC-AUC: {test_auc:.4f}")

# Save the best regularized model
joblib.dump(best_model, "models/regularized_logreg.joblib")