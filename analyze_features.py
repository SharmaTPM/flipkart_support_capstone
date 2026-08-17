import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

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

# 3. Fit Model Pipeline
num_transformer = Pipeline([("imputer", SimpleImputer(strategy="median"))])
cat_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", num_transformer, num_cols),
    ("cat", cat_transformer, cat_cols)
])

rf_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=200, max_depth=10, class_weight="balanced", random_state=42))
])

rf_model.fit(X_train, y_train)

# 4. Extract Feature Names after Preprocessing
ohe_feature_names = list(rf_model.named_steps["preprocessor"]
                         .named_transformers_["cat"]
                         .named_steps["encoder"]
                         .get_feature_names_out(cat_cols))
all_feature_names = num_cols + ohe_feature_names

# 5. Gini Impurity Importance
gini_importances = rf_model.named_steps["classifier"].feature_importances_
df_gini = pd.DataFrame({"feature": all_feature_names, "gini_importance": gini_importances})
df_gini = df_gini.sort_values(by="gini_importance", ascending=False).reset_index(drop=True)

# 6. Permutation Importance on Test Set
perm_result = permutation_importance(rf_model, X_test, y_test, scoring="roc_auc", n_repeats=10, random_state=42)
df_perm = pd.DataFrame({
    "feature": X.columns,
    "perm_importance_mean": perm_result.importances_mean
}).sort_values(by="perm_importance_mean", ascending=False).reset_index(drop=True)

print("=" * 65)
print("--- TOP 5 GINI IMPURITY IMPORTANCES ---")
print("=" * 65)
print(df_gini.head(5).to_string(index=False))

print("\n" + "=" * 65)
print("--- PERMUTATION IMPORTANCE (RAW FEATURES ON TEST SET) ---")
print("=" * 65)
print(df_perm.head(5).to_string(index=False))