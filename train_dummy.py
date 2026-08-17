import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

# 1. Load Data
df = pd.read_csv("orders_dataset.csv")

X = df.drop(columns=["order_id", "returned"])
y = df["returned"]

# 2. Stratified 80/20 Train-Test Split (random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# 3. Fit Baseline DummyClassifier (most_frequent strategy)
dummy = DummyClassifier(strategy="most_frequent")
dummy.fit(X_train, y_train)

# 4. Predict on Test Set
y_pred = dummy.predict(X_test)

# 5. Compute Metrics
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, pos_label=1)

print(f"--- Task 3 Baseline DummyClassifier ---")
print(f"Accuracy: {acc:.4f}")
print(f"F1-Score (returned=1): {f1:.4f}\n")
print("Full Classification Report:")
print(classification_report(y_test, y_pred, zero_division=0))
