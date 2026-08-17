import pandas as pd

# Load dataset
df = pd.read_csv("orders_dataset.csv")

# 1. Total row count & overall return rate
print(f"Total Rows: {len(df)}")
print(f"Overall Return Rate: {df['returned'].mean():.4f}")

# 2. Percentage of missing rating_given values
missing_rating_pct = df["rating_given"].isna().mean() * 100
print(f"Missing rating_given: {missing_rating_pct:.2f}%\n")

# 3. Missing rate in rating_given broken down by payment_method
print("--- Missing Rating Rate by Payment Method ---")
missing_by_pay = df.groupby("payment_method")["rating_given"].apply(lambda x: x.isna().mean()).round(4)
print(missing_by_pay)
print()

# 4. Return rate by product_category
print("--- Return Rate by Product Category ---")
print(df.groupby("product_category")["returned"].mean().round(4))
print()

# 5. Return rate by payment_method
print("--- Return Rate by Payment Method ---")
print(df.groupby("payment_method")["returned"].mean().round(4))