import pandas as pd
import numpy as np

np.random.seed(42)
n = 500

df = pd.DataFrame({
    "Order Date": pd.date_range("2025-01-01", periods=n, freq="D"),
    "Category": np.random.choice(["Furniture", "Technology", "Office Supplies"], n),
    "Region": np.random.choice(["North", "South", "East", "West"], n),
    "Sales": np.random.randint(50, 2000, n),
})

df["Profit"] = (df["Sales"] * np.random.uniform(-0.1, 0.35, n)).round(2)

df.to_csv("data.csv", index=False)
print("data.csv created with", len(df), "rows")