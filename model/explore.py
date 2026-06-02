import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/yield_df.csv")

# first look
print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nFirst 5 rows:")
print(df.head())

# check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# basic stats
print("\nBasic statistics:")
print(df.describe())

# what crops are in the dataset?
print("\nUnique crops:", df["Item"].nunique())
print(df["Item"].value_counts().head(10))

# plot yield distribution
plt.figure(figsize=(10, 5))
plt.hist(df["hg/ha_yield"], bins=50, color="#1D9E75", edgecolor="white")
plt.title("Crop yield distribution")
plt.xlabel("Yield (hg/ha)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("data/yield_distribution.png")
print("\nPlot saved to data/yield_distribution.png")
