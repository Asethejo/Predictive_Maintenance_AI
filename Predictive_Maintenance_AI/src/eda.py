
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load official AI4I dataset
df = pd.read_csv("data/ai4i2020.csv")

print("Dataset loaded successfully!")
print("Dataset Shape:", df.shape)

# 1. Machine Failure Distribution
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="Machine failure")
plt.title("Machine Failure Distribution")
plt.xlabel("Machine Failure (0 = No, 1 = Yes)")
plt.ylabel("Number of Machines")
plt.tight_layout()
plt.show()

# 2. Machine Type Distribution
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="Type")
plt.title("Machine Type Distribution")
plt.xlabel("Machine Type")
plt.ylabel("Number of Machines")
plt.tight_layout()
plt.show()

# 3. Air Temperature vs Machine Failure
plt.figure(figsize=(7, 5))
sns.boxplot(data=df, x="Machine failure", y="Air temperature [K]")
plt.title("Air Temperature vs Machine Failure")
plt.xlabel("Machine Failure")
plt.ylabel("Air Temperature (K)")
plt.tight_layout()
plt.show()

# 4. Rotational Speed vs Machine Failure
plt.figure(figsize=(7, 5))
sns.boxplot(data=df, x="Machine failure", y="Rotational speed [rpm]")
plt.title("Rotational Speed vs Machine Failure")
plt.xlabel("Machine Failure")
plt.ylabel("Rotational Speed (rpm)")
plt.tight_layout()
plt.show()

# 5. Correlation Heatmap
numeric_df = df.select_dtypes(include="number")

plt.figure(figsize=(12, 8))
sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()

print("EDA completed successfully!")
