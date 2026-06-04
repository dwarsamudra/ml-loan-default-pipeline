"""
notebooks/eda.py  — Exploratory Data Analysis
Author: Anvesh Dubey
Run this as a script OR open in Jupyter as a .py notebook (percent format).
"""

# %% [markdown]
# # Exploratory Data Analysis — Loan Default Dataset
# **Author:** Anvesh Dubey
# This notebook explores the synthetic loan dataset, checks distributions,
# correlations, and class balance before training.

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_theme(style="whitegrid", palette="Blues_r")
os.makedirs("../static", exist_ok=True)

# %% [markdown]
# ## 1. Load Data

# %%
df = pd.read_csv("../data/loan_data.csv")
print(f"Shape: {df.shape}")
print(df.head())

# %% [markdown]
# ## 2. Basic Info & Missing Values

# %%
print("\nData Types:\n", df.dtypes)
print("\nMissing Values:\n", df.isnull().sum())
print("\nClass Balance:\n", df["default"].value_counts(normalize=True))

# %% [markdown]
# ## 3. Numeric Distributions

# %%
num_cols = ["age", "income", "loan_amount", "credit_score", "months_employed", "num_prev_loans"]
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
for ax, col in zip(axes.flatten(), num_cols):
    sns.histplot(df[col], kde=True, ax=ax, color="#185FA5")
    ax.set_title(col)
plt.suptitle("Numeric Feature Distributions", fontweight="bold")
plt.tight_layout()
plt.savefig("../static/eda_distributions.png", dpi=120)
plt.show()
print("Saved → static/eda_distributions.png")

# %% [markdown]
# ## 4. Target vs Features

# %%
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
for ax, col in zip(axes.flatten(), num_cols):
    sns.boxplot(x="default", y=col, data=df, ax=ax,
                palette=["#185FA5", "#E24B4A"])
    ax.set_xticklabels(["No Default", "Default"])
    ax.set_title(f"{col} by Default")
plt.suptitle("Feature Distribution by Default Label", fontweight="bold")
plt.tight_layout()
plt.savefig("../static/eda_boxplots.png", dpi=120)
plt.show()

# %% [markdown]
# ## 5. Correlation Heatmap

# %%
plt.figure(figsize=(9, 7))
corr = df[num_cols + ["default"]].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
            cmap="Blues", linewidths=0.5)
plt.title("Correlation Matrix", fontweight="bold")
plt.tight_layout()
plt.savefig("../static/eda_correlation.png", dpi=120)
plt.show()

# %% [markdown]
# ## 6. Categorical Features

# %%
cat_cols = ["employment_status", "education", "loan_purpose"]
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
for ax, col in zip(axes, cat_cols):
    rate = df.groupby(col)["default"].mean().sort_values(ascending=False)
    sns.barplot(x=rate.index, y=rate.values, ax=ax, palette="Blues_r")
    ax.set_title(f"Default Rate by {col}")
    ax.set_ylabel("Default Rate")
    ax.set_ylim(0, 0.6)
plt.suptitle("Default Rate by Category", fontweight="bold")
plt.tight_layout()
plt.savefig("../static/eda_categorical.png", dpi=120)
plt.show()

print("\nEDA complete. All plots saved to static/")
