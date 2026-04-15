# =============================================================
# Part I — Exploratory Data Analysis (EDA)
# =============================================================

import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# -------------------------------------------------------------
# Output folder
# -------------------------------------------------------------
OUTPUT_DIR = "../output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def save_fig(filename):
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150, bbox_inches="tight")
    plt.close() 


# -------------------------------------------------------------
# Load Data
# -------------------------------------------------------------
df = pd.read_csv("data/final_panel.csv")


# -------------------------------------------------------------
# Add FF12 Industry Classification
# -------------------------------------------------------------
def ff12_industry(sic):
    if pd.isna(sic):
        return "Other"
    sic = int(sic)
    if 100 <= sic <= 999:
        return "Agriculture"
    elif 1000 <= sic <= 1299 or 1400 <= sic <= 1499:
        return "Mining"
    elif 1300 <= sic <= 1399:
        return "Oil"
    elif 1500 <= sic <= 1799:
        return "Construction"
    elif (2000 <= sic <= 2399 or 2700 <= sic <= 2749 or
          2770 <= sic <= 2799 or 3100 <= sic <= 3199 or
          3940 <= sic <= 3989):
        return "Consumer Goods"
    elif (2500 <= sic <= 2519 or 2590 <= sic <= 2599 or
          3630 <= sic <= 3659 or 3710 <= sic <= 3711 or
          3714 <= sic <= 3716 or 3750 <= sic <= 3751 or
          sic == 3792 or 3900 <= sic <= 3939 or
          3990 <= sic <= 3999):
        return "Consumer Durables"
    elif 2600 <= sic <= 2699 or 2750 <= sic <= 2769 or 3000 <= sic <= 3099:
        return "Manufacturing"
    elif (2800 <= sic <= 2829 or 2840 <= sic <= 2899 or
          3030 <= sic <= 3099 or 3200 <= sic <= 3569 or
          3580 <= sic <= 3629 or 3700 <= sic <= 3709 or
          3712 <= sic <= 3713 or 3715 <= sic <= 3715 or
          3717 <= sic <= 3749 or 3752 <= sic <= 3791 or
          3793 <= sic <= 3899):
        return "Chemicals"
    elif (3570 <= sic <= 3579 or 3660 <= sic <= 3692 or
          3694 <= sic <= 3699 or 7370 <= sic <= 7379):
        return "Business Equipment"
    elif 4900 <= sic <= 4999:
        return "Utilities"
    elif 5000 <= sic <= 5999 or 7200 <= sic <= 7299 or 7600 <= sic <= 7699:
        return "Shops"
    elif 2830 <= sic <= 2836 or 8000 <= sic <= 8099:
        return "Healthcare"
    elif 6000 <= sic <= 6999:
        return "Finance"
    else:
        return "Other"


if "ff12" not in df.columns:
    df["ff12"] = df["sic"].apply(ff12_industry)

df["log_size"] = np.log(df["size"])


# -------------------------------------------------------------
# 1. ESG Score Distribution
# -------------------------------------------------------------
plt.figure(figsize=(6, 4))
sns.histplot(df["esg_score"], bins=30, kde=True)
plt.title("Distribution of ESG Scores")
plt.xlabel("ESG Score")
plt.ylabel("Frequency")
save_fig("part1_fig_esg_dist.png")


# -------------------------------------------------------------
# 2. Financial Variables Distribution
# -------------------------------------------------------------
for col, label in [("roa", "ROA"), ("lev", "Leverage"), ("log_size", "Log Size")]:
    plt.figure(figsize=(6, 4))
    sns.histplot(df[col], bins=30, kde=True)
    plt.title(f"Distribution of {label}")
    plt.xlabel(label)
    plt.ylabel("Frequency")
    save_fig(f"part1_fig_{col}_dist.png")


# -------------------------------------------------------------
# 3. Industry Distribution (FF12)
# -------------------------------------------------------------
plt.figure(figsize=(8, 4))
df["ff12"].value_counts().plot(kind="bar")
plt.title("Fama-French 12 Industry Distribution")
plt.xlabel("Industry")
plt.ylabel("Number of Firms")
plt.xticks(rotation=45)
save_fig("part1_fig_industry_dist.png")


# -------------------------------------------------------------
# 4. ESG Score by Industry
# -------------------------------------------------------------
plt.figure(figsize=(10, 5))
sns.boxplot(data=df, x="ff12", y="esg_score")
plt.title("ESG Score by Industry (FF12)")
plt.xlabel("Industry")
plt.ylabel("ESG Score")
plt.xticks(rotation=45)
save_fig("part1_fig_esg_by_industry.png")


# -------------------------------------------------------------
# 5. Correlation Matrix
# -------------------------------------------------------------
corr = df[["esg_score", "roa", "lev", "log_size"]].corr()

plt.figure(figsize=(6, 4))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
save_fig("part1_fig_corr_matrix.png")


# -------------------------------------------------------------
# 6. ESG Score Over Time
# -------------------------------------------------------------
esg_time = df.groupby("fyear")["esg_score"].mean()

plt.figure(figsize=(8, 4))
esg_time.plot(marker='o')
plt.title("Average ESG Score Over Time")
plt.xlabel("Fiscal Year")
plt.ylabel("Average ESG Score")
plt.grid(True)
save_fig("part1_fig_esg_over_time.png")


# -------------------------------------------------------------
# 7. ROA Over Time
# -------------------------------------------------------------
roa_time = df.groupby("fyear")["roa"].mean()

plt.figure(figsize=(8, 4))
roa_time.plot(marker='o')
plt.title("Average ROA Over Time")
plt.xlabel("Fiscal Year")
plt.ylabel("Average ROA")
plt.grid(True)
save_fig("part1_fig_roa_over_time.png")

# -------------------------------------------------------------
# 8. ESG Score Over Time by Industry
# -------------------------------------------------------------
esg_ind_time = df.groupby(["fyear", "ff12"])["esg_score"].mean().reset_index()

plt.figure(figsize=(10, 5))
sns.lineplot(data=esg_ind_time, x="fyear", y="esg_score", hue="ff12")
plt.title("ESG Score Over Time by Industry")
plt.xlabel("Fiscal Year")
plt.ylabel("Average ESG Score")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
save_fig("part1_fig_esg_by_industry_over_time.png")
# -------------------------------------------------------------
# Summary Statistics → CSV
# -------------------------------------------------------------
summary = df[["esg_score", "roa", "lev", "log_size"]].describe().T
summary.to_csv(os.path.join(OUTPUT_DIR, "part1_summary.csv"))

print("Done. Files saved to:", OUTPUT_DIR)
print(summary)