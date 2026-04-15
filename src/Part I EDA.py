# Part I — Exploratory Data Analysis (EDA)

## Overview

We conduct EDA by visualizing the distribution of ESG scores and key financial variables (ROA, leverage, and firm size), examining industry composition using Fama-French classifications, comparing ESG across industries, and analyzing correlations among variables. These steps help us understand the data structure and justify the inclusion of control variables and industry fixed effects in the regression analysis.

---

## Prerequisites

```python
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv("final_panel.csv")
```

> The `ff12` column (Fama-French 12 industry classification) must already be added to `df` before running the plots below.

---

## 1. ESG Score Distribution

Plot the distribution of ESG scores to examine overall variation and detect skewness/outliers.

```python
plt.figure(figsize=(6, 4))
sns.histplot(df["esg_score"], bins=30, kde=True)
plt.title("Distribution of ESG Scores")
plt.xlabel("ESG Score")
plt.ylabel("Frequency")
plt.show()
```

---

## 2. Financial Variables Distribution

Visualize distributions of key financial variables (ROA, leverage, size) to understand their shape and identify skewness or extreme values.

```python
# Log transform size (commonly used in finance)
df["log_size"] = np.log(df["size"])

for col in ["roa", "lev", "log_size"]:
    plt.figure(figsize=(6, 4))
    sns.histplot(df[col], bins=30, kde=True)
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.show()
```

---

## 3. Industry Distribution (Fama-French 12)

Display the number of firms in each Fama-French 12 industry to understand sample composition across sectors.

```python
plt.figure(figsize=(8, 4))
df["ff12"].value_counts().plot(kind="bar")
plt.title("Fama-French 12 Industry Distribution")
plt.xlabel("Industry")
plt.ylabel("Number of Firms")
plt.xticks(rotation=45)
plt.show()
```

---

## 4. ESG Score by Industry

Compare ESG score distributions across industries to identify cross-industry differences (important for regression controls).

```python
plt.figure(figsize=(10, 5))
sns.boxplot(data=df, x="ff12", y="esg_score")
plt.title("ESG Score by Industry (FF12)")
plt.xlabel("Industry")
plt.ylabel("ESG Score")
plt.xticks(rotation=45)
plt.show()
```

---

## 5. Correlation Matrix

Compute and visualize correlations between ESG and key financial variables to assess relationships and potential multicollinearity.

```python
corr = df[["esg_score", "roa", "lev", "log_size"]].corr()

plt.figure(figsize=(6, 4))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.show()
```

---

## 6. ESG Score Over Time

Plot average ESG score over time to examine temporal trends.

```python
esg_time = df.groupby("year")["esg_score"].mean()

plt.figure(figsize=(8, 4))
esg_time.plot(marker='o')
plt.title("Average ESG Score Over Time")
plt.xlabel("Year")
plt.ylabel("Average ESG Score")
plt.grid(True)
plt.show()
```

---

## 7. ROA Over Time

Plot average ROA over time to observe trends in firm performance.

```python
roa_time = df.groupby("year")["roa"].mean()

plt.figure(figsize=(8, 4))
roa_time.plot(marker='o')
plt.title("Average ROA Over Time")
plt.xlabel("Year")
plt.ylabel("Average ROA")
plt.grid(True)
plt.show()
```

---

## 8. ESG Score Over Time by Industry

Plot ESG trends over time for different industries to identify sector-specific patterns.

```python
esg_ind_time = df.groupby(["year", "ff12"])["esg_score"].mean().reset_index()

plt.figure(figsize=(10, 5))
sns.lineplot(data=esg_ind_time, x="year", y="esg_score", hue="ff12")
plt.title("ESG Score Over Time by Industry")
plt.xlabel("Year")
plt.ylabel("Average ESG Score")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()
```

---

## Summary of Findings

| Analysis | Key Insight |
|---|---|
| ESG Score Distribution | Reveals overall variation, skewness, and potential outliers in ESG scores |
| Financial Variables Distribution | Shows individual patterns and extreme values in ROA, leverage, and firm size |
| Industry Distribution (FF12) | Indicates which sectors are most heavily represented in the sample |
| ESG Score by Industry | Reveals cross-industry differences in ESG performance and within-industry variability |
| Correlation Matrix | Indicates linear associations between ESG and financial controls; informs model specification |
| ESG Score Over Time | Captures temporal trends in ESG reporting/performance across the sample period |
| ROA Over Time | Tracks trends in firm profitability over time |
| ESG Over Time by Industry | Highlights sector-specific ESG adoption trends and divergences |

Overall, the EDA provides crucial insights into the data's structure, variable distributions, inter-variable relationships, and temporal and industry-specific patterns. These findings guide model selection and the inclusion of appropriate control variables and fixed effects in subsequent regression analysis.
