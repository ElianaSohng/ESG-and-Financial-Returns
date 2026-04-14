# Part II - What determines firms’ ESG performance?

""" Main design

1. We examine two broad channels that may explain firm-level ESG performance: internal firm characteristics and external industry environment.
- For the internal channel, we begin with a core specification including firm size, profitability (measured by ROA), and leverage.
- For the external channel, we test whether a firm's ESG performance is related to lagged peer ESG within the same FF12 industry.

2. After estimating the core internal model, we take a closer look at the profitability result. Because ROA combines multiple dimensions of firm performance, we further decompose it into profitability and asset turnover. This allows us to distinguish whether ESG is more closely related to firms' ability to generate profits or to their operational efficiency.

3. We compare the decomposed specification with the earlier core model and use the results to identify which firm-level factors appear to be the most important determinants of ESG performance.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

plt.rcParams["figure.figsize"] = (8, 5)
pd.set_option("display.max_columns", 200)
pd.set_option("display.float_format", lambda x: f"{x:,.4f}")

"""**1. Load data and check the sample**

We use the cleaned firm-year panel prepared for the project.  
This file already contains:
- ESG score
- accounting variables
- pre-built `size`, `lev`, and `roa`
- firm identifiers and SIC codes
"""

df = pd.read_csv("data/final_panel.csv")
print("Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

if "gvkey" in df.columns:
    firm_id = "gvkey"
elif "permno" in df.columns:
    firm_id = "permno"
else:
    raise ValueError("Cannot find firm identifier. Need gvkey or permno.")

df["year"] = df["fyear"].astype(int)
df = df.sort_values([firm_id, "year"]).copy()
df.head()

"""
2. Build FF12 industries from SIC

We use **FF12** rather than a simple 2-digit SIC grouping because:
- the rest of the project already uses FF12 in industry EDA
- FF12 is more standard in finance-style empirical work
- it gives a cleaner and more interpretable industry grouping for fixed effects and peer construction
"""

def sic_to_ff12(sic):
    try:
        s = int(float(sic))
    except (ValueError, TypeError):
        return "Other"

    if (100 <= s <= 999) or (2000 <= s <= 2399) or (2700 <= s <= 2749) or \
       (2770 <= s <= 2799) or (3100 <= s <= 3199) or (3940 <= s <= 3989):
        return "Consumer Non-Durables"

    elif (2500 <= s <= 2519) or (2590 <= s <= 2599) or (3630 <= s <= 3659) or \
         s in (3710, 3711, 3714, 3716) or (3750 <= s <= 3751) or s == 3792 or \
         (3900 <= s <= 3939) or (3990 <= s <= 3999):
        return "Consumer Durables"

    elif (2520 <= s <= 2589) or (2600 <= s <= 2699) or (2750 <= s <= 2769) or \
         (2800 <= s <= 2829) or (2840 <= s <= 2899) or (3000 <= s <= 3099) or \
         (3200 <= s <= 3569) or (3580 <= s <= 3621) or (3700 <= s <= 3709) or \
         (3717 <= s <= 3749) or (3752 <= s <= 3791) or (3793 <= s <= 3799) or \
         (3830 <= s <= 3839) or (3860 <= s <= 3899):
        return "Manufacturing"

    elif (1200 <= s <= 1399) or (2900 <= s <= 2999):
        return "Energy"

    elif 2830 <= s <= 2836:
        return "Healthcare"

    elif (7372 <= s <= 7374) or (3570 <= s <= 3579) or (3660 <= s <= 3695) or \
         (3810 <= s <= 3829) or (3840 <= s <= 3859):
        return "Technology"

    elif 4800 <= s <= 4899:
        return "Telecom"

    elif 4900 <= s <= 4999:
        return "Utilities"

    elif (5000 <= s <= 5999) or (7000 <= s <= 7299) or (7400 <= s <= 7499):
        return "Retail & Services"

    elif 6000 <= s <= 6999:
        return "Finance"

    else:
        return "Other"

df["ff12"] = df["sic"].apply(sic_to_ff12)
df["ff12"].value_counts().sort_index()

"""
**3. Construct internal and external variables**

 ### Core internal variables
- `lag_size`
- `lag_roa`
- `lag_lev`

### Decomposed / extended internal variables
- `lag_profit_margin`
- `lag_asset_turnover`
- `lag_lt_debt_share`
- `lag_sale_growth`

### External variable
- `peer_esg_lag_loo`: lagged leave-one-out industry peer ESG within FF12

Using **lagged** firm characteristics helps keep the direction of interpretation cleaner: we are relating earlier firm attributes to later ESG performance.


**Why we choose these factors: to test the following hypotheses**

- **Size**: bigger firms have more resources and visibility → expected **positive**.
- **ROA**: more profitable firms can better support ESG investment → expected **positive**.
- **Leverage**: higher debt may constrain ESG spending → expected **negative or mixed**.
- **Peer ESG**: industry norms may shape firm ESG behavior → expected **positive**.
- **ROA decomposition**: we split ROA into **profit margin** and **asset turnover** to see whether ESG is more related to profitability or efficiency.
"""

# Core lags
for v in ["size", "roa", "lev"]:
    df[f"lag_{v}"] = df.groupby(firm_id)[v].shift(1)

# Decomposed / extended internal variables
df["profit_margin"] = np.where(
    (df["sale"].notna()) & (df["sale"] != 0),
    df["ni"] / df["sale"],
    np.nan
)

df["asset_turnover"] = np.where(
    (df["at"].notna()) & (df["at"] != 0),
    df["sale"] / df["at"],
    np.nan
)

total_debt = df["dltt"].fillna(0) + df["dlc"].fillna(0)
df["lt_debt_share"] = np.where(
    total_debt > 0,
    df["dltt"].fillna(0) / total_debt,
    np.nan
)

df["sale_lag_raw"] = df.groupby(firm_id)["sale"].shift(1)
df["sale_growth"] = np.where(
    (df["sale_lag_raw"].notna()) & (df["sale_lag_raw"] != 0),
    df["sale"] / df["sale_lag_raw"] - 1,
    np.nan
)

for v in ["profit_margin", "asset_turnover", "lt_debt_share", "sale_growth"]:
    df[f"lag_{v}"] = df.groupby(firm_id)[v].shift(1)

# External peer ESG: FF12-year leave-one-out mean, then lag by firm
grp = df.groupby(["ff12", "year"])["esg_score"]
df["ff12_year_sum"] = grp.transform("sum")
df["ff12_year_n"] = grp.transform("count")

df["peer_esg_loo"] = np.where(
    df["ff12_year_n"] > 1,
    (df["ff12_year_sum"] - df["esg_score"]) / (df["ff12_year_n"] - 1),
    np.nan
)

df["peer_esg_lag_loo"] = df.groupby(firm_id)["peer_esg_loo"].shift(1)

df[[
    "lag_size", "lag_roa", "lag_lev",
    "lag_profit_margin", "lag_asset_turnover",
    "lag_lt_debt_share", "lag_sale_growth",
    "peer_esg_lag_loo"
]].head()

"""
## 4. Winsorize lagged regressors

To reduce the influence of extreme outliers, we winsorize the lagged regressors at the 1st and 99th percentiles.
"""

def winsorize_series(s, lower=0.01, upper=0.99):
    if s.notna().sum() == 0:
        return s
    lo = s.quantile(lower)
    hi = s.quantile(upper)
    return s.clip(lo, hi)

winsor_cols = [
    "lag_size", "lag_roa", "lag_lev",
    "lag_profit_margin", "lag_asset_turnover",
    "lag_lt_debt_share", "lag_sale_growth",
    "peer_esg_lag_loo"
]

for col in winsor_cols:
    df[col] = winsorize_series(df[col])

df[winsor_cols].describe().T[["count", "mean", "std", "min", "max"]]

"""
## 5. Correlation heat map

This is only a descriptive first look.  
It helps us see broad patterns.
"""

heatmap_cols = [
    "esg_score",
    "lag_size", "lag_roa", "lag_lev",
    "lag_profit_margin", "lag_asset_turnover",
    "lag_lt_debt_share", "lag_sale_growth",
    "peer_esg_lag_loo"
]

corr = df[heatmap_cols].dropna().corr()

plt.figure(figsize=(9, 7))
im = plt.imshow(corr, aspect="auto")
plt.colorbar(im)
plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
plt.yticks(range(len(corr.index)), corr.index)
plt.title("Correlation Heatmap for Part II Variables")
plt.tight_layout()
plt.show()

corr.round(3)

"""
### What the heat map suggests

Before controlling for anything:
- `lag_size` has the strongest positive simple correlation with ESG
- `lag_roa` is also positively correlated with ESG
- `lag_lt_debt_share` is positively correlated with ESG
- `lag_sale_growth` is negatively correlated with ESG
- `peer_esg_lag_loo` is positively correlated with ESG, but this may partly reflect broad industry differences

These are only descriptive signals. The regression models below tell us which relationships remain after controls.
"""

## 6. Model setup


def extract_coefs(model, variables, model_name):
    rows = []
    for v in variables:
        rows.append({
            "model": model_name,
            "variable": v,
            "coef": model.params.get(v, np.nan),
            "std_err": model.bse.get(v, np.nan),
            "p_value": model.pvalues.get(v, np.nan),
            "nobs": int(model.nobs),
            "adj_r2": model.rsquared_adj
        })
    return pd.DataFrame(rows)

"""
## 7. Core internal model
This is the clean baseline internal specification:


$$
ESG_{i,t} = \alpha + \beta_1 lag\_size_{i,t} + \beta_2 lag\_roa_{i,t} + \beta_3 lag\_lev_{i,t} + YearFE + IndustryFE + \varepsilon_{i,t}
$$

This gives a simple baseline story:
- Do larger firms have higher ESG?
- Do more profitable firms have higher ESG?
- Does leverage matter once we control for year and industry?
"""

pars_cols = ["esg_score", "lag_size", "lag_roa", "lag_lev", "year", "ff12"]
pars_df = df[pars_cols].dropna().copy()

m_pars = smf.ols(
    "esg_score ~ lag_size + lag_roa + lag_lev + C(year) + C(ff12)",
    data=pars_df
).fit(cov_type="HC1")

pars_table = extract_coefs(m_pars, ["lag_size", "lag_roa", "lag_lev"], "Parsimonious internal")
pars_table

"""
### Interpretation of the core internal model

Using the current cleaned sample:
- `lag_size` is **positive and strongly significant**
- `lag_roa` is **positive and strongly significant**
- `lag_lev` is **much weaker** and only marginal / unstable

This suggests that, at a broad level, ESG performance is more strongly associated with:
1. **firm scale / resources**
2. **profitability**
than with total leverage alone.
"""

"""
## 8. External peer model
This isolates the industry-peer channel:

$$
ESG_{i,t} = \alpha + \beta_1 peer\_esg\_lag\_loo_{i,t} + YearFE + \varepsilon_{i,t}
$$

We also estimate:

$$
ESG_{i,t} = \alpha + \beta_1 peer\_esg\_lag\_loo_{i,t} + YearFE + IndustryFE + \varepsilon_{i,t}
$$

to see whether the peer effect survives once broad FF12 industry differences are absorbed.

Now we isolate the industry-peer channel.  
The key question is:

> Does a firm's ESG score appear higher when the prior ESG level of its FF12 peers is higher?

We first estimate a simple peer model, then add FF12 fixed effects.
"""

ext_cols = ["esg_score", "peer_esg_lag_loo", "year"]
ext_df = df[ext_cols].dropna().copy()

m_ext = smf.ols(
    "esg_score ~ peer_esg_lag_loo + C(year)",
    data=ext_df
).fit(cov_type="HC1")

ext_ff_cols = ["esg_score", "peer_esg_lag_loo", "year", "ff12"]
ext_ff_df = df[ext_ff_cols].dropna().copy()

m_ext_ff = smf.ols(
    "esg_score ~ peer_esg_lag_loo + C(year) + C(ff12)",
    data=ext_ff_df
).fit(cov_type="HC1")

ext_table = pd.concat([
    extract_coefs(m_ext, ["peer_esg_lag_loo"], "External only"),
    extract_coefs(m_ext_ff, ["peer_esg_lag_loo"], "External + FF12 FE")
], ignore_index=True)

ext_table

"""
### How to interpret the peer ESG result

The pattern we expect — and the current data support — is:

- `peer_esg_lag_loo` looks **strongly positive** in the simple external model
- but once we add **industry fixed effects**, the coefficient becomes much weaker and no longer robust

The logic is important:

- Without industry FE, the peer variable captures a lot of **between-industry differences**
- With industry FE, we strip out those broad industry-level average differences
- What remains is closer to **within-industry peer variation**

So if the coefficient weakens sharply, the right interpretation is:

> the peer ESG measure mainly captures **broad industry ESG environment**, rather than a strong and robust within-industry peer spillover.
"""

"""
## 9. Joint internal model
Since ROA is strongly associated with ESG in the core internal model, we next decompose ROA into profit margin and asset turnover to better understand which dimension drives the relation.

### Joint internal model
A key issue is that:

$$
ROA = \frac{NI}{AT} = \frac{NI}{SALE} \times \frac{SALE}{AT}
$$

So ROA mechanically combines:
- **profit margin** = `ni / sale`
- **asset turnover** = `sale / at`


This is the model that should carry the main weight for identifying **key internal factors**:

$$
ESG_{i,t} =
\alpha
+ \beta_1 lag\_size_{i,t}
+ \beta_2 lag\_profit\_margin_{i,t}
+ \beta_3 lag\_asset\_turnover_{i,t}
+ \beta_4 lag\_lev_{i,t}
+ \beta_5 lag\_lt\_debt\_share_{i,t}
+ \beta_6 lag\_sale\_growth_{i,t}
+ YearFE + IndustryFE + \varepsilon_{i,t}
$$


This model is preferred for the "key determinants" question because the candidate internal channels are allowed to compete in the **same regression**.
"""

joint_cols = [
    "esg_score",
    "lag_size", "lag_profit_margin", "lag_asset_turnover",
    "lag_lev", "lag_lt_debt_share", "lag_sale_growth",
    "year", "ff12"
]
joint_df = df[joint_cols].dropna().copy()

m_joint = smf.ols(
    "esg_score ~ lag_size + lag_profit_margin + lag_asset_turnover + lag_lev + lag_lt_debt_share + lag_sale_growth + C(year) + C(ff12)",
    data=joint_df
).fit(cov_type="HC1")

joint_table = extract_coefs(
    m_joint,
    ["lag_size", "lag_profit_margin", "lag_asset_turnover", "lag_lev", "lag_lt_debt_share", "lag_sale_growth"],
    "Joint internal"
)

joint_table

"""
### Main takeaway from the joint internal model

This is the model that should drive the final "key determinants" discussion.

The current results show:

- `lag_size` remains **positive and very robust**
- `lag_asset_turnover` remains **positive and strongly significant**
- `lag_lt_debt_share` remains **positive and strongly significant**
- `lag_sale_growth` remains **negative and strongly significant**
- `lag_profit_margin` is **not robust**
- `lag_lev` is **not robust**

That means the strongest internal story is not simply "profitable firms have higher ESG."  
A more refined reading is:

1. **Larger firms** tend to have higher ESG
2. **Operational efficiency** matters: firms with higher asset turnover tend to have higher ESG
3. **Debt structure** matters more than total leverage: firms with a larger long-term debt share tend to have higher ESG
4. **Rapid growth** is associated with lower ESG, consistent with the idea that fast expansion can outpace ESG investment / disclosure systems
"""

"""
## 10. Same-sample comparison: core vs joint internal model

One caution is that the joint model uses a smaller common sample because variables such as sales growth and long-term debt share are missing for some firm-years.

So to compare model fit fairly, we should re-estimate the core model on the **same sample** used by the joint model.
"""

joint_sample_df = df[[
    "esg_score", "lag_size", "lag_roa", "lag_lev",
    "lag_profit_margin", "lag_asset_turnover",
    "lag_lt_debt_share", "lag_sale_growth",
    "year", "ff12"
]].dropna().copy()

m_pars_same = smf.ols(
    "esg_score ~ lag_size + lag_roa + lag_lev + C(year) + C(ff12)",
    data=joint_sample_df
).fit(cov_type="HC1")

compare_table = pd.DataFrame([
    {
        "model": "Parsimonious internal (same joint sample)",
        "nobs": int(m_pars_same.nobs),
        "adj_r2": m_pars_same.rsquared_adj
    },
    {
        "model": "Joint internal",
        "nobs": int(m_joint.nobs),
        "adj_r2": m_joint.rsquared_adj
    }
])

compare_table

"""
### Why this comparison matters

If the joint model has a higher adjusted R-squared **on the same sample**, then we can say:

> the richer internal-channel model adds explanatory power, rather than just changing because the sample changed.

In the current data, this same-sample comparison supports the richer joint internal specification.
"""


## 11. Final concise result tables



main_tables = pd.concat([pars_table, ext_table, joint_table], ignore_index=True)
main_tables

from pathlib import Path

# make an output folder
output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

models = {
    "internal": m_pars,
    "external": m_ext,
    "combined": m_joint,
    "external_ff12": m_ext_ff,
}

for name, model in models.items():
    coef_table = model.summary2().tables[1]   # coefficient table as a DataFrame

    print(f"\n{'='*20} {name.upper()} {'='*20}")
    print(coef_table)

    coef_table.to_csv(output_dir / f"{name}_coef_table.csv")