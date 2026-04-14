"""
RSM8224 Group Project – Part III Analysis (Final Version)
==========================================================
Does Doing Good Mean Doing Well?

This script runs the core empirical tests for Part III:

  TABLE 1 – Future ROA (t+1) ~ ESG Score (t) + Controls
  TABLE 2 – Future 12-Month Return (t+1) ~ ESG Score (t) + Controls
  TABLE 3A – Robustness: ROA regressions (z-score, dummy, excl. financials)
  TABLE 3B – Robustness: Return regressions (z-score, dummy, excl. financials)

Design choices:
  - "Lagged ESG" = ESG at fiscal year t; outcomes measured at t+1.
  - Earnings news proxy = ROA(t) - ROA(t-1), a backward-looking measure
    of earnings momentum available at portfolio formation. Only computed
    when fiscal years are consecutive (no gap-year contamination).
  - B/M = book equity (CEQ) / market equity (|price| x shares outstanding
    from CRSP, measured in December of fiscal year t).
  - ROA regressions use size = log(total assets) as the size control.
  - Return regressions use log_me = log(market equity) as the size control,
    which better aligns with asset pricing conventions.
  - All standard errors clustered by firm (gvkey).

Outputs (in output/ directory):
  Tables : part3_tables.txt, part3_roa_regression.csv,
           part3_ret_regression.csv, part3_robustness.csv
  Figures: part3_fig_esg_roa.png, part3_fig_esg_roa_resid.png,
           part3_fig_esg_ret.png, part3_fig_esg_ret_resid.png,
           part3_fig_tercile.png
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


# =============================================================================
# 0. PATHS & SETTINGS
# =============================================================================
DATA_PATH  = Path("data/final_panel.csv")
CRSP_PATH  = Path("data/crsp_monthly.csv")
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_BINS = 20


def sig_stars(p: float) -> str:
    if p < 0.01:   return "***"
    elif p < 0.05: return "**"
    elif p < 0.10: return "*"
    return ""


def safe_tercile(x):
    """Assign tercile labels, handling duplicate bin edges gracefully."""
    try:
        return pd.qcut(x, 3, labels=["Low", "Mid", "High"], duplicates="drop")
    except (ValueError, TypeError):
        return pd.Series(pd.NA, index=x.index)


# =============================================================================
# 1. LOAD PANEL DATA
# =============================================================================
print("=" * 70)
print("PART III: DOES DOING GOOD MEAN DOING WELL?")
print("=" * 70)

df = pd.read_csv(DATA_PATH)
print(f"\nLoaded panel: {df.shape[0]:,} firm-years, {df.shape[1]} columns")

for col in ["esg_score", "roa", "future_roa", "future_return",
            "lev", "size", "ni", "at", "sale", "ceq",
            "fyear", "sic", "permno"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")


# =============================================================================
# 2. CONSTRUCT MARKET EQUITY & TRUE BOOK-TO-MARKET FROM CRSP
# =============================================================================
print("\n--- Constructing Market Equity from CRSP ---")

crsp_raw = pd.read_csv(CRSP_PATH, low_memory=False)
crsp_raw.columns = [str(c).strip().lower() for c in crsp_raw.columns]

rename_map = {"mthcaldt": "date", "mthret": "ret", "mthprc": "prc"}
crsp_raw = crsp_raw.rename(columns=rename_map)

crsp_raw["date"]   = pd.to_datetime(crsp_raw["date"], errors="coerce")
crsp_raw["permno"] = pd.to_numeric(crsp_raw["permno"], errors="coerce")

prc_col = "prc" if "prc" in crsp_raw.columns else None
shr_col = None
for candidate in ["shrout", "mthshrout", "shrtshrout"]:
    if candidate in crsp_raw.columns:
        shr_col = candidate
        break

HAS_TRUE_BM = False

if prc_col and shr_col:
    crsp_raw[prc_col] = pd.to_numeric(crsp_raw[prc_col], errors="coerce")
    crsp_raw[shr_col] = pd.to_numeric(crsp_raw[shr_col], errors="coerce")

    # CRSP uses negative price for bid-ask midpoint; take absolute value
    crsp_raw["abs_prc"] = crsp_raw[prc_col].abs()

    # Market equity = |price| x shares outstanding
    crsp_raw["me"] = crsp_raw["abs_prc"] * crsp_raw[shr_col]

    crsp_raw["year"]  = crsp_raw["date"].dt.year
    crsp_raw["month"] = crsp_raw["date"].dt.month

    crsp_me = crsp_raw.dropna(subset=["me"]).copy()
    crsp_me = crsp_me[crsp_me["me"] > 0]

    # For each permno-year: prefer December, fall back to last available month
    crsp_me = crsp_me.sort_values(["permno", "year", "month"])
    crsp_dec     = crsp_me[crsp_me["month"] == 12].drop_duplicates(
        subset=["permno", "year"], keep="last"
    )
    crsp_last    = crsp_me.drop_duplicates(subset=["permno", "year"], keep="last")
    crsp_dec_set = set(zip(crsp_dec["permno"], crsp_dec["year"]))
    crsp_fill    = crsp_last[
        ~crsp_last.apply(lambda r: (r["permno"], r["year"]) in crsp_dec_set, axis=1)
    ]
    me_annual = pd.concat([crsp_dec, crsp_fill], ignore_index=True)
    me_annual = me_annual[["permno", "year", "me"]].rename(columns={"year": "fyear"})
    me_annual = me_annual.drop_duplicates(subset=["permno", "fyear"], keep="last")

    df = pd.merge(df, me_annual, on=["permno", "fyear"], how="left")

    # True B/M = book equity / market equity
    df["bm"] = df["ceq"] / df["me"]
    df.loc[df["bm"] <= 0, "bm"] = np.nan
    df["bm"] = df["bm"].replace([np.inf, -np.inf], np.nan)

    # Log market equity for return regressions
    df["log_me"] = np.log(df["me"])
    df["log_me"] = df["log_me"].replace([np.inf, -np.inf], np.nan)

    HAS_TRUE_BM = True
    print(f"  Market equity merged. Non-missing B/M: {df['bm'].notna().sum():,}")
else:
    print("  [WARNING] CRSP lacks price/shares columns.")
    print("  Falling back to equity-to-assets ratio as B/M proxy.")
    df["bm"] = df["ceq"] / df["at"]
    df.loc[df["bm"] <= 0, "bm"] = np.nan
    df["bm"] = df["bm"].replace([np.inf, -np.inf], np.nan)
    df["log_me"] = df["size"].copy()  # fallback: use log(assets)

bm_label = ("Book Equity / Market Equity (Dec. of fiscal year t)"
            if HAS_TRUE_BM else "Book Equity / Total Assets (proxy)")


# =============================================================================
# 3. CONSTRUCT ADDITIONAL VARIABLES
# =============================================================================
print("\n--- Constructing Analysis Variables ---")

df = df.sort_values(["gvkey", "fyear"]).copy()

# (a) Standardized ESG score (within-year z-score)
df["esg_z"] = df.groupby("fyear")["esg_score"].transform(
    lambda x: (x - x.mean()) / x.std()
)

# (b) ESG terciles with safe handling of duplicate edges
df["esg_tercile"] = df.groupby("fyear")["esg_score"].transform(safe_tercile)

# (c) High-ESG dummy (top tercile = 1)
df["esg_high"] = (df["esg_tercile"] == "High").astype(int)

# (d) Lagged earnings change with consecutive-year check
#     Only compute when fyear(t) == fyear(t-1) + 1 to avoid gap-year contamination
df["fyear_lag1"] = df.groupby("gvkey")["fyear"].shift(1)
df["roa_lag1"]   = df.groupby("gvkey")["roa"].shift(1)
df["earn_chg"]   = np.where(
    df["fyear"] == df["fyear_lag1"] + 1,
    df["roa"] - df["roa_lag1"],
    np.nan
)

# (e) Sales growth (t vs t-1), with inf protection
df["sale_growth"] = df.groupby("gvkey")["sale"].pct_change()
df["sale_growth"] = df["sale_growth"].replace([np.inf, -np.inf], np.nan)
# Also apply consecutive-year check
df["sale_growth"] = np.where(
    df["fyear"] == df["fyear_lag1"] + 1,
    df["sale_growth"],
    np.nan
)

# (f) Industry and year identifiers
if "sic2" not in df.columns:
    df["sic2"] = (df["sic"] // 100).astype("Int64")
df["is_financial"] = ((df["sic"] >= 6000) & (df["sic"] <= 6999)).astype(int)
df["fyear_cat"]    = df["fyear"].astype(str)

# Summary
print("\n--- Key Variable Summary ---")
key_vars = ["esg_score", "esg_z", "roa", "future_roa", "future_return",
            "earn_chg", "lev", "size", "log_me", "bm"]
key_vars = [v for v in key_vars if v in df.columns]
print(df[key_vars].describe().round(4).to_string())

print(f"\nTotal firm-years: {len(df):,}")
print(f"With future_roa:  {df['future_roa'].notna().sum():,}")
print(f"With earn_chg:    {df['earn_chg'].notna().sum():,}")
print(f"With B/M:         {df['bm'].notna().sum():,}")
print(f"B/M definition:   {bm_label}")


# =============================================================================
# 4. REGRESSION HELPERS
# =============================================================================
def run_ols(
    data: pd.DataFrame,
    y_col: str,
    x_cols: list[str],
    fe_cols: list[str] | None = None,
    cluster_col: str = "gvkey",
    label: str = ""
) -> dict | None:
    """OLS with optional FE dummies and firm-clustered SEs."""
    reg_df = data.copy()
    rhs_cols = list(x_cols)

    if fe_cols:
        for fe in fe_cols:
            dummies = pd.get_dummies(reg_df[fe], prefix=fe, drop_first=True, dtype=float)
            reg_df = pd.concat([reg_df, dummies], axis=1)
            rhs_cols += dummies.columns.tolist()

    all_cols = [y_col] + rhs_cols + [cluster_col]
    all_cols = [c for c in all_cols if c in reg_df.columns]
    reg_df = reg_df.dropna(subset=all_cols)

    # Remove any remaining inf values
    for c in [y_col] + rhs_cols:
        if c in reg_df.columns and reg_df[c].dtype.kind == "f":
            reg_df = reg_df[np.isfinite(reg_df[c])]

    if reg_df.shape[0] < 50:
        print(f"  [WARNING] {label}: Only {reg_df.shape[0]} obs — skipping.")
        return None

    Y = reg_df[y_col].values.astype(float)
    X = sm.add_constant(reg_df[rhs_cols].values.astype(float))
    col_names = ["const"] + rhs_cols

    result = sm.OLS(Y, X).fit(
        cov_type="cluster",
        cov_kwds={"groups": reg_df[cluster_col].values},
        use_t=True
    )
    result._col_names = col_names

    return {
        "result":  result,
        "label":   label,
        "y_col":   y_col,
        "n_obs":   int(result.nobs),
        "n_firms": reg_df[cluster_col].nunique(),
        "r2":      result.rsquared,
        "r2_adj":  result.rsquared_adj,
        "fe":      fe_cols or [],
    }


def format_regression_table(
    results_list: list[dict],
    display_vars: list[str],
    title: str = ""
) -> str:
    """Format regression results into a publication-style text table."""
    n = len(results_list)
    w = max(80, 20 + 16 * n)
    lines = ["", "=" * w, title, "=" * w]

    # Header row
    header = f"{'Variable':<20s}"
    for i in range(n):
        header += f"  {'(' + str(i+1) + ')':<14s}"
    lines.append(header)

    dep = f"{'Dep var:':<20s}"
    for r in results_list:
        dep += f"  {r['y_col']:<14s}"
    lines.append(dep)
    lines.append("-" * w)

    # Coefficient rows
    for var in display_vars:
        cl = f"{var:<20s}"
        sl = f"{'':<20s}"
        for r in results_list:
            cn = r["result"]._col_names
            if var in cn:
                idx = cn.index(var)
                c = r["result"].params[idx]
                s = r["result"].bse[idx]
                p = r["result"].pvalues[idx]
                cl += f"  {c:>10.4f}{sig_stars(p):<3s} "
                sl += f"  ({s:>9.4f})    "
            else:
                cl += f"  {'':>14s}"
                sl += f"  {'':>14s}"
        lines.append(cl)
        lines.append(sl)

    lines.append("-" * w)

    # Footer diagnostics
    for lbl, key in [("Observations", "n_obs"), ("Firms", "n_firms")]:
        row = f"{lbl:<20s}"
        for r in results_list:
            row += f"  {r[key]:>14,d}"
        lines.append(row)

    for lbl, key in [("R-squared", "r2"), ("Adj. R-squared", "r2_adj")]:
        row = f"{lbl:<20s}"
        for r in results_list:
            row += f"  {r[key]:>14.4f}"
        lines.append(row)

    for lbl, fe_name in [("Year FE", "fyear_cat"), ("Industry FE", "sic2")]:
        row = f"{lbl:<20s}"
        for r in results_list:
            row += f"  {'Yes' if fe_name in r['fe'] else 'No':>14s}"
        lines.append(row)

    row = f"{'Cluster':<20s}"
    for _ in results_list:
        row += f"  {'Firm':>14s}"
    lines.append(row)

    lines += ["=" * w,
              "Firm-clustered standard errors in parentheses.",
              "*** p<0.01, ** p<0.05, * p<0.10", ""]
    return "\n".join(lines)


def results_to_csv(results_list: list[dict], savepath: Path):
    """Export coefficients (excluding FE dummies) to CSV."""
    rows = []
    for i, r in enumerate(results_list, 1):
        res = r["result"]
        cn = res._col_names
        for j, var in enumerate(cn):
            if var.startswith("fyear_cat_") or var.startswith("sic2_"):
                continue
            rows.append({
                "model": i, "model_label": r["label"], "dep_var": r["y_col"],
                "variable": var, "coefficient": res.params[j],
                "std_error": res.bse[j], "t_stat": res.tvalues[j],
                "p_value": res.pvalues[j], "sig": sig_stars(res.pvalues[j]),
                "n_obs": r["n_obs"], "n_firms": r["n_firms"],
                "r2": r["r2"], "r2_adj": r["r2_adj"],
            })
    pd.DataFrame(rows).to_csv(savepath, index=False)
    print(f"  Saved: {savepath}")


# =============================================================================
# 5. TABLE 1: FUTURE ROA REGRESSIONS
# =============================================================================
print("\n" + "=" * 70)
print("TABLE 1: ESG AND FUTURE ROA")
print("=" * 70)

# ROA regressions use size = log(total assets)
roa_sample = df.dropna(subset=["future_roa", "esg_score", "roa", "lev", "size"]).copy()
print(f"ROA sample: {roa_sample.shape[0]:,} obs, {roa_sample['gvkey'].nunique():,} firms")

roa_m1 = run_ols(roa_sample, "future_roa", ["esg_score"],
                  label="Univariate")

roa_m2 = run_ols(roa_sample, "future_roa", ["esg_score", "roa", "size", "lev"],
                  label="+ Controls")

roa_m3 = run_ols(roa_sample, "future_roa", ["esg_score", "roa", "size", "lev"],
                  fe_cols=["fyear_cat", "sic2"], label="+ FE")

roa_m4 = run_ols(
    roa_sample.dropna(subset=["bm", "sale_growth"]),
    "future_roa",
    ["esg_score", "roa", "size", "lev", "bm", "sale_growth"],
    fe_cols=["fyear_cat", "sic2"], label="+ Extended"
)

roa_results = [r for r in [roa_m1, roa_m2, roa_m3, roa_m4] if r]
roa_table = format_regression_table(
    roa_results,
    display_vars=["esg_score", "roa", "size", "lev", "bm", "sale_growth", "const"],
    title="TABLE 1: FUTURE ROA (t+1) ON LAGGED ESG SCORE (t)"
)
print(roa_table)


# =============================================================================
# 6. TABLE 2: FUTURE STOCK RETURN REGRESSIONS
# =============================================================================
print("\n" + "=" * 70)
print("TABLE 2: ESG AND FUTURE STOCK RETURNS")
print("=" * 70)

# Return regressions use log_me as the size control (asset pricing convention)
ret_sample = df.dropna(
    subset=["future_return", "esg_score", "roa", "lev", "log_me"]
).copy()
print(f"Return sample: {ret_sample.shape[0]:,} obs, {ret_sample['gvkey'].nunique():,} firms")

ret_m1 = run_ols(ret_sample, "future_return", ["esg_score"],
                  label="Unconditional")

ret_m2 = run_ols(ret_sample, "future_return", ["esg_score"],
                  fe_cols=["fyear_cat"], label="+ Year FE")

ret_m3 = run_ols(ret_sample, "future_return",
                  ["esg_score", "roa", "log_me", "lev"],
                  fe_cols=["fyear_cat", "sic2"], label="+ Controls + FE")

ret_m4 = run_ols(
    ret_sample.dropna(subset=["bm"]),
    "future_return",
    ["esg_score", "roa", "log_me", "lev", "bm"],
    fe_cols=["fyear_cat", "sic2"], label="+ B/M"
)

ret_m5 = run_ols(
    ret_sample.dropna(subset=["earn_chg", "bm"]),
    "future_return",
    ["esg_score", "roa", "log_me", "lev", "bm", "earn_chg"],
    fe_cols=["fyear_cat", "sic2"], label="+ Earnings news"
)

ret_m6 = run_ols(
    ret_sample.dropna(subset=["earn_chg", "bm", "sale_growth"]),
    "future_return",
    ["esg_score", "roa", "log_me", "lev", "bm", "earn_chg", "sale_growth"],
    fe_cols=["fyear_cat", "sic2"], label="Full model"
)

ret_results = [r for r in [ret_m1, ret_m2, ret_m3, ret_m4, ret_m5, ret_m6] if r]
ret_table = format_regression_table(
    ret_results,
    display_vars=["esg_score", "roa", "log_me", "lev", "bm",
                  "earn_chg", "sale_growth", "const"],
    title="TABLE 2: FUTURE 12-MONTH RETURN (t+1) ON LAGGED ESG SCORE (t)"
)
print(ret_table)


# =============================================================================
# 7. TABLE 3A & 3B: ROBUSTNESS CHECKS (split by dependent variable)
# =============================================================================
print("\n" + "=" * 70)
print("TABLE 3: ROBUSTNESS CHECKS")
print("=" * 70)

base_fe = ["fyear_cat", "sic2"]

# --- 3A: ROA robustness ---
roa_rob1 = run_ols(roa_sample, "future_roa",
                    ["esg_z", "roa", "size", "lev"],
                    fe_cols=base_fe, label="ESG z-score")

roa_rob2 = run_ols(roa_sample, "future_roa",
                    ["esg_high", "roa", "size", "lev"],
                    fe_cols=base_fe, label="High-ESG dummy")

roa_rob3 = run_ols(roa_sample[roa_sample["is_financial"] == 0],
                    "future_roa",
                    ["esg_score", "roa", "size", "lev"],
                    fe_cols=base_fe, label="Excl. financials")

roa_rob_results = [r for r in [roa_rob1, roa_rob2, roa_rob3] if r]
rob_table_a = format_regression_table(
    roa_rob_results,
    display_vars=["esg_score", "esg_z", "esg_high", "roa", "size", "lev", "const"],
    title="TABLE 3A: ROBUSTNESS – FUTURE ROA"
)
print(rob_table_a)

# --- 3B: Return robustness ---
ret_rob1 = run_ols(ret_sample, "future_return",
                    ["esg_z", "roa", "log_me", "lev"],
                    fe_cols=base_fe, label="ESG z-score")

ret_rob2 = run_ols(ret_sample, "future_return",
                    ["esg_high", "roa", "log_me", "lev"],
                    fe_cols=base_fe, label="High-ESG dummy")

ret_rob3 = run_ols(ret_sample[ret_sample["is_financial"] == 0],
                    "future_return",
                    ["esg_score", "roa", "log_me", "lev"],
                    fe_cols=base_fe, label="Excl. financials")

ret_rob_results = [r for r in [ret_rob1, ret_rob2, ret_rob3] if r]
rob_table_b = format_regression_table(
    ret_rob_results,
    display_vars=["esg_score", "esg_z", "esg_high", "roa", "log_me", "lev", "const"],
    title="TABLE 3B: ROBUSTNESS – FUTURE RETURN"
)
print(rob_table_b)


# =============================================================================
# 8. ECONOMIC MAGNITUDE
# =============================================================================
print("\n" + "=" * 70)
print("ECONOMIC MAGNITUDE ANALYSIS")
print("=" * 70)

esg_std = df["esg_score"].std()
esg_iqr = df["esg_score"].quantile(0.75) - df["esg_score"].quantile(0.25)

for panel_name, results, y_var in [
    ("ROA", roa_results, "future_roa"),
    ("Return", ret_results, "future_return")
]:
    if not results:
        continue
    # Use most saturated model that still has esg_score
    best = None
    for r in reversed(results):
        if "esg_score" in r["result"]._col_names:
            best = r
            break
    if best is None:
        continue

    res = best["result"]
    cn  = res._col_names
    idx = cn.index("esg_score")
    coef, se, pval = res.params[idx], res.bse[idx], res.pvalues[idx]

    y_mean = df[y_var].mean()
    y_std  = df[y_var].std()

    effect_1sd  = coef * esg_std
    effect_iqr  = coef * esg_iqr
    std_beta    = coef * esg_std / y_std if y_std > 0 else np.nan
    pct_of_mean = (effect_1sd / abs(y_mean) * 100) if abs(y_mean) > 1e-6 else np.nan

    print(f"\n{'─' * 55}")
    print(f"Panel: {panel_name}  |  Model: {best['label']}")
    print(f"{'─' * 55}")
    print(f"  ESG coefficient:              {coef:.6f} (SE={se:.6f}, p={pval:.4f})")
    print(f"  ESG score std dev:            {esg_std:.4f}")
    print(f"  ESG score IQR:                {esg_iqr:.4f}")
    print(f"  1-SD ESG effect on {y_var}:     {effect_1sd:.4f} ({effect_1sd*100:.2f} pp)")
    print(f"  IQR ESG effect on {y_var}:      {effect_iqr:.4f} ({effect_iqr*100:.2f} pp)")
    print(f"  Standardized beta:            {std_beta:.4f}")
    print(f"  Effect as % of mean {y_var}:    {pct_of_mean:.1f}%")


# =============================================================================
# 9. ESG TERCILE PORTFOLIO ANALYSIS
# =============================================================================
print("\n" + "=" * 70)
print("ESG TERCILE PORTFOLIO ANALYSIS")
print("=" * 70)

tercile_sample = df.dropna(subset=["esg_tercile", "future_return", "future_roa"])

tercile_stats = (
    tercile_sample
    .groupby("esg_tercile", observed=True)
    .agg(
        n_obs=("future_return", "size"),
        mean_esg=("esg_score", "mean"),
        mean_return=("future_return", "mean"),
        median_return=("future_return", "median"),
        std_return=("future_return", "std"),
        mean_roa=("future_roa", "mean"),
        median_roa=("future_roa", "median"),
        mean_size=("size", "mean"),
        mean_lev=("lev", "mean"),
    )
    .round(4)
)

print("\n--- Average Characteristics by ESG Tercile ---")
print(tercile_stats.to_string())

# High-minus-Low spread with t-test
if "High" in tercile_stats.index and "Low" in tercile_stats.index:
    for outcome, label in [("future_return", "Return"), ("future_roa", "ROA")]:
        high = tercile_sample.loc[tercile_sample["esg_tercile"] == "High", outcome].dropna()
        low  = tercile_sample.loc[tercile_sample["esg_tercile"] == "Low", outcome].dropna()
        spread = high.mean() - low.mean()
        t_stat, t_pval = stats.ttest_ind(high, low, equal_var=False)
        print(f"\n  High-minus-Low {label}: {spread:.4f} ({spread*100:.2f} pp), "
              f"t={t_stat:.2f}, p={t_pval:.4f} {sig_stars(t_pval)}")


# =============================================================================
# 10. FIGURES
# =============================================================================
print("\n" + "=" * 70)
print("GENERATING FIGURES")
print("=" * 70)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def binned_scatter(
    data, x_col, y_col, n_bins,
    xlabel, ylabel, title, savepath,
    residualize_cols=None
):
    """Binned scatter plot, optionally as a partial regression (residualized)."""
    plot_df = data.dropna(subset=[x_col, y_col]).copy()

    if residualize_cols:
        plot_df = plot_df.dropna(subset=residualize_cols)
        C = sm.add_constant(plot_df[residualize_cols].values.astype(float))
        plot_df["y_r"] = sm.OLS(plot_df[y_col].values, C).fit().resid
        plot_df["x_r"] = sm.OLS(plot_df[x_col].values, C).fit().resid
        x_plot, y_plot = "x_r", "y_r"
        xlabel += " (residualized)"
        ylabel += " (residualized)"
    else:
        x_plot, y_plot = x_col, y_col

    plot_df["bin"] = pd.qcut(plot_df[x_plot], n_bins, labels=False, duplicates="drop")
    binned = plot_df.groupby("bin").agg(
        x_mean=(x_plot, "mean"),
        y_mean=(y_plot, "mean"),
        y_se=(y_plot, lambda s: s.std() / np.sqrt(len(s)))
    ).reset_index()

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.errorbar(binned["x_mean"], binned["y_mean"],
                yerr=1.96 * binned["y_se"],
                fmt="o", color="#2C5F8A", ms=7,
                ecolor="#B0B0B0", elinewidth=1, capsize=3,
                label="Bin mean (\u00b11.96 SE)")

    if len(binned) > 2:
        z = np.polyfit(binned["x_mean"], binned["y_mean"], 1)
        xr = np.linspace(binned["x_mean"].min(), binned["x_mean"].max(), 100)
        ax.plot(xr, np.poly1d(z)(xr), "--", color="#C44E52", lw=1.5,
                label=f"Linear fit (slope={z[0]:.5f})")

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(savepath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {savepath}")


# Fig 1 & 2: ESG vs Future ROA
roa_ps = df.dropna(subset=["esg_score", "future_roa"])
binned_scatter(roa_ps, "esg_score", "future_roa", N_BINS,
               "ESG Score (t)", "Future ROA (t+1)",
               "ESG Score and Future ROA",
               OUTPUT_DIR / "part3_fig_esg_roa.png")

binned_scatter(roa_ps.dropna(subset=["roa", "size", "lev"]),
               "esg_score", "future_roa", N_BINS,
               "ESG Score (t)", "Future ROA (t+1)",
               "ESG Score and Future ROA (Controlling for Size, Leverage, Current ROA)",
               OUTPUT_DIR / "part3_fig_esg_roa_resid.png",
               residualize_cols=["roa", "size", "lev"])

# Fig 3 & 4: ESG vs Future Return
ret_ps = df.dropna(subset=["esg_score", "future_return"])
binned_scatter(ret_ps, "esg_score", "future_return", N_BINS,
               "ESG Score (t)", "Future 12-Month Return (t+1)",
               "ESG Score and Future Stock Return",
               OUTPUT_DIR / "part3_fig_esg_ret.png")

binned_scatter(ret_ps.dropna(subset=["roa", "log_me", "lev"]),
               "esg_score", "future_return", N_BINS,
               "ESG Score (t)", "Future 12-Month Return (t+1)",
               "ESG Score and Future Return (Controlling for Log(ME), Leverage, ROA)",
               OUTPUT_DIR / "part3_fig_esg_ret_resid.png",
               residualize_cols=["roa", "log_me", "lev"])

# Fig 5: Tercile bar chart
tercile_bar = df.dropna(subset=["esg_tercile", "future_return", "future_roa"])
tercile_means = (
    tercile_bar.groupby("esg_tercile", observed=True)
    .agg(Return=("future_return", "mean"), ROA=("future_roa", "mean"))
)
for label in ["Low", "Mid", "High"]:
    if label not in tercile_means.index:
        tercile_means.loc[label] = np.nan
tercile_means = tercile_means.loc[["Low", "Mid", "High"]]

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
colors = ["#D9534F", "#F0AD4E", "#5CB85C"]
axes[0].bar(tercile_means.index, tercile_means["Return"], color=colors, edgecolor="white")
axes[0].set_ylabel("Mean Future 12-Month Return")
axes[0].set_title("Stock Return by ESG Tercile", fontweight="bold")
axes[0].yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=1))

axes[1].bar(tercile_means.index, tercile_means["ROA"], color=colors, edgecolor="white")
axes[1].set_ylabel("Mean Future ROA")
axes[1].set_title("Future ROA by ESG Tercile", fontweight="bold")
axes[1].yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=1))

for ax in axes:
    ax.set_xlabel("ESG Tercile")
    ax.grid(axis="y", alpha=0.25)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "part3_fig_tercile.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {OUTPUT_DIR / 'part3_fig_tercile.png'}")


# =============================================================================
# 11. SAVE ALL OUTPUT FILES
# =============================================================================
print("\n" + "=" * 70)
print("SAVING OUTPUT FILES")
print("=" * 70)

if roa_results:
    results_to_csv(roa_results, OUTPUT_DIR / "part3_roa_regression.csv")
if ret_results:
    results_to_csv(ret_results, OUTPUT_DIR / "part3_ret_regression.csv")

all_rob = [r for r in roa_rob_results + ret_rob_results if r]
if all_rob:
    results_to_csv(all_rob, OUTPUT_DIR / "part3_robustness.csv")

with open(OUTPUT_DIR / "part3_tables.txt", "w") as f:
    f.write("RSM8224 Group Project – Part III Regression Tables\n")
    f.write("=" * 80 + "\n\n")
    f.write(roa_table + "\n\n")
    f.write(ret_table + "\n\n")
    f.write(rob_table_a + "\n\n")
    f.write(rob_table_b + "\n\n")
    f.write("\n--- Notes ---\n")
    f.write("'earn_chg' = ROA(t) - ROA(t-1): backward-looking earnings momentum\n")
    f.write("  available to investors at portfolio formation. Only computed for\n")
    f.write("  consecutive fiscal years to avoid gap-year contamination.\n")
    f.write(f"B/M definition: {bm_label}\n")
    f.write("  Market equity approximated using December (or last available month)\n")
    f.write("  of the fiscal year. For non-December fiscal year ends, this\n")
    f.write("  introduces a minor timing mismatch.\n")
    f.write("ROA regressions use size = log(total assets).\n")
    f.write("Return regressions use log_me = log(market equity) as the size\n")
    f.write("  control, following asset pricing conventions.\n")
print(f"  Saved: {OUTPUT_DIR / 'part3_tables.txt'}")


# =============================================================================
# 12. INTERPRETATION GUIDE
# =============================================================================
print("\n" + "=" * 70)
print("INTERPRETATION GUIDE FOR THE REPORT")
print("=" * 70)

print("""
FRAMING YOUR FINDINGS UNDER DOUBLE MATERIALITY:

FINANCIAL MATERIALITY (ESG -> Firm value):
  - Table 1: If ESG predicts future ROA after controls + FE, ESG factors
    are financially material — they relate to real operating performance.
  - Table 2: If ESG predicts returns unconditionally (Models 1-2) but NOT
    after controls (Models 3-6), markets efficiently price ESG info.
    If ESG STILL predicts returns after controls, either:
      (a) ESG captures risk not spanned by standard controls, or
      (b) the market underprices ESG information.
    If ESG predicts returns NEGATIVELY after controls, high-ESG firms earn
    lower expected returns — consistent with a "greenium" (Pastor et al. 2022).

IMPACT MATERIALITY (Firm -> Society):
  - High ESG reflects management of externalities, regardless of whether
    it feeds back into financial returns.
  - If high-ESG firms earn SIMILAR returns (no penalty), investors can
    "do good without doing badly" — ESG alignment is costless.
  - If there IS a return penalty, tension exists between values and value
    that investors must navigate (Starks, 2023).

EARNINGS NEWS CONTROL:
  - 'earn_chg' = ROA(t) - ROA(t-1) captures backward-looking earnings
    momentum available at portfolio formation.
  - If ESG remains significant after controlling for earn_chg, ESG
    contains return-relevant information beyond recent earnings trends.
  - Limitation: this is not a true unexpected earnings measure (SUE).
    We lack analyst forecast data to construct precise earnings surprises.

KEY REFERENCES:
  - Khan, Serafeim & Yoon (2016): materiality matters for returns
  - Bolton & Kacperczyk (2021): carbon risk is priced
  - Pastor, Stambaugh & Taylor (2022): green assets earn lower returns
  - Starks (2023): values vs. value distinction
  - Serafeim & Yoon (2023): ESG news and stock price reactions
""")

print("=" * 70)
print("PART III ANALYSIS COMPLETE")
print("=" * 70)
