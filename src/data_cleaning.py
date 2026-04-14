"""
RSM8224 Group Project – Data Cleaning & Panel Construction
===========================================================
ESG Performance and Financial Returns (Fiscal Years 2013–2023)

Data sources (all from WRDS):
  - Compustat Annual (compustat_annual.csv)
  - CRSP Monthly   (crsp_monthly.csv)
  - CCM Link Table (ccm_link.csv)
  - LSEG/Refinitiv ESG (ESG.csv)

This script produces a firm-year panel with:
  - ESG scores (overall composite)
  - Accounting variables (ROA, leverage, size, etc.)
  - Forward 12-month stock returns (compounded from CRSP monthly)
  - Forward ROA (strict t+1 fiscal-year match)
  - Industry classifications (SIC-2 and Fama-French 12)
"""

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# 0. PATHS & SETTINGS
# =============================================================================
BASE = Path(r"./data")  # <-- change to your local data directory

COMP_PATH = BASE / "compustat_annual.csv"
CCM_PATH  = BASE / "ccm_link.csv"
CRSP_PATH = BASE / "crsp_monthly.csv"
ESG_PATH  = BASE / "ESG.csv"

OUTPUT_FINAL  = BASE / "final_panel.csv"
OUTPUT_MASTER = BASE / "master_panel_with_intermediates.csv"

# Winsorization settings
WINSORIZE_BY_YEAR = True   # True = winsorize within each fiscal year (recommended)
WINSOR_LOWER = 0.01
WINSOR_UPPER = 0.99

# If True, raise an error when CRSP lacks shrcd/exchcd columns
STRICT_CRSP_FILTER_CHECK = True

# ESG score column candidates (checked left to right; first match wins)
ESG_SCORE_CANDIDATES = [
    "esg_score", "tr_tresgscore", "tresgscore", "esgscore", "valuescore"
]

# Industry grouping: "sic2" for 2-digit SIC, "ff12" for Fama-French 12
INDUSTRY_MODE = "sic2"


# =============================================================================
# 1. HELPER FUNCTIONS
# =============================================================================
def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lower-case and strip whitespace from all column names."""
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def require_columns(df: pd.DataFrame, required: list[str], df_name: str) -> None:
    """Raise ValueError if any required columns are missing."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{df_name} missing columns: {missing}")


def clean_cusip8(s: pd.Series) -> pd.Series:
    """Standardize CUSIP to upper-case 8-character alphanumeric string."""
    return (
        s.astype(str)
         .str.strip()
         .str.upper()
         .str.replace(r"[^A-Z0-9]", "", regex=True)
         .str[:8]
    )


def winsorize_series(
    s: pd.Series,
    lower: float = 0.01,
    upper: float = 0.99
) -> pd.Series:
    """Winsorize a numeric series at the given percentiles."""
    non_na = s.dropna()
    if non_na.empty:
        return s
    lo = non_na.quantile(lower)
    hi = non_na.quantile(upper)
    return s.clip(lower=lo, upper=hi)


def winsorize_columns(
    df: pd.DataFrame,
    cols: list[str],
    group_col: str | None = None,
    lower: float = 0.01,
    upper: float = 0.99
) -> pd.DataFrame:
    """
    Winsorize selected columns, optionally within groups (e.g. by year).
    Skips columns not present in the DataFrame.
    """
    df = df.copy()
    for col in cols:
        if col not in df.columns:
            continue
        if group_col is None:
            df[col] = winsorize_series(df[col], lower, upper)
        else:
            df[col] = df.groupby(group_col)[col].transform(
                lambda x: winsorize_series(x, lower, upper)
            )
    return df


def detect_esg_score_column(esg: pd.DataFrame) -> str:
    """
    Detect the overall ESG score column from a list of candidates.
    Prints a warning if 'valuescore' is used (requires manual verification).
    """
    found = [c for c in ESG_SCORE_CANDIDATES if c in esg.columns]
    if not found:
        raise ValueError(
            f"No ESG score column detected. Checked: {ESG_SCORE_CANDIDATES}. "
            "Please verify your WRDS export and update ESG_SCORE_CANDIDATES."
        )
    chosen = found[0]
    print(f"[INFO] Using ESG score column: '{chosen}'")
    if chosen == "valuescore":
        warnings.warn(
            "Using 'valuescore' as the overall ESG score. Please confirm in "
            "your WRDS export that this corresponds to the overall ESG Score "
            "(TR.TRESGScore), not a sub-pillar or controversy score."
        )
    return chosen


# -------------------------------------------------------------------------
# Industry classification helpers
# -------------------------------------------------------------------------
def make_sic2(sic: pd.Series) -> pd.Series:
    """Convert raw SIC codes to 2-digit industry groups."""
    sic_num = pd.to_numeric(sic, errors="coerce")
    return (sic_num // 100).astype("Int64")


def fama_french_12(sic: pd.Series) -> pd.Series:
    """
    Map 4-digit SIC codes to Fama-French 12 industry groups.
    Based on Ken French's specification:
    https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/det_12_ind_port.html
    """
    sic = pd.to_numeric(sic, errors="coerce")
    out = pd.Series("Other", index=sic.index, dtype="object")

    # 1 - NoDur: Consumer NonDurables
    out[((sic >= 100) & (sic <= 999)) |
        ((sic >= 2000) & (sic <= 2399)) |
        ((sic >= 2700) & (sic <= 2749)) |
        ((sic >= 2770) & (sic <= 2799)) |
        ((sic >= 3100) & (sic <= 3199)) |
        ((sic >= 3940) & (sic <= 3989))] = "NoDur"

    # 2 - Durbl: Consumer Durables
    out[((sic >= 2500) & (sic <= 2519)) |
        ((sic >= 2590) & (sic <= 2599)) |
        ((sic >= 3630) & (sic <= 3659)) |
        ((sic >= 3710) & (sic <= 3711)) |
        ((sic >= 3714) & (sic <= 3714)) |
        ((sic >= 3716) & (sic <= 3716)) |
        ((sic >= 3750) & (sic <= 3751)) |
        ((sic >= 3792) & (sic <= 3792)) |
        ((sic >= 3900) & (sic <= 3939)) |
        ((sic >= 3990) & (sic <= 3999))] = "Durbl"

    # 3 - Manuf: Manufacturing
    out[((sic >= 2520) & (sic <= 2589)) |
        ((sic >= 2600) & (sic <= 2699)) |
        ((sic >= 2750) & (sic <= 2769)) |
        ((sic >= 2800) & (sic <= 2829)) |
        ((sic >= 2840) & (sic <= 2899)) |
        ((sic >= 3000) & (sic <= 3099)) |
        ((sic >= 3200) & (sic <= 3569)) |
        ((sic >= 3580) & (sic <= 3629)) |
        ((sic >= 3660) & (sic <= 3692)) |
        ((sic >= 3694) & (sic <= 3699)) |
        ((sic >= 3712) & (sic <= 3713)) |
        ((sic >= 3715) & (sic <= 3715)) |
        ((sic >= 3717) & (sic <= 3749)) |
        ((sic >= 3752) & (sic <= 3791)) |
        ((sic >= 3793) & (sic <= 3799)) |
        ((sic >= 3830) & (sic <= 3839)) |
        ((sic >= 3860) & (sic <= 3899))] = "Manuf"

    # 4 - Enrgy: Oil, Gas, and Coal Extraction and Products
    out[((sic >= 1200) & (sic <= 1399)) |
        ((sic >= 2900) & (sic <= 2999))] = "Enrgy"

    # 5 - Chems: Chemicals and Allied Products
    out[((sic >= 2800) & (sic <= 2829)) |
        ((sic >= 2840) & (sic <= 2899))] = "Chems"

    # 6 - BusEq: Business Equipment (Computers, Software, Electronics)
    out[((sic >= 3570) & (sic <= 3579)) |
        ((sic >= 3660) & (sic <= 3692)) |
        ((sic >= 3694) & (sic <= 3699)) |
        ((sic >= 3810) & (sic <= 3829)) |
        ((sic >= 7370) & (sic <= 7379))] = "BusEq"

    # 7 - Telcm: Telephone and Television Transmission
    out[((sic >= 4800) & (sic <= 4899))] = "Telcm"

    # 8 - Utils: Utilities
    out[((sic >= 4900) & (sic <= 4949))] = "Utils"

    # 9 - Shops: Wholesale, Retail, and Some Services
    out[((sic >= 5000) & (sic <= 5999)) |
        ((sic >= 7200) & (sic <= 7299)) |
        ((sic >= 7600) & (sic <= 7699))] = "Shops"

    # 10 - Hlth: Healthcare, Medical Equipment, and Drugs
    out[((sic >= 2830) & (sic <= 2839)) |
        ((sic >= 3693) & (sic <= 3693)) |
        ((sic >= 3840) & (sic <= 3859)) |
        ((sic >= 8000) & (sic <= 8099))] = "Hlth"

    # 11 - Money: Finance
    out[((sic >= 6000) & (sic <= 6999))] = "Money"

    # 12 - Other: everything not classified above (already default)

    return out


def add_industry_variables(df: pd.DataFrame, mode: str = "sic2") -> pd.DataFrame:
    """Add SIC-2 and an industry_group column to the panel."""
    df = df.copy()
    df["sic2"] = make_sic2(df["sic"])

    if mode == "ff12":
        df["industry_group"] = fama_french_12(df["sic"])
    else:
        df["industry_group"] = df["sic2"].astype("string")

    return df


# -------------------------------------------------------------------------
# Forward 12-month return computation (memory-friendly)
# -------------------------------------------------------------------------
def compute_forward_12m_return(
    firm_ccm: pd.DataFrame,
    crsp: pd.DataFrame
) -> pd.DataFrame:
    """
    For each firm-year, compound monthly CRSP returns over the 12 months
    following the fiscal year end. Uses a permno-level grouping strategy
    to avoid creating a massive Cartesian-product merge.

    Requires firm_ccm to have: gvkey, permno, datadate, fyear,
    start_month, end_month.
    Requires crsp to have: permno, date, ret.

    Returns rows with n_months >= 12 only.
    """
    results = []

    crsp_sorted = crsp[["permno", "date", "ret"]].sort_values(["permno", "date"]).copy()
    crsp_by_permno = {
        int(p): g[["date", "ret"]].values
        for p, g in crsp_sorted.groupby("permno", sort=False)
        if not pd.isna(p)
    }

    use_cols = ["gvkey", "permno", "datadate", "fyear", "start_month", "end_month"]
    firm_small = firm_ccm[use_cols].copy()

    for permno, g_firm in firm_small.groupby("permno", sort=False):
        if pd.isna(permno):
            continue
        permno_int = int(permno)
        arr = crsp_by_permno.get(permno_int)
        if arr is None:
            continue

        dates_arr = arr[:, 0]  # datetime64 values
        rets_arr  = arr[:, 1].astype(float)

        for row in g_firm.itertuples(index=False):
            start_dt = np.datetime64(row.start_month)
            end_dt   = np.datetime64(row.end_month)

            mask = (dates_arr >= start_dt) & (dates_arr < end_dt)
            sel = rets_arr[mask]
            sel = sel[~np.isnan(sel)]

            n_months = len(sel)
            future_return = float(np.prod(1.0 + sel) - 1.0) if n_months > 0 else np.nan

            results.append({
                "gvkey":         row.gvkey,
                "permno":        row.permno,
                "datadate":      row.datadate,
                "fyear":         row.fyear,
                "n_months":      n_months,
                "future_return": future_return,
            })

    out = pd.DataFrame(results)
    if out.empty:
        return out

    # Require at least 12 months of valid returns
    out = out[out["n_months"] >= 12].copy()
    return out


# =============================================================================
# 2. LOAD RAW DATA
# =============================================================================
print("=" * 60)
print("LOADING RAW DATA")
print("=" * 60)

comp = standardize_columns(pd.read_csv(COMP_PATH, low_memory=False))
ccm  = standardize_columns(pd.read_csv(CCM_PATH,  low_memory=False))
crsp = standardize_columns(pd.read_csv(CRSP_PATH, low_memory=False))
esg  = standardize_columns(pd.read_csv(ESG_PATH,  low_memory=False, encoding="latin1"))

print(f"  Compustat : {comp.shape}")
print(f"  CCM       : {ccm.shape}")
print(f"  CRSP      : {crsp.shape}")
print(f"  ESG       : {esg.shape}")


# =============================================================================
# 3. CLEAN COMPUSTAT
# =============================================================================
print("\n" + "=" * 60)
print("CLEANING COMPUSTAT")
print("=" * 60)

required_comp_cols = [
    "gvkey", "datadate", "cusip", "sic", "fyear",
    "at", "ni", "sale", "ceq", "dltt", "dlc"
]
require_columns(comp, required_comp_cols, "Compustat")

# Apply standard Compustat industrial format filters
for col, val in [("indfmt", "INDL"), ("datafmt", "STD"),
                 ("consol", "C"), ("popsrc", "D")]:
    if col in comp.columns:
        comp = comp[comp[col] == val]

comp["datadate"] = pd.to_datetime(comp["datadate"], errors="coerce")

for col in ["fyear", "sic", "at", "ni", "sale", "ceq", "dltt", "dlc"]:
    comp[col] = pd.to_numeric(comp[col], errors="coerce")

# Restrict to fiscal years 2013–2023
comp = comp[(comp["fyear"] >= 2013) & (comp["fyear"] <= 2023)].copy()

# Standardize CUSIP to 8 characters
comp["cusip8"] = clean_cusip8(comp["cusip"])

# Require positive total assets
comp = comp[comp["at"] > 0].copy()

# Fill missing debt with zero before computing leverage
comp["dltt"] = comp["dltt"].fillna(0)
comp["dlc"]  = comp["dlc"].fillna(0)

# Construct accounting variables
comp["roa"]  = comp["ni"] / comp["at"]
comp["lev"]  = (comp["dltt"] + comp["dlc"]) / comp["at"]
comp["size"] = np.log(comp["at"])

# Deduplicate: keep the latest datadate per firm-year
comp = comp.sort_values(["gvkey", "fyear", "datadate"])
comp = comp.drop_duplicates(subset=["gvkey", "fyear"], keep="last")

print(f"  After cleaning: {comp.shape}")


# =============================================================================
# 4. CLEAN ESG
# =============================================================================
print("\n" + "=" * 60)
print("CLEANING ESG")
print("=" * 60)

require_columns(esg, ["year", "isin"], "ESG")

# Detect the correct overall ESG score column
esg_score_col = detect_esg_score_column(esg)

esg["year"] = pd.to_numeric(esg["year"], errors="coerce")
esg = esg[(esg["year"] >= 2013) & (esg["year"] <= 2023)].copy()

esg["esg_score"] = pd.to_numeric(esg[esg_score_col], errors="coerce")

# Extract US-listed firms via ISIN prefix, then derive 8-digit CUSIP
# US ISIN format: "US" + 9-digit CUSIP + 1 check digit
# Characters 3–10 of ISIN = first 8 digits of CUSIP (matches Compustat)
esg["isin"] = esg["isin"].astype(str).str.strip().str.upper()
esg = esg[esg["isin"].str.startswith("US", na=False)].copy()
esg["cusip8"] = clean_cusip8(esg["isin"].str[2:10])

# Rename year to fyear for consistent firm-year merging
esg = esg.rename(columns={"year": "fyear"})

# Keep only essential columns (plus company name if available)
keep_cols = ["fyear", "cusip8", "esg_score"]
for name_col in ["comname", "companyname", "name"]:
    if name_col in esg.columns:
        keep_cols.append(name_col)
        break

esg = esg[keep_cols].copy()
esg = esg.dropna(subset=["fyear", "cusip8", "esg_score"])

# Report and handle duplicates
dup_count = esg.duplicated(subset=["cusip8", "fyear"], keep=False).sum()
print(f"  Duplicate (cusip8, fyear) rows before dedup: {dup_count}")

# Keep the last entry per firm-year (typically the most recent update)
esg = esg.sort_values(["cusip8", "fyear"])
esg = esg.drop_duplicates(subset=["cusip8", "fyear"], keep="last")

print(f"  After cleaning: {esg.shape}")


# =============================================================================
# 5. MERGE ESG TO COMPUSTAT (on cusip8 + fyear)
# =============================================================================
print("\n" + "=" * 60)
print("MERGING ESG TO COMPUSTAT")
print("=" * 60)

firm = pd.merge(
    comp, esg,
    on=["cusip8", "fyear"],
    how="inner",
    validate="m:1"
)

print(f"  After merge: {firm.shape}")


# =============================================================================
# 6. CLEAN CCM LINK TABLE
# =============================================================================
print("\n" + "=" * 60)
print("CLEANING CCM LINK TABLE")
print("=" * 60)

required_ccm_cols = ["gvkey", "lpermno", "linktype", "linkprim", "linkdt", "linkenddt"]
require_columns(ccm, required_ccm_cols, "CCM")

ccm["linkdt"]    = pd.to_datetime(ccm["linkdt"], errors="coerce")
ccm["linkenddt"] = pd.to_datetime(ccm["linkenddt"], errors="coerce")

# Open-ended links: fill missing end date with a far-future date
ccm["linkenddt"] = ccm["linkenddt"].fillna(pd.Timestamp("2099-12-31"))

# Keep only valid link types as specified in the assignment
ccm = ccm[
    ccm["linktype"].isin(["LC", "LU", "LS"]) &
    ccm["linkprim"].isin(["C", "P"])
].copy()

ccm = ccm.rename(columns={"lpermno": "permno"})
ccm["permno"] = pd.to_numeric(ccm["permno"], errors="coerce")
ccm = ccm.sort_values(["gvkey", "linkdt", "linkenddt", "permno"])

print(f"  After cleaning: {ccm.shape}")


# =============================================================================
# 7. MERGE FIRM-YEAR WITH CCM (gvkey + date-window validation)
# =============================================================================
print("\n" + "=" * 60)
print("MERGING FIRM DATA WITH CCM")
print("=" * 60)

firm_ccm = pd.merge(
    firm,
    ccm[["gvkey", "permno", "linktype", "linkprim", "linkdt", "linkenddt"]],
    on="gvkey",
    how="left"
)

# Keep only rows where the fiscal year end falls within the valid link window
firm_ccm = firm_ccm[
    (firm_ccm["datadate"] >= firm_ccm["linkdt"]) &
    (firm_ccm["datadate"] <= firm_ccm["linkenddt"])
].copy()

# If multiple permnos remain, prefer primary links, then keep first permno
firm_ccm = firm_ccm.sort_values(["gvkey", "fyear", "linkprim", "permno"])
firm_ccm = firm_ccm.drop_duplicates(subset=["gvkey", "fyear"], keep="first")

print(f"  After merge: {firm_ccm.shape}")


# =============================================================================
# 8. CLEAN CRSP MONTHLY
# =============================================================================
print("\n" + "=" * 60)
print("CLEANING CRSP MONTHLY")
print("=" * 60)

# Rename CIZ-style columns to standard names
rename_map = {"mthcaldt": "date", "mthret": "ret", "mthprc": "prc"}
crsp = crsp.rename(columns=rename_map)

required_crsp_cols = ["permno", "date", "ret"]
require_columns(crsp, required_crsp_cols, "CRSP")

crsp["date"]   = pd.to_datetime(crsp["date"], errors="coerce")
crsp["permno"] = pd.to_numeric(crsp["permno"], errors="coerce")
crsp["ret"]    = pd.to_numeric(crsp["ret"], errors="coerce")

# Keep 2013-01 through 2024-12 to cover 12-month windows after 2023 fiscal year ends
crsp = crsp[(crsp["date"] >= "2013-01-01") & (crsp["date"] <= "2024-12-31")].copy()

# Apply domestic common stock filters as required by the assignment
if "shrcd" not in crsp.columns or "exchcd" not in crsp.columns:
    msg = (
        "CRSP does not contain 'shrcd' and/or 'exchcd'. "
        "The assignment requires shrcd IN (10,11) and exchcd IN (1,2,3). "
        "Please re-download CRSP with these fields from WRDS."
    )
    if STRICT_CRSP_FILTER_CHECK:
        raise ValueError(msg)
    else:
        warnings.warn(msg)
else:
    crsp["shrcd"]  = pd.to_numeric(crsp["shrcd"],  errors="coerce")
    crsp["exchcd"] = pd.to_numeric(crsp["exchcd"], errors="coerce")
    crsp = crsp[
        crsp["shrcd"].isin([10, 11]) &
        crsp["exchcd"].isin([1, 2, 3])
    ].copy()

# Deduplicate: keep last record per permno-month
crsp = crsp.sort_values(["permno", "date"])
crsp = crsp.drop_duplicates(subset=["permno", "date"], keep="last")

print(f"  After cleaning: {crsp.shape}")


# =============================================================================
# 9. BUILD FORWARD 12-MONTH COMPOUNDED RETURN
# =============================================================================
print("\n" + "=" * 60)
print("COMPUTING FORWARD 12-MONTH RETURNS")
print("=" * 60)

# Define the return accumulation window:
#   start_month = first day of the month after fiscal year end
#   end_month   = 12 months after start_month (exclusive upper bound)
firm_ccm["start_month"] = (
    firm_ccm["datadate"].dt.to_period("M") + 1
).dt.to_timestamp()
firm_ccm["end_month"] = (
    firm_ccm["datadate"].dt.to_period("M") + 13
).dt.to_timestamp()

future_ret = compute_forward_12m_return(firm_ccm, crsp)

if future_ret.empty:
    raise ValueError(
        "No valid forward 12-month returns were computed. "
        "Check that CRSP data covers the period after your fiscal year ends."
    )

print(f"  Firm-years with valid 12-month returns: {future_ret.shape[0]}")

# Merge forward returns back to the firm-year panel
panel = pd.merge(
    firm_ccm,
    future_ret[["gvkey", "permno", "datadate", "fyear", "n_months", "future_return"]],
    on=["gvkey", "permno", "datadate", "fyear"],
    how="inner",
    validate="1:1"
)

print(f"  Panel after adding forward returns: {panel.shape}")


# =============================================================================
# 10. BUILD FORWARD ROA (strict t+1 fiscal year match)
# =============================================================================
print("\n" + "=" * 60)
print("COMPUTING FORWARD ROA")
print("=" * 60)

# Create a lookup: for each firm, the ROA observed in fyear Y
# is assigned as the "future_roa" for fyear Y-1
roa_lookup = panel[["gvkey", "fyear", "roa"]].copy()
roa_lookup["fyear"] = roa_lookup["fyear"] - 1
roa_lookup = roa_lookup.rename(columns={"roa": "future_roa"})

panel = pd.merge(
    panel,
    roa_lookup,
    on=["gvkey", "fyear"],
    how="left",
    validate="m:1"
)

n_future_roa = panel["future_roa"].notna().sum()
print(f"  Observations with non-missing future ROA: {n_future_roa}")


# =============================================================================
# 11. ADD INDUSTRY CLASSIFICATION VARIABLES
# =============================================================================
print("\n" + "=" * 60)
print("ADDING INDUSTRY VARIABLES")
print("=" * 60)

panel = add_industry_variables(panel, mode=INDUSTRY_MODE)

n_industries = panel["industry_group"].nunique()
print(f"  Industry grouping mode: {INDUSTRY_MODE}")
print(f"  Number of distinct industry groups: {n_industries}")


# =============================================================================
# 12. FINAL SAMPLE RESTRICTIONS
# =============================================================================
print("\n" + "=" * 60)
print("APPLYING FINAL SAMPLE RESTRICTIONS")
print("=" * 60)

# Ensure fiscal year range
panel = panel[(panel["fyear"] >= 2013) & (panel["fyear"] <= 2023)].copy()

# Master panel requires: ESG score, forward 12m return, positive assets
panel = panel.dropna(subset=["esg_score", "future_return", "at"]).copy()

# Deduplicate at firm-year level (keep latest datadate if ties exist)
panel = panel.sort_values(["gvkey", "fyear", "datadate"])
panel = panel.drop_duplicates(subset=["gvkey", "fyear"], keep="last").copy()

print(f"  Panel after restrictions: {panel.shape}")


# =============================================================================
# 13. WINSORIZE CONTINUOUS VARIABLES
# =============================================================================
print("\n" + "=" * 60)
print("WINSORIZING")
print("=" * 60)

winsor_cols = ["esg_score", "roa", "future_roa", "future_return", "lev", "size"]
winsor_cols = [c for c in winsor_cols if c in panel.columns]

group_col = "fyear" if WINSORIZE_BY_YEAR else None
panel = winsorize_columns(
    panel,
    cols=winsor_cols,
    group_col=group_col,
    lower=WINSOR_LOWER,
    upper=WINSOR_UPPER,
)

print(f"  Winsorized columns: {winsor_cols}")
print(f"  By year: {WINSORIZE_BY_YEAR}")


# =============================================================================
# 14. SELECT FINAL COLUMNS & SAVE
# =============================================================================
print("\n" + "=" * 60)
print("SAVING OUTPUT")
print("=" * 60)

final_cols = [
    # Identifiers
    "gvkey", "permno", "datadate", "fyear",
    "cusip8", "sic", "sic2", "industry_group",
    # ESG
    "esg_score",
    # Accounting (raw)
    "at", "ni", "sale", "ceq", "dltt", "dlc",
    # Accounting (constructed)
    "size", "lev", "roa",
    # Forward-looking outcomes
    "future_roa", "future_return",
    # Metadata
    "n_months",
]
final_cols = [c for c in final_cols if c in panel.columns]

final_panel = panel[final_cols].copy()

# One last dedup safety check
final_panel = final_panel.sort_values(["gvkey", "fyear", "datadate"])
final_panel = final_panel.drop_duplicates(subset=["gvkey", "fyear"], keep="last")

# Save
OUTPUT_FINAL.parent.mkdir(parents=True, exist_ok=True)

final_panel.to_csv(OUTPUT_FINAL, index=False)
panel.to_csv(OUTPUT_MASTER, index=False)

print(f"  Final panel shape: {final_panel.shape}")
print(f"  Saved final panel to:  {OUTPUT_FINAL}")
print(f"  Saved master panel to: {OUTPUT_MASTER}")


# =============================================================================
# 15. SUMMARY DIAGNOSTICS
# =============================================================================
print("\n" + "=" * 60)
print("SUMMARY DIAGNOSTICS")
print("=" * 60)

print("\nMissing values in final panel:")
missing = final_panel.isna().sum()
print(missing[missing > 0].sort_values(ascending=False))
if missing.sum() == 0:
    print("  (none)")

print(f"\nFiscal year coverage: {int(final_panel['fyear'].min())} – {int(final_panel['fyear'].max())}")
print(f"Unique firms (gvkey): {final_panel['gvkey'].nunique()}")
print(f"Total firm-years:     {len(final_panel)}")

print("\nObservations per fiscal year:")
print(final_panel.groupby("fyear").size().to_string())

print("\nDescriptive statistics (key variables):")
desc_cols = ["esg_score", "roa", "future_roa", "future_return", "lev", "size"]
desc_cols = [c for c in desc_cols if c in final_panel.columns]
print(final_panel[desc_cols].describe().round(4).to_string())

print("\nPreview (first 5 rows):")
print(final_panel.head().to_string())

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)
