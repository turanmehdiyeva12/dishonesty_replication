from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================
# SETTINGS
# ============================================================
try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()

DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

PATH = DATA_DIR / "FINAL_REPOSITORY_MAIN_FEB2023.dta"

OUTCOME = "empl_public100"
ENDOG = "failed_exam100"
INSTR = "dist_cutoff_left1000"
COHORT = "c_clase"
CLUSTER = "cohort_id"
DRAFT = "draft_exempt"

REGION_DUMMIES = ["dreg1", "dreg2", "dreg3", "dreg4", "dreg5", "dreg6"]
REGION_MAP = {
    "dreg1": "Cuyo",
    "dreg2": "GBA",
    "dreg3": "NEA",
    "dreg4": "NOA",
    "dreg5": "Pampeana",
    "dreg6": "Patagonia",
}

REFERENCE_REGION = "Pampeana"
MIN_OBS_PER_REGION = 100
MIN_CLUSTERS_PER_REGION = 50
WEAK_IV_F_THRESHOLD = 10.0

# Candidate variables for the summary table.
# Variables that do not exist in your file will be skipped automatically.
SUMMARY_VARS = [
    ("draft_exempt", "Draft exempt"),
    ("failed_exam100", "Failed medical examination"),
    ("empl_public100", "Public sector employee"),
    ("empl_private100", "Private sector employee"),
    ("empl_public_merit100", "Meritocratic public employee"),
    ("empl_public_nonmerit100", "Non-meritocratic public employee"),
    ("attriter", "Attriter"),
]

# ============================================================
# HELPERS
# ============================================================
def build_region(df: pd.DataFrame) -> pd.Series:
    region_matrix = df[REGION_DUMMIES].fillna(0).to_numpy()
    sums = region_matrix.sum(axis=1)

    region_names = np.array([REGION_MAP[c] for c in REGION_DUMMIES], dtype=object)
    region = np.full(len(df), np.nan, dtype=object)

    valid = sums == 1
    if not np.all(valid):
        bad = int((~valid).sum())
        print(f"Warning: {bad} rows do not have exactly one region dummy = 1. They will be dropped.")

    idx = region_matrix.argmax(axis=1)
    region[valid] = region_names[idx[valid]]
    return pd.Series(region, index=df.index, name="region")


def cohort_fe(df: pd.DataFrame) -> pd.DataFrame:
    return pd.get_dummies(df[COHORT].astype("category"), drop_first=True).astype(float)


def region_cohort_fe(df: pd.DataFrame) -> pd.DataFrame:
    cell = df["region"].astype(str) + "_c" + df[COHORT].astype(int).astype(str)
    return pd.get_dummies(pd.Categorical(cell), drop_first=True).astype(float)


def iv_2sls_cluster(y, w, d, q, clusters):
    y_arr = np.asarray(y, dtype=float).reshape(-1, 1)
    w_arr = np.asarray(w, dtype=float)
    d_arr = np.asarray(d, dtype=float)
    q_arr = np.asarray(q, dtype=float)

    if d_arr.ndim == 1:
        d_arr = d_arr.reshape(-1, 1)
    if q_arr.ndim == 1:
        q_arr = q_arr.reshape(-1, 1)

    x_arr = np.column_stack([w_arr, d_arr])
    z_arr = np.column_stack([w_arr, q_arr])

    zz_inv = np.linalg.pinv(z_arr.T @ z_arr)
    bread_inner = x_arr.T @ z_arr @ zz_inv @ z_arr.T @ x_arr
    bread = np.linalg.pinv(bread_inner)

    beta = bread @ (x_arr.T @ z_arr @ zz_inv @ z_arr.T @ y_arr)
    resid = (y_arr - x_arr @ beta).reshape(-1)

    zu = pd.DataFrame(z_arr * resid[:, None], columns=[f"z{i}" for i in range(z_arr.shape[1])])
    zu["cluster"] = np.asarray(clusters)
    gsum = zu.groupby("cluster", sort=False).sum(numeric_only=True).to_numpy()
    meat = gsum.T @ gsum

    cov = bread @ (x_arr.T @ z_arr @ zz_inv @ meat @ zz_inv @ z_arr.T @ x_arr) @ bread

    n_obs = x_arr.shape[0]
    n_params = x_arr.shape[1]
    n_clusters = gsum.shape[0]
    if n_clusters > 1 and n_obs > n_params:
        small_sample = (n_clusters / (n_clusters - 1)) * ((n_obs - 1) / (n_obs - n_params))
        cov *= small_sample

    se = np.sqrt(np.diag(cov))
    param_names = list(w.columns) + list(d.columns)

    return {
        "coef": pd.Series(beta.flatten(), index=param_names, name="coef"),
        "se": pd.Series(se, index=param_names, name="se"),
        "cov": pd.DataFrame(cov, index=param_names, columns=param_names),
        "nobs": int(n_obs),
        "nclusters": int(n_clusters),
        "resid": resid,
    }


def first_stage_ols(df: pd.DataFrame):
    x = sm.add_constant(pd.concat([df[[INSTR]].astype(float), cohort_fe(df)], axis=1), has_constant="add")
    return sm.OLS(df[ENDOG].astype(float), x).fit(cov_type="cluster", cov_kwds={"groups": df[CLUSTER]})


def reduced_form_ols(df: pd.DataFrame):
    x = sm.add_constant(pd.concat([df[[INSTR]].astype(float), cohort_fe(df)], axis=1), has_constant="add")
    return sm.OLS(df[OUTCOME].astype(float), x).fit(cov_type="cluster", cov_kwds={"groups": df[CLUSTER]})


def ppoints(x: float) -> float:
    return 100.0 * x


def summarize_data(raw_df: pd.DataFrame, analysis_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for var, label in SUMMARY_VARS:
        if var not in raw_df.columns:
            continue

        full = pd.to_numeric(raw_df[var], errors="coerce")
        samp = pd.to_numeric(analysis_df[var], errors="coerce")

        rows.append({
            "Variable": label,
            "Full male cohorts": full.mean() * 100 if full.dropna().isin([0, 1]).all() else full.mean(),
            "Draft-exempt analysis sample": samp.mean() * 100 if samp.dropna().isin([0, 1]).all() else samp.mean(),
        })

    # observations row
    rows.append({
        "Variable": "Observations",
        "Full male cohorts": len(raw_df),
        "Draft-exempt analysis sample": len(analysis_df),
    })

    return pd.DataFrame(rows)


# ============================================================
# LOAD DATA
# ============================================================
if not PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {PATH}")

raw_df = pd.read_stata(PATH, convert_categoricals=False).copy()

required = [OUTCOME, ENDOG, INSTR, COHORT, CLUSTER, DRAFT] + REGION_DUMMIES
missing = [c for c in required if c not in raw_df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# Convert numeric fields
for col in required:
    if col != CLUSTER:
        raw_df[col] = pd.to_numeric(raw_df[col], errors="coerce")

# ============================================================
# SUMMARY STATISTICS TABLE
# ============================================================
summary_table = summarize_data(raw_df, raw_df[raw_df[DRAFT] == 1].copy())
summary_table.to_csv(OUTPUT_DIR / "summary_statistics.csv", index=False)

print("\n=== Summary statistics ===")
print(summary_table)

# ============================================================
# ANALYSIS SAMPLE
# ============================================================
df = raw_df.copy()
df["region"] = build_region(df)

df = df[df[DRAFT] == 1].copy()
df = df.dropna(subset=[OUTCOME, ENDOG, INSTR, COHORT, CLUSTER, "region"]).copy()
df[CLUSTER] = df[CLUSTER].astype(str)

print("\n=== Analysis sample ===")
print("Rows:", len(df))
print("Clusters:", df[CLUSTER].nunique())
print(df["region"].value_counts().sort_index())

# ============================================================
# 0) BASELINE OVERALL MODEL
# ============================================================
base_fs = first_stage_ols(df)
base_rf = reduced_form_ols(df)

base_w = sm.add_constant(cohort_fe(df), has_constant="add")
base_d = df[[ENDOG]].astype(float)
base_q = df[[INSTR]].astype(float)
base_iv = iv_2sls_cluster(df[OUTCOME], base_w, base_d, base_q, df[CLUSTER])

baseline_summary = pd.DataFrame([{
    "sample": "All draft-exempt men",
    "n": base_iv["nobs"],
    "clusters": base_iv["nclusters"],
    "first_stage_coef": base_fs.params[INSTR],
    "first_stage_se": base_fs.bse[INSTR],
    "first_stage_t": base_fs.tvalues[INSTR],
    "first_stage_F_t2": float(base_fs.tvalues[INSTR] ** 2),
    "reduced_form_coef": base_rf.params[INSTR],
    "reduced_form_se": base_rf.bse[INSTR],
    "iv_coef_raw": base_iv["coef"][ENDOG],
    "iv_se_raw": base_iv["se"][ENDOG],
    "iv_coef_pp": ppoints(base_iv["coef"][ENDOG]),
    "iv_se_pp": ppoints(base_iv["se"][ENDOG]),
}])

baseline_summary.to_csv(OUTPUT_DIR / "baseline_iv_summary.csv", index=False)

print("\n=== Baseline IV ===")
print(
    f"2SLS effect = {ppoints(base_iv['coef'][ENDOG]):.2f} pp "
    f"(SE {ppoints(base_iv['se'][ENDOG]):.2f}); "
    f"first-stage t = {base_fs.tvalues[INSTR]:.2f}, "
    f"F ~= {base_fs.tvalues[INSTR] ** 2:.2f}"
)

# ============================================================
# 1) REGION-BY-REGION 2SLS
# ============================================================
region_rows = []

for region in sorted(df["region"].unique()):
    sub = df[df["region"] == region].copy()

    if len(sub) < MIN_OBS_PER_REGION:
        print(f"Skipping {region}: only {len(sub)} observations.")
        continue
    if sub[CLUSTER].nunique() < MIN_CLUSTERS_PER_REGION:
        print(f"Skipping {region}: only {sub[CLUSTER].nunique()} clusters.")
        continue

    fs = first_stage_ols(sub)
    rf = reduced_form_ols(sub)

    w = sm.add_constant(cohort_fe(sub), has_constant="add")
    d = sub[[ENDOG]].astype(float)
    q = sub[[INSTR]].astype(float)
    iv = iv_2sls_cluster(sub[OUTCOME], w, d, q, sub[CLUSTER])

    region_rows.append({
        "region": region,
        "n": iv["nobs"],
        "clusters": iv["nclusters"],
        "first_stage_coef": fs.params[INSTR],
        "first_stage_se": fs.bse[INSTR],
        "first_stage_t": fs.tvalues[INSTR],
        "first_stage_F_t2": float(fs.tvalues[INSTR] ** 2),
        "weak_iv_flag_F_lt_10": bool((fs.tvalues[INSTR] ** 2) < WEAK_IV_F_THRESHOLD),
        "reduced_form_coef": rf.params[INSTR],
        "reduced_form_se": rf.bse[INSTR],
        "iv_coef_raw": iv["coef"][ENDOG],
        "iv_se_raw": iv["se"][ENDOG],
        "iv_coef_pp": ppoints(iv["coef"][ENDOG]),
        "iv_se_pp": ppoints(iv["se"][ENDOG]),
        "ci_low_pp": ppoints(iv["coef"][ENDOG] - 1.96 * iv["se"][ENDOG]),
        "ci_high_pp": ppoints(iv["coef"][ENDOG] + 1.96 * iv["se"][ENDOG]),
    })

region_table = pd.DataFrame(region_rows).sort_values("iv_coef_pp").reset_index(drop=True)
region_table.to_csv(OUTPUT_DIR / "region_specific_iv_results.csv", index=False)

print("\n=== Region-specific IV results (pp) ===")
print(region_table[["region", "n", "clusters", "first_stage_F_t2", "weak_iv_flag_F_lt_10", "iv_coef_pp", "iv_se_pp"]])

# ============================================================
# 2) POOLED INTERACTION MODEL
# IMPORTANT: region x cohort fixed effects
# ============================================================
ref_region = REFERENCE_REGION if REFERENCE_REGION in set(df["region"]) else df["region"].value_counts().idxmax()
other_regions = [r for r in sorted(df["region"].unique()) if r != ref_region]

for region in other_regions:
    reg_dummy = (df["region"] == region).astype(int)
    df[f"failed_x_{region}"] = df[ENDOG] * reg_dummy
    df[f"dist_x_{region}"] = df[INSTR] * reg_dummy

pooled_w = sm.add_constant(region_cohort_fe(df), has_constant="add")
pooled_d = df[[ENDOG] + [f"failed_x_{r}" for r in other_regions]].astype(float)
pooled_q = df[[INSTR] + [f"dist_x_{r}" for r in other_regions]].astype(float)

pooled_iv = iv_2sls_cluster(df[OUTCOME], pooled_w, pooled_d, pooled_q, df[CLUSTER])

implied_rows = []

base_coef = pooled_iv["coef"][ENDOG]
base_var = pooled_iv["cov"].loc[ENDOG, ENDOG]
base_se = float(np.sqrt(base_var))

implied_rows.append({
    "region": ref_region,
    "coef_raw": base_coef,
    "se_raw": base_se,
    "coef_pp": ppoints(base_coef),
    "se_pp": ppoints(base_se),
    "ci_low_pp": ppoints(base_coef - 1.96 * base_se),
    "ci_high_pp": ppoints(base_coef + 1.96 * base_se),
})

for region in other_regions:
    term = f"failed_x_{region}"
    coef = pooled_iv["coef"][ENDOG] + pooled_iv["coef"][term]
    var = (
        pooled_iv["cov"].loc[ENDOG, ENDOG]
        + pooled_iv["cov"].loc[term, term]
        + 2.0 * pooled_iv["cov"].loc[ENDOG, term]
    )
    se = float(np.sqrt(var))

    implied_rows.append({
        "region": region,
        "coef_raw": coef,
        "se_raw": se,
        "coef_pp": ppoints(coef),
        "se_pp": ppoints(se),
        "ci_low_pp": ppoints(coef - 1.96 * se),
        "ci_high_pp": ppoints(coef + 1.96 * se),
    })

implied = pd.DataFrame(implied_rows).sort_values("coef_pp").reset_index(drop=True)
implied.to_csv(OUTPUT_DIR / "pooled_implied_region_effects.csv", index=False)

# ============================================================
# 3) JOINT TEST OF HETEROGENEITY
# ============================================================
interaction_terms = [f"failed_x_{r}" for r in other_regions]
idx = [pooled_iv["coef"].index.get_loc(term) for term in interaction_terms]

coef_vec = pooled_iv["coef"].iloc[idx].to_numpy()
cov_mat = pooled_iv["cov"].iloc[idx, idx].to_numpy()

wald_chi2 = float(coef_vec.T @ np.linalg.pinv(cov_mat) @ coef_vec)
q_test = len(interaction_terms)
wald_F = wald_chi2 / q_test
p_value_F = 1.0 - stats.f.cdf(wald_F, q_test, pooled_iv["nclusters"] - 1)

joint_test = pd.DataFrame([{
    "test": "All region interaction effects = 0",
    "reference_region": ref_region,
    "chi2": wald_chi2,
    "q": q_test,
    "F": wald_F,
    "df1": q_test,
    "df2": pooled_iv["nclusters"] - 1,
    "p_value_F": p_value_F,
}])

joint_test.to_csv(OUTPUT_DIR / "joint_heterogeneity_test.csv", index=False)

print("\n=== Implied pooled region effects (pp) ===")
print(implied[["region", "coef_pp", "se_pp", "ci_low_pp", "ci_high_pp"]])

print("\n=== Joint heterogeneity test ===")
print(joint_test)

# ============================================================
# 4) FIGURE: REGIONAL IV EFFECTS
# ============================================================
plot_df = implied.sort_values("coef_pp").copy()

plt.figure(figsize=(8, 5))
plt.errorbar(
    x=plot_df["coef_pp"],
    y=plot_df["region"],
    xerr=1.96 * plot_df["se_pp"],
    fmt="o",
    capsize=4
)
plt.axvline(0, linestyle="--")
plt.xlabel("Effect of Failed Medical Exam on Public Employment (percentage points)")
plt.ylabel("Region")
plt.title("IV Estimates by Region")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "figure1_region_iv_effects.png", dpi=300, bbox_inches="tight")
plt.show()

# ============================================================
# 5) FINAL PAPER TABLE
# ============================================================
paper_table = pd.concat([
    baseline_summary.assign(kind="Baseline"),
    region_table.assign(kind="Region-specific")
], ignore_index=True, sort=False)

paper_table.to_csv(OUTPUT_DIR / "paper_iv_table.csv", index=False)

print("\nDone.")
print(f"Outputs saved in: {OUTPUT_DIR}")