import os
import pandas as pd
import statsmodels.formula.api as smf
from linearmodels.iv import IV2SLS

COHORT_FE = "dcohort2 + dcohort3 + dcohort4 + dcohort5"


def _fmt(x, digits=2):
    if pd.isna(x):
        return ""
    return f"{x:.{digits}f}"


def _fmt3(x):
    if pd.isna(x):
        return ""
    return f"{x:.3f}"


def _fmt4(x):
    if pd.isna(x):
        return ""
    return f"{x:.4f}"


def _fmt5(x):
    if pd.isna(x):
        return ""
    return f"{x:.5f}"


def _fit_ols_safe(formula, data, cluster_var=None):
    model = smf.ols(formula, data=data)

    if cluster_var is None:
        return model.fit()

    n_clusters = data[cluster_var].nunique(dropna=True)
    if n_clusters >= 2:
        return model.fit(cov_type="cluster", cov_kwds={"groups": data[cluster_var]})
    return model.fit()


def _fit_iv_safe(formula, data, cluster_var=None):
    model = IV2SLS.from_formula(formula, data=data)

    if cluster_var is None:
        return model.fit()

    n_clusters = data[cluster_var].nunique(dropna=True)
    if n_clusters >= 2:
        return model.fit(cov_type="clustered", clusters=data[cluster_var])
    return model.fit()


def run_table2(main_df, women_df, cohort76_df):
    os.makedirs("output", exist_ok=True)

    main_df = main_df[main_df["attriter"] == 0].copy()
    cluster_col = "cohort_id"
    needed_fe = ["dcohort2", "dcohort3", "dcohort4", "dcohort5"]

    # Panel A, col 1
    a1_df = main_df[main_df["draft_eligible"] == 1].dropna(
        subset=["empl_public100", "dist_cutoff_right1000", cluster_col] + needed_fe
    ).copy()

    a1 = _fit_ols_safe(
        f"empl_public100 ~ dist_cutoff_right1000 + {COHORT_FE}",
        a1_df,
        cluster_var=cluster_col
    )

    # Panel A, col 2
    a2_df = women_df[women_df["draft_exempt"] == 1].dropna(
        subset=["empl_public100", "dist_cutoff_left1000", "cohort_id"] + needed_fe
    ).copy()

    a2 = _fit_ols_safe(
        f"empl_public100 ~ dist_cutoff_left1000 + {COHORT_FE}",
        a2_df,
        cluster_var="cohort_id"
    )

    # Panel A, col 3
    a3_df = cohort76_df.dropna(
        subset=["empl_public100", "dist_cutoff_left1000"]
    ).copy()

    a3 = _fit_ols_safe(
        "empl_public100 ~ dist_cutoff_left1000",
        a3_df,
        cluster_var=None
    )

    # Panel B
    b_df = main_df[main_df["draft_exempt"] == 1].copy()

    b1_df = b_df.dropna(
        subset=["empl_private100", "failed_exam", "dist_cutoff_left1000", cluster_col] + needed_fe
    ).copy()

    b1 = _fit_iv_safe(
        f"empl_private100 ~ 1 + {COHORT_FE} + [failed_exam ~ dist_cutoff_left1000]",
        b1_df,
        cluster_var=cluster_col
    )

    b2_df = b_df.dropna(
        subset=["merit100", "failed_exam", "dist_cutoff_left1000", cluster_col] + needed_fe
    ).copy()

    b2 = _fit_iv_safe(
        f"merit100 ~ 1 + {COHORT_FE} + [failed_exam ~ dist_cutoff_left1000]",
        b2_df,
        cluster_var=cluster_col
    )

    b3_df = b_df.dropna(
        subset=["nonmerit100", "failed_exam", "dist_cutoff_left1000", cluster_col] + needed_fe
    ).copy()

    b3 = _fit_iv_safe(
        f"nonmerit100 ~ 1 + {COHORT_FE} + [failed_exam ~ dist_cutoff_left1000]",
        b3_df,
        cluster_var=cluster_col
    )

    panel_a = pd.DataFrame([
        [
            "DistanceToCutoff",
            _fmt4(a1.params["dist_cutoff_right1000"]),
            _fmt4(a2.params["dist_cutoff_left1000"]),
            ""
        ],
        [
            " ",
            f"({_fmt4(a1.bse['dist_cutoff_right1000'])})",
            f"({_fmt4(a2.bse['dist_cutoff_left1000'])})",
            ""
        ],
        [
            "DistanceToPreviousYearCutoff",
            "",
            "",
            _fmt5(a3.params["dist_cutoff_left1000"])
        ],
        [
            " ",
            "",
            "",
            f"({_fmt5(a3.bse['dist_cutoff_left1000'])})"
        ],
        [
            "Observations",
            f"{len(a1_df):,}",
            f"{len(a2_df):,}",
            f"{len(a3_df):,}"
        ],
        [
            "Sample",
            "Draft-eligible men",
            "Women with exempt ID",
            "Draft-exempt men"
        ],
        [
            "Cohorts",
            "1958–1962",
            "1958–1962",
            "1976"
        ],
        [
            "Dependent variable mean",
            _fmt(a1_df["empl_public100"].mean()),
            _fmt(a2_df["empl_public100"].mean()),
            _fmt(a3_df["empl_public100"].mean())
        ],
        [
            "Estimation method",
            "OLS",
            "OLS",
            "OLS"
        ]
    ], columns=[
        "",
        "PublicSectorEmployee (1)",
        "FemalePublicSectorEmployee (2)",
        "PublicSectorEmployee (1976) (3)"
    ])

    panel_b = pd.DataFrame([
        [
            "FailedMedicalExamination",
            _fmt(b1.params["failed_exam"]),
            _fmt3(b2.params["failed_exam"]),
            _fmt(b3.params["failed_exam"])
        ],
        [
            " ",
            f"({_fmt(b1.std_errors['failed_exam'])})",
            f"({_fmt3(b2.std_errors['failed_exam'])})",
            f"({_fmt(b3.std_errors['failed_exam'])})"
        ],
        [
            "Observations",
            f"{len(b1_df):,}",
            f"{len(b2_df):,}",
            f"{len(b3_df):,}"
        ],
        [
            "Sample",
            "Draft-exempt men",
            "Draft-exempt men",
            "Draft-exempt men"
        ],
        [
            "Cohorts",
            "1958–1962",
            "1958–1962",
            "1958–1962"
        ],
        [
            "Dependent variable mean",
            _fmt(b1_df["empl_private100"].mean()),
            _fmt3(b2_df["merit100"].mean()),
            _fmt(b3_df["nonmerit100"].mean())
        ],
        [
            "Estimation method",
            "2SLS",
            "2SLS",
            "2SLS"
        ]
    ], columns=[
        "",
        "PrivateSectorEmployee (1)",
        "MeritocraticPublicSectorEmployee (2)",
        "NonMeritocraticPublicSectorEmployee (3)"
    ])

    print("\nTABLE 2 — Falsification Tests and Further Results\n")
    print("Panel A\n")
    print(panel_a.to_string(index=False))
    print("\nPanel B\n")
    print(panel_b.to_string(index=False))

    panel_a.to_csv("output/table2_panel_a.csv", index=False)
    panel_b.to_csv("output/table2_panel_b.csv", index=False)