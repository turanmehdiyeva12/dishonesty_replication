import os
import pandas as pd
import statsmodels.formula.api as smf
from linearmodels.iv import IV2SLS

COHORT_FE = "dcohort2 + dcohort3 + dcohort4 + dcohort5"


def _fmt(x, digits=2):
    if pd.isna(x):
        return ""
    return f"{x:.{digits}f}"


def _fmt4(x):
    if pd.isna(x):
        return ""
    return f"{x:.4f}"


def _table1_dataframe(c1, c2, c3, c4, c5, c6, c7, c8,
                      n1, n2, n3, n4, n5, n6, n7, n8,
                      mean1, mean2, mean3, mean4, mean5, mean6, mean7, mean8):
    rows = [
        [
            "FailedMedicalExamination",
            _fmt(c1.params["failed_exam"]), _fmt(c2.params["failed_exam"]), "",
            "", _fmt(c5.params["failed_exam"]), "", "", _fmt(c8.params["failed_exam"])
        ],
        [
            " ",
            f"({_fmt(c1.bse['failed_exam'])})", f"({_fmt(c2.bse['failed_exam'])})", "",
            "", f"({_fmt(c5.std_errors['failed_exam'])})", "", "", f"({_fmt(c8.std_errors['failed_exam'])})"
        ],
        [
            "DistanceToCutoff",
            "", "", _fmt4(c3.params["dist_cutoff_left1000"]),
            _fmt4(c4.params["dist_cutoff_left1000"]), "", "", "", ""
        ],
        [
            " ",
            "", "", f"({_fmt4(c3.bse['dist_cutoff_left1000'])})",
            f"({_fmt4(c4.bse['dist_cutoff_left1000'])})", "", "", "", ""
        ],
        [
            "DistanceToPreviousYearCutoff",
            "", "", "", "", "",
            _fmt4(c6.params["prev_dist_cutoff_left1000"]),
            _fmt4(c7.params["prev_dist_cutoff_left1000"]), ""
        ],
        [
            " ",
            "", "", "", "", "",
            f"({_fmt4(c6.bse['prev_dist_cutoff_left1000'])})",
            f"({_fmt4(c7.bse['prev_dist_cutoff_left1000'])})", ""
        ],
        [
            "Estimation method",
            "OLS", "OLS", "OLS (first stage)", "OLS (reduced form)",
            "2SLS", "OLS (first stage)", "OLS (reduced form)", "2SLS"
        ],
        [
            "Dependent variable mean",
            _fmt(mean1), _fmt(mean2), _fmt(mean3), _fmt(mean4),
            _fmt(mean5), _fmt(mean6), _fmt(mean7), _fmt(mean8)
        ],
        [
            "Observations",
            f"{n1:,}", f"{n2:,}", f"{n3:,}", f"{n4:,}",
            f"{n5:,}", f"{n6:,}", f"{n7:,}", f"{n8:,}"
        ]
    ]

    columns = [
        "",
        "Public Sector Employee (1)",
        "Private Sector Employee (2)",
        "Failed Medical Examination (3)",
        "Public Sector Employee (4)",
        "Public Sector Employee (5)",
        "Failed Medical Examination (6)",
        "Public Sector Employee (7)",
        "Public Sector Employee (8)"
    ]
    return pd.DataFrame(rows, columns=columns)


def run_table1(df):
    os.makedirs("output", exist_ok=True)

    df = df[df["attriter"] == 0].copy()
    exempt = df[df["draft_exempt"] == 1].copy()

    needed_fe = ["dcohort2", "dcohort3", "dcohort4", "dcohort5"]

    c1_df = exempt.dropna(subset=["empl_public100", "failed_exam", "cohort_id"] + needed_fe).copy()
    c1 = smf.ols(
        f"empl_public100 ~ failed_exam + {COHORT_FE}",
        data=c1_df
    ).fit(cov_type="cluster", cov_kwds={"groups": c1_df["cohort_id"]})

    c2_df = exempt.dropna(subset=["empl_private100", "failed_exam", "cohort_id"] + needed_fe).copy()
    c2 = smf.ols(
        f"empl_private100 ~ failed_exam + {COHORT_FE}",
        data=c2_df
    ).fit(cov_type="cluster", cov_kwds={"groups": c2_df["cohort_id"]})

    c3_df = exempt.dropna(subset=["failed_exam100", "dist_cutoff_left1000", "cohort_id"] + needed_fe).copy()
    c3 = smf.ols(
        f"failed_exam100 ~ dist_cutoff_left1000 + {COHORT_FE}",
        data=c3_df
    ).fit(cov_type="cluster", cov_kwds={"groups": c3_df["cohort_id"]})

    c4_df = df.dropna(subset=["empl_public100", "dist_cutoff_left1000", "cohort_id"] + needed_fe).copy()
    c4 = smf.ols(
        f"empl_public100 ~ dist_cutoff_left1000 + {COHORT_FE}",
        data=c4_df
    ).fit(cov_type="cluster", cov_kwds={"groups": c4_df["cohort_id"]})

    c5_df = exempt.dropna(subset=["empl_public100", "failed_exam", "dist_cutoff_left1000", "cohort_id"] + needed_fe).copy()
    c5 = IV2SLS.from_formula(
        f"empl_public100 ~ 1 + {COHORT_FE} + [failed_exam ~ dist_cutoff_left1000]",
        data=c5_df
    ).fit(cov_type="clustered", clusters=c5_df["cohort_id"])

    c6_df = exempt.dropna(subset=["failed_exam100", "prev_dist_cutoff_left1000", "cohort_id"] + needed_fe).copy()
    c6 = smf.ols(
        f"failed_exam100 ~ prev_dist_cutoff_left1000 + {COHORT_FE}",
        data=c6_df
    ).fit(cov_type="cluster", cov_kwds={"groups": c6_df["cohort_id"]})

    c7_df = exempt.dropna(subset=["empl_public100", "prev_dist_cutoff_left1000", "cohort_id"] + needed_fe).copy()
    c7 = smf.ols(
        f"empl_public100 ~ prev_dist_cutoff_left1000 + {COHORT_FE}",
        data=c7_df
    ).fit(cov_type="cluster", cov_kwds={"groups": c7_df["cohort_id"]})

    c8_df = exempt.dropna(subset=["empl_public100", "failed_exam", "prev_dist_cutoff_left1000", "cohort_id"] + needed_fe).copy()
    c8 = IV2SLS.from_formula(
        f"empl_public100 ~ 1 + {COHORT_FE} + [failed_exam ~ prev_dist_cutoff_left1000]",
        data=c8_df
    ).fit(cov_type="clustered", clusters=c8_df["cohort_id"])

    tbl = _table1_dataframe(
        c1, c2, c3, c4, c5, c6, c7, c8,
        len(c1_df), len(c2_df), len(c3_df), len(c4_df), len(c5_df), len(c6_df), len(c7_df), len(c8_df),
        c1_df["empl_public100"].mean(),
        c2_df["empl_private100"].mean(),
        c3_df["failed_exam100"].mean(),
        c4_df["empl_public100"].mean(),
        c5_df["empl_public100"].mean(),
        c6_df["failed_exam100"].mean(),
        c7_df["empl_public100"].mean(),
        c8_df["empl_public100"].mean()
    )

    print("\nTABLE 1 — Medical Examination, Employment, and Distance to Cutoff\n")
    print(tbl.to_string(index=False))
    tbl.to_csv("output/table1.csv", index=False)