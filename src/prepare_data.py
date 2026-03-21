def prepare_main(df):
    # Most of these already exist in your .dta, so only create if missing
    if "empl_public100" not in df.columns and "empl_public" in df.columns:
        df["empl_public100"] = df["empl_public"] * 100

    if "empl_private100" not in df.columns and "empl_private" in df.columns:
        df["empl_private100"] = df["empl_private"] * 100

    if "failed_exam100" not in df.columns and "failed_exam" in df.columns:
        df["failed_exam100"] = df["failed_exam"] * 100

    if "cond_easy100" not in df.columns and "cond_easy" in df.columns:
        df["cond_easy100"] = df["cond_easy"] * 100

    if "cond_inter100" not in df.columns and "cond_inter" in df.columns:
        df["cond_inter100"] = df["cond_inter"] * 100

    if "cond_hard100" not in df.columns and "cond_hard" in df.columns:
        df["cond_hard100"] = df["cond_hard"] * 100

    if "merit100" not in df.columns and "merit" in df.columns:
        df["merit100"] = df["merit"] * 100

    if "nonmerit100" not in df.columns and "nonmerit" in df.columns:
        df["nonmerit100"] = df["nonmerit"] * 100

    return df