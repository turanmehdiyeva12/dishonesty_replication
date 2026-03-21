import os
import pandas as pd
import matplotlib.pyplot as plt


def binscatter(x, y, bins):
    df = pd.DataFrame({"x": x, "y": y}).dropna()

    df["bin"] = pd.qcut(df["x"], bins, duplicates="drop")

    grouped = df.groupby("bin", observed=False)

    return grouped["x"].mean(), grouped["y"].mean()


def run_figures(df):

    os.makedirs("output", exist_ok=True)

    # -----------------------------
    # FIGURE 1
    # -----------------------------

    fig_df = df.copy()

    plt.figure(figsize=(8,6))

    variables = {
        "failed_exam100": ("All conditions","black","s"),
        "cond_easy100": ("Easy-to-cheat conditions","blue","o"),
        "cond_inter100": ("Intermediate conditions","green","D"),
        "cond_hard100": ("Hard-to-cheat conditions","red","^")
    }

    for var,(label,color,marker) in variables.items():

        x,y = binscatter(
            fig_df["distancecutoff_norm"],
            fig_df[var],
            bins=40
        )

        plt.scatter(x,y,color=color,marker=marker,label=label)

    plt.axvline(0,linestyle="--",color="black")

    plt.xlabel("Normalized difference between draft lottery number and cutoff")
    plt.ylabel("Failure rate in medical examination (percent)")

    plt.legend()

    plt.tight_layout()

    # Save
    plt.savefig("output/figure1.png",dpi=300)

    # Show in PyCharm
    plt.show()

    # -----------------------------
    # FIGURE 2
    # -----------------------------

    fig2_df = df[(df.attriter == 0) & (df.draft_exempt == 1)]

    plt.figure(figsize=(10,4))

    # Panel A
    plt.subplot(1,2,1)

    x,y = binscatter(
        fig2_df["distancecutoff_norm"],
        fig2_df["empl_public100"],
        bins=12
    )

    plt.scatter(x,y)

    plt.title("Panel A. Public employment")

    plt.xlabel("Normalized difference between draft lottery number and cutoff")
    plt.ylabel("Public sector employment (percent)")

    # Panel B
    plt.subplot(1,2,2)

    x,y = binscatter(
        fig2_df["distancecutoff_norm"],
        fig2_df["empl_private100"],
        bins=12
    )

    plt.scatter(x,y)

    plt.title("Panel B. Private employment")

    plt.xlabel("Normalized difference between draft lottery number and cutoff")
    plt.ylabel("Private sector employment (percent)")

    plt.tight_layout()

    # Save
    plt.savefig("output/figure2.png",dpi=300)

    # Show
    plt.show()