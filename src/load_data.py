import pandas as pd

def load_data():

    main_df = pd.read_stata("data/FINAL_REPOSITORY_MAIN_FEB2023.dta")
    women_df = pd.read_stata("data/FINAL_Employment_Women.dta")
    cohort76_df = pd.read_stata("data/FINAL_Employment_1976.dta")

    return main_df, women_df, cohort76_df