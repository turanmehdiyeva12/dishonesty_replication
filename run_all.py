from src.load_data import load_data
from src.table1 import run_table1
from src.table2 import run_table2
from src.figures import run_figures

def main():

    main_df, women_df, cohort76_df = load_data()

    run_table1(main_df)

    run_table2(main_df, women_df, cohort76_df)

    run_figures(main_df)

    print("Replication complete")

if __name__ == "__main__":
    main()