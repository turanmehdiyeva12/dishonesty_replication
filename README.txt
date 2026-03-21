Replication Package
Paper: "Dishonesty and Public Employment"
Cruces, Rossi, and Schargrodsky (AER: Insights, 2023)

This package replicates the main tables and figures in the paper using Python.

Replicated Results
------------------
Table 1: Medical Examination, Employment, and Distance to Cutoff
Table 2: Falsification Tests and Further Results
Figure 1: Failure Rate in Medical Examination vs Draft Lottery Number
Figure 2: Reduced Form: Public and Private Employment vs Distance to Cutoff

Appendix tables and figures are not replicated.

Project Structure
-----------------
run_all.py             Master script to run the replication
src/                   Python scripts for tables and figures
data/                  Replication datasets
output/                Generated tables and figures

Data Files
----------
FINAL_REPOSITORY_MAIN_FEB2023.dta
FINAL_Employment_Women.dta
FINAL_Employment_1976.dta

Software Requirements
---------------------
Python 3.10+

Required packages:

pandas
numpy
statsmodels
linearmodels
matplotlib
pyreadstat

Installation
------------
Install packages using:

pip install pandas numpy statsmodels linearmodels matplotlib pyreadstat

How to Run the Replication
--------------------------
Open a terminal in the project directory and run:

python run_all.py

This will generate:

output/table1.csv
output/table2_panel_a.csv
output/table2_panel_b.csv
output/figure1.png
output/figure2.png

Notes
-----
The code reproduces the main regression results using the same
sample restrictions, fixed effects, and clustered standard errors
as described in the paper.

Figures are produced using binscatter-style quantile binning.