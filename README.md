# ApplicationPrincipalComponents_11

This project analyzes the Telco customer churn dataset using numeric scatter plots,
Pearson correlations, and principal component analysis (PCA). The written results,
figures, and references are in [REPORT.md](REPORT.md).

## Reproduce the analysis

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the analysis with the assignment workbook or its CSV equivalent:

```bash
python analyze_telco_churn.py path/to/Telco-Customer-Churn.xlsx --output-dir report_assets
```

The script excludes customer IDs and churn labels from the PCA, encodes categorical
predictors, scales the input features, and writes plots and CSV tables to the output
directory. Churn is used only to color the plots.

The assignment workbook was not included in this repository. The checked-in report
uses the matching public WA_Fn-UseC Telco churn CSV linked in its references; rerun
the script with the assignment workbook to reproduce results against that exact file.
