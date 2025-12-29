# FairHire: Skills, Gender, and Labor Market Outcomes (2018–2025)

## Problem Statement
This project investigates the relative importance of skills and gender in predicting employment and wage outcomes in the U.S. labor market. Using microdata from the IPUMS CPS ASEC (2018–2025), we assess whether gender provides incremental predictive power for labor market status once detailed skill and occupation controls are included.

## Data Description
- **Source:** IPUMS CPS ASEC microdata, 2018–2025 (https://cps.ipums.org/)
- **Sample:** U.S. civilian labor force, restricted to skilled workers as defined in the project code.
- **Variables:** Demographics (age, sex, education), occupation, industry, year, employment status, wage income.
- **Access:** Data must be requested directly from IPUMS. No raw microdata are included in this repository.

## Methodology
- **Preprocessing:** Data cleaning, skilled worker definition, and variable selection are performed in Python using pandas and scikit-learn.
- **Modeling:**
  - **Employment Model:** Logistic regression predicting employment status (EMPLOYED) using two feature sets:
    1. Skills-only: age, education, occupation, industry, year
    2. Skills + gender: above plus sex
  - **Wage Model:** Linear regression predicting wage income (INCWAGE) with analogous feature sets (see code for details).
- **Evaluation:**
  - Employment model: Area Under the ROC Curve (AUC)
  - Wage model: R² (coefficient of determination)

## Key Results

### Employment (Skilled workers)
- Skills-only AUC: 0.9497
- Skills + gender AUC: 0.9498
- Incremental lift from gender: 0.00012

### Wages (Skilled & employed)
- Skills-only R²: 0.2773
- Skills + gender R²: 0.2917
- Incremental R² from gender: 0.0144

> In both models, the inclusion of gender provides negligible incremental predictive power once skills and occupation are controlled for.

## Reproducibility
1. **Install dependencies:**
   ```sh
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. **Obtain IPUMS CPS ASEC data (2018–2025):**
   - Register and download extracts from https://cps.ipums.org/ (see `01_download_ipums.py` for automated download script).
3. **Run analysis pipeline:**
   - Execute scripts in order: `01_download_ipums.py` → ... → `10_plot_coefficients.py`
   - See code comments for details and customization.

## Reproducibility, Ethics, and Limitations
- **Data citation:** Use of IPUMS CPS data requires proper citation. Refer to IPUMS guidelines and cite the data source in any publications or derivative work.
- **Observational analysis:** All results are based on observational, cross-sectional survey data. The models estimate associations, not causal effects. No causal claims regarding the impact of gender or skills on employment or wages should be made from these results.
- **Interpretation boundaries:** The findings indicate the incremental predictive value of gender after accounting for skills and occupation, but do not address underlying mechanisms, discrimination, or policy effects. Results are conditional on the variables and definitions used in this analysis.
- **Responsible use:** Gender-related results should be interpreted with care. The absence of predictive power for gender, conditional on skills, does not imply the absence of gender disparities in the labor market, nor does it preclude the existence of structural or unmeasured factors. Users are encouraged to consider broader social, economic, and institutional contexts when interpreting these findings.

## Citation
If using this code or methodology, please cite the IPUMS CPS data and acknowledge this repository.
