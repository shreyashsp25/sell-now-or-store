# Sell Now or Store?

**Forecasting onion and maize prices in Karnataka mandis, and testing in rupees whether a forecast helps a farmer decide when to sell.**

Foundations of Data Science project, Team 10, Amrita Vishwa Vidyapeetham, Bengaluru
Janani Sree (BL.SC.U4CSE24227) · Shreyash Shanmukh Piriya (BL.SC.U4CSE24037)

---

## The question

At harvest a farmer can sell at today's mandi price or store the crop and hope the price rises. Most studies stop at forecast accuracy. We ask a harder question: **does a better forecast actually earn the farmer more money?**

## Data

- Agmarknet daily mandi prices (2003-2026), taken from the Hugging Face Agmarknet CSV files
- 267,369 onion and maize records, cleaned to 253,833, across 149 Karnataka markets
- Turned into a weekly median price panel for each mandi

## What we did

| Step | Notebook | Main methods |
|---|---|---|
| 1. Overview | `01_overview_and_crop_selection` | Why onion (perishable, volatile) and maize (storable, stable) |
| 2. Cleaning | `02_cleaning_missing_outliers` | Market name merge, cause-based error removal, linear interpolation |
| 3. EDA and tests | `03_eda_statistics_hypothesis_tests` | Seasonal index, Kruskal-Wallis, Wilcoxon, Mann-Whitney U, chi-square, Holm correction |
| 4. Features and PCA | `04_features_and_pca` | 31 features, PCA, stratified sampling |
| 5. Forecasting | `05_forecasting_models` | Ridge, Random Forest, Gradient Boosting, time-series CV, conformal 80% ranges |
| 6. Simulation | `06_sell_or_store_simulation` | 2,868 real harvest decisions, 2,000 Monte Carlo cost scenarios |

## Key results

- Onion prices fall to **0.70x** their yearly average in May and rise to **1.15x** in December. Maize stays near 1x.
- Forecast error: Random Forest for onion (MAE 0.32 on log price) and Ridge for maize (0.09). Both are **11-16% better** than the seasonal baseline.
- Storing onion without a forecast loses **Rs 203 per quintal**. The forecast rule gains up to **Rs 154 per quintal** with cold storage, and Rs 27 for maize.
- The forecast rule is the best rule in **79-100%** of the cost scenarios.

## Dashboard

A Streamlit app where a farmer enters a crop, district, date and price and gets 4, 8 and 12-week forecasts with a sell-or-store signal.

```bash
pip install -r requirements.txt
streamlit run dashboard/app.py
```

## Folder structure

```
fds/
├── dashboard/app.py          Streamlit app
├── data/                     Cleaned data and results (CSV)
├── models/                   Trained models (joblib)
├── notebooks/                Notebooks 01-06
├── extract_karnataka.py      Builds data/karnataka_all_years.csv from raw files
└── requirements.txt
```

## Rebuilding the full data

The raw Agmarknet files (about 7 GB) and `data/karnataka_all_years.csv` are too large for GitHub. To rebuild them:

1. Download the Agmarknet CSV files from Hugging Face into `raw_data/`
2. Run `python extract_karnataka.py`
3. Run the notebooks in order

## References

1. Harish Nayak et al., 2024, *Scientific Reports* 14:17203
2. Avinash et al., 2024, *Statistics and Applications* 22(2)
3. Suresha et al., 2021, *Asian Journal of Research in Computer Science* 12(4)
4. Jain et al., 2020, IBM Research India (arXiv 2009.04171)
5. Ma et al., Carnegie Mellon University (arXiv 1812.05173)
6. Ajmal et al., 2024, *AJAEES* 42(3)
7. Patil et al., 2026 (arXiv 2604.01792)
8. Roy et al., 2024, RBI Working Paper 08/2024
9. Bhattacharya and Chowdhury, 2021, NIPFP Working Paper 336
10. Indhushree and Reddy, 2015, *Journal of Research ANGRAU* 43(3-4)
11. Gammelli et al., bike-sharing demand forecasts for inventory management (arXiv 2108.00858)
12. Theodorou et al., 2025, *European Journal of Operational Research* 322(2)
