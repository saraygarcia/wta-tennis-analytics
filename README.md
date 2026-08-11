# WTA Tennis Analytics — Match Predictor 🎾

An end-to-end data science project analyzing the WTA tour (2020-2026), featuring interactive visualizations, network analysis of rivalries, and a machine learning model for match prediction deployed as a Streamlit app.

## Live Demo

Run the predictor locally:
```bash
python -m streamlit run app.py
```

Select any two WTA players, choose a surface (Hard, Clay, Grass), and get a win probability prediction based on ranking, recent form, and serve statistics.

## Key Results

- **16,056 matches** analyzed across 7 seasons (2020-2026)
- **Predictive model:** Random Forest with 61.8% accuracy and 68.0% AUC-ROC (validated with temporal split)
- **Top predictor:** Ranking difference explains 25% of match outcomes
- **Feature engineering impact:** Adding historical stats (aces, break points, win rate) improved AUC from 67.6% to 68.0%

## Highlights

### The Sabalenka vs Gauff Rivalry

One of the most intense rivalries in modern tennis — 13 matches, 7-6 in favor of Sabalenka.

| Surface | Sabalenka | Gauff | Insight |
|---|---|---|---|
| Hard | 6 | 4 | Sabalenka dominates on her best surface |
| Clay | 1 | 2 | Gauff wins the big ones (Roland Garros Final) |
| Grass | 0 | 0 | Never played |

**54% of their matches go to 3 sets** — average duration of 120 minutes. Gauff leads 2-1 in Grand Slams, showing she elevates her game when it matters most.

### Who Rules Each Surface

Interactive heatmaps comparing win rates across surfaces reveal specialists vs all-court players. Swiatek dominates Clay (90%+ win rate), while Sabalenka excels on Hard courts.

### Network of Rivalries

A NetworkX graph reveals the structure of WTA rivalries — who plays whom most often, and how win rates create tiers of dominance.

### Serve Analysis

Radar charts comparing aces per match, first serve %, break points saved, and win rate across the top WTA players.

## Project Structure

```
wta-tennis-analytics/
│
├── data/
│   ├── wta_matches_2020.csv          # Match data by year
│   ├── wta_matches_2021.csv
│   ├── wta_matches_2022.csv
│   ├── wta_matches_2023.csv
│   ├── wta_matches_2024.csv
│   ├── wta_matches_2025.csv
│   └── wta_matches_2026.csv
│
├── notebooks/
│   └── 01_exploracion_wta.ipynb      # Full analysis notebook
│
├── outputs/                           # Exported visualizations
│   ├── 01_top15_winrate.png
│   ├── 02_heatmap_superficies.png
│   ├── 03_h2h_evolucion.png
│   ├── 04_radar_servicio.png
│   ├── 05_red_rivalidades.png
│   └── 06_feature_importance_prediccion.png
│
├── app.py                             # Streamlit predictor app
├── modelo_rf_wta.pkl                  # Trained Random Forest model
├── datos_wta.pkl                      # Processed match data
├── README.md
└── requirements.txt
```

## Analysis Breakdown

### 1. EDA — Tour Overview
- Match volume by year (COVID-19 impact in 2020 clearly visible)
- Top 15 players by win rate with interactive year-by-year animation (Plotly)
- Surface distribution: Hard (61%), Clay (28%), Grass (10%)

### 2. Surface Analysis
- Heatmap of win rate by surface for top players (filterable by year)
- Identification of surface specialists vs all-court players

### 3. Rivalry Deep Dive — Sabalenka vs Gauff
- Complete H2H record with tournament context
- Breakdown by surface and tournament category (Grand Slam, WTA 1000, Finals)
- Intensity analysis: 3-set frequency, match duration, tiebreaks

### 4. Serve Statistics
- Comparative analysis: aces, double faults, 1st serve %, 1st serve won %, break points saved
- Radar chart comparison of top players
- All functions filterable by year

### 5. Network Analysis
- Rivalry network graph built with NetworkX
- Node size = total matches played, node color = win rate
- Edge thickness = number of head-to-head matches
- Dark theme visualization

### 6. Predictive Model
- **Approach:** predict match winner using pre-match available data only
- **Models compared:** Logistic Regression, Random Forest, XGBoost
- **Split:** temporal (train 2020-2024, test 2025-2026) — not random, to avoid data leakage
- **Features:** ranking, ranking difference, recent win rate, aces per match, break points saved %, surface
- **Result:** Random Forest champion (61.8% accuracy, 68.0% AUC)
- **Deployed** as interactive Streamlit app

### Model Limitations (Honest Assessment)

The model achieves ~62% accuracy, which is realistic for tennis prediction. Professional betting models with much richer data (point-by-point, serve speed, court positioning) reach ~67-70%. The main limitations are:

1. **Mental factor:** Cannot capture psychological dynamics (e.g., Gauff's ability to elevate in Grand Slams)
2. **Match context:** Round, pressure, crowd support are not quantified
3. **Recent form nuance:** A 50% win rate averages hot streaks and cold spells without distinguishing trend direction

These limitations are documented intentionally — understanding what a model cannot do is as important as what it can.

## Tech Stack

- **Python 3.13** — core language
- **pandas, numpy** — data manipulation
- **matplotlib, seaborn** — static visualizations
- **plotly** — interactive animated charts
- **NetworkX** — rivalry network graph
- **scikit-learn** — Logistic Regression, Random Forest, metrics
- **XGBoost** — gradient boosting classifier
- **Streamlit** — web app deployment
- **joblib** — model serialization

## Data Source

Jeff Sackmann's WTA Tennis Dataset (via [Aneeshers/tennis-sackmann-archive](https://github.com/Aneeshers/tennis-sackmann-archive)) — Creative Commons Attribution-NonCommercial-ShareAlike 4.0.

## About

**Author:** Saray Garcia — Senior Analyst, Data & AI Risk @ Scotia GBS Colombia

**Portfolio:**
- [Project 1: Credit Default Predictor](https://github.com/saraygarcia/credit-default-predictor) — XGBoost, SHAP, PSI monitoring
- [Project 2: Customer Segmentation](https://github.com/saraygarcia/customer-segmentation) — K-Means, PCA, segment profiling
- **Project 3: WTA Tennis Analytics** (this repo) — NetworkX, Plotly, Streamlit deployment

**Contact:** [LinkedIn](https://linkedin.com/in/saraydgarciag) | [GitHub](https://github.com/saraygarcia)
