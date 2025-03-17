# Fantasy Football Draft Assistant (Part 1: Data Acquisition & Analysis)

## Overview  
This project is the **first phase** of our Fantasy Football Draft Assistant, a **data-driven** tool designed to assist fantasy football managers in making optimal draft decisions. In this **Part 1** of our Capstone Project, we focused on:
- **Data Acquisition**: Collecting relevant fantasy football and NFL statistics.
- **Preprocessing**: Cleaning, structuring, and organizing the dataset for analysis.
- **Exploratory Data Analysis (EDA)**: Identifying key trends, distributions, and relationships within the data.
- **Feature Engineering**: Creating new, meaningful features to improve model performance in future phases.

Future work in **Part 2 (next quarter)** will focus on:
- **Machine Learning Modeling**: Predicting fantasy performance using regression models.
- **Recommender System Construction**: Developing a system to optimize draft selections.
- **Dashboard UI Development**: Building an interactive tool for users.

## Data Sources  
Our dataset consists of **NFL player statistics from the 2022-2024 seasons**, collected from reputable sources:
- **Pro Football Reference**: Player stats (passing, rushing, receiving, kicking, team defense, special teams).
- **FantasyPros**: Average Draft Position (ADP) data from ESPN, NFL.com, RTSports, and Sleeper.
- **StatMuse**: Advanced statistical insights, including 2-point conversions.
- **NFL.com**: Defensive scoring and team performance statistics.

## Repository Structure  
Our repository is structured into two primary branches, each with its own focus:

### `web-scraping` Branch  
This branch contains all scripts related to **data acquisition** and **initial dataset preparation**.  
**Main Files & Folders**:  
- `/Capstone/` – Main directory for data acquisition and preprocessing.  
  - `MainScraper.py` – Web scraper for pulling NFL & FantasyPros stats.  
  - `defense_points_allowed.py` – Calculates defensive points allowed.  
  - `drop_multicollinearity.py` – Removes redundant correlated features.  
  - `fantasy_points_calc.py` – Computes fantasy points based on various scoring methods.  
  - `final_dataset_calc.py` – Merges and cleans the final dataset.  
  - `data/` – Folder containing the cleaned datasets.  
- `README.md` – Documentation for the web-scraping process.  
- `requirements.txt` – Project requirements

### `EDA` Branch  
This branch is dedicated to **data analysis, feature engineering, and preprocessing**.  
**Main Files & Folders**:  
- `data/` – Processed datasets used for EDA and modeling.  
- `.gitignore` – Ignore unnecessary files in commits.  
- `CalebScalingDistributions.ipynb` – Scaling distributions for player stats.  
- `Categorical and Univariate Analysis.ipynb` – Univariate and categorical data analysis.  
- `David EDA.ipynb` – Exploratory analysis on player statistics.  
- `HashimVIF.ipynb` / `HashimVIF2.ipynb` – Correlation and Variance Inflation Factor (VIF) analysis.  
- `New Visualizations.ipynb` – Various data visualizations.  
- `Tommy EDA.ipynb` / `Tommy EDAv2.ipynb` – Additional exploratory analysis.  
- `investigate_3tm_2tm.ipynb` – Handling of multi-team player transfers.  
- `README.md` – Documentation for the web-scraping process.  
- `requirements.txt` – Project requirements

## Steps in Part 1  

### 1. Data Acquisition & Preprocessing  
Run the **final_dataset_calc.py** script, which:  
- Calls **MainScraper.py** to collect **NFL player statistics from 2022-2024**.  
- Calls **fantasy_points_calc.py** to compute fantasy points for each season.  
- Merges all seasons into a **final structured dataset**.  

```sh
python Capstone/final_dataset_calc.py
```

### 2. Exploratory Data Analysis (EDA)  
- **Univariate & Bivariate Analysis**: Histograms, boxplots, and trend analysis.  
- **Missing Value & Outlier Handling**: Retained key outliers that indicate player performance.  
- **Correlation & Multicollinearity Checks**: Reduced redundant variables.  

Run Jupyter Notebook to explore EDA insights:  
```sh
jupyter notebook
# Open and run any EDA notebook
```  

### 3. Feature Engineering & Multicollinearity Reduction  
Run the **drop_multicollinearity.py** script to refine the dataset by:  
- Removing **redundant correlated variables** using Variance Inflation Factor (VIF) analysis.  
- Creating **new derived features** such as composite player performance metrics.  

```sh
python Capstone/drop_multicollinearity.py
```

## Next Steps in Part 2  

**Machine Learning Models**  
- Initial Model Training & Evaluation (MLP Regression, XGBoost, Linear Regression).  
- Pipeline Adjustments & Feature Weighting.  
  - **Consider Winsorization** to handle extreme outliers.  
  - **Adjust Season Weighting** to prioritize recent performances.  

**Recommender System Construction**  
- Implementing **custom draft logic** (e.g., "If no QB in round 1, prioritize WR/RB").  
- Optimizing player selection based on real-time draft inputs.  

**Dashboard Development**  
- Building an **interactive web-based UI** for fantasy football managers.  

## Installation & Usage  

**1. Clone the repository**  
```sh
git clone https://github.com/caleb10miller/FantasyFootballHelper.git
cd FantasyFootballHelper
```  
**2. Switch to the desired branch**  
- **For Data Acquisition (Web Scraping):**  
  ```sh
  git checkout web-scraping
  ```  
- **For EDA & Feature Engineering:**  
  ```sh
  git checkout EDA
  ```  

**Install dependencies**  
```sh
pip install -r requirements.txt
```  

**Run Data Processing Scripts in Order**  
```sh
python Capstone/final_dataset_calc.py
python Capstone/drop_multicollinearity.py
```  

## Contributors  
- **Caleb Miller** – cm3962@drexel.edu  
- **Hashim Afzal** – ha695@drexel.edu 
- **Thomas Kiefer** – tmk326@drexel.edu  
- **David Blankenship** – dwb65@drexel.edu  

## References  
- [Pro Football Reference](https://www.pro-football-reference.com)  
- [FantasyPros](https://www.fantasypros.com)  
- [StatMuse](https://www.statmuse.com)  
- [NFL.com](https://www.nfl.com)  

## License  
This project is open-source and available under the **MIT License**.

---  
**This README is for Part 1 of the project. The repository will be updated with additional sections as we progress into Part 2.**  
