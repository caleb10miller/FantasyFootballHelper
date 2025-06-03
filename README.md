# Fantasy Football Draft Assistant Capstone Project

## Overview  
This repository contains Fantasy Football Draft Assistant, a **data-driven** tool designed to assist fantasy football managers in making optimal draft decisions. Our Capstone project for Drexel University included the following features:
- **Data Acquisition**: Collecting relevant fantasy football and NFL statistics.
- **Preprocessing**: Cleaning, structuring, and organizing the dataset for analysis.
- **Exploratory Data Analysis (EDA)**: Identifying key trends, distributions, and relationships within the data.
- **Feature Engineering**: Creating new, meaningful features to improve model performance in future phases.
- **Machine Learning Modeling**: Predicting fantasy performance using regression models.
- **Recommender System Construction**: Developing a hybrid system that combines ML model outputs with rule-based heuristics to enhance draft strategy.
- **Dashboard UI Development**: Building an interactive tool for users. A demo video is linked below showcasing the dashboard's functionality and a full draft simulation.

## Data Sources  
Our dataset consists of **NFL player statistics from the 2018-2024 seasons**, collected from reputable sources:
- **Pro Football Reference**: Player stats (passing, rushing, receiving, kicking, team defense, special teams).
- **FantasyPros**: Average Draft Position (ADP) data from ESPN, NFL.com, RTSports, and Sleeper.
- **StatMuse**: Advanced statistical insights, including 2-point conversions.
- **NFL.com**: Defensive scoring and team performance statistics.

## Repository Structure  
Our repository is structured into two primary branches, each with its own focus:

### `main` Branch  
This branch contains all deliverables, licensing, and documentation related to the project.  
**Main Files & Folders**:  
- `deliverables/` – Main directory for all deliverables.  
  - `DSCI591/` – Folder with all deliverables related to part 1 of the Capstone.
    - `Capstone_Part1_FinalDoc.pdf` - Final document containing links to the repo and all reports / presentations for part 1.
    - `DataAcquisitionAndPreprocessingReport.pdf` - Report documenting the acquisition and preprocessing process in part 1.
    - `EDAandFeatureEngineeringReport.pdf` - Report documenting our EDA and feature engineering in part 1.
    - `LaunchReport.pdf` - Launch report walking through our initial project idea.
    - `PitchPresentation.mp4` - Presentation pitching our initial project idea.
    - `Status_Presentation_Capstone_Final_video.mp4` - Final presentation for the first half of our Capstone.
  - `DSCI592/` - Folder with all deliverables related to part 2 of the Capstone.
    - `Capstone_Part2_LaunchPresentation.mp4` - Launch presentation documenting what was done in part 1 of the capstone and what is left to be done
    - `Capstone_Part2_LaunchReport.pdf` - Launch report documenting what was done in part 1 of the capstone and what is left to be done.
    - `Capstone_Part2_PitchPresentation.mp4` - Presentation documenting what we have done in the first half of part 2 of the capstone including initial modeling.
    - `MachineLearningPresentation.mp4` - Presentation walking through our modeling phase of this project.
- `.gitattributes` - Required for git lfs tracking for large files.
- `.gitignore` - Used to ignore pycache, DS Store, etc.
- `LICENSE` - MIT licensing
- `README.md` – Documentation for the entire project.  
- `requirements.txt` – Project requirements

### `acquisition-and-preprocessing` Branch  
This branch contains all scripts related to **data acquisition** and **initial dataset preparation**.  
**Main Files & Folders**:  
- `data/` – Directory for data storage.
  - `2018/` - 2018 data storage
  - `2019/` - 2019 data storage
  - `2020/` - 2020 data storage
  - `2021/` - 2021 data storage
  - `2022/` - 2022 data storage
  - `2023/` - 2023 data storage
  - `2024/` - 2024 data storage
  - `final_data/` - final data storage
- `retired_files/` - Directory for retired files
- `.gitignore` - Used to ignore pycache, DS Store, etc.
- `LICENSE` - MIT licensing
- `MainScraper.py` – Web scraper for pulling NFL & FantasyPros stats.
- `README.md` – Documentation for the entire project.  
- `add_context_variables.py` - Script to add engineered features like year over year trends and deltas.  
- `defense_points_allowed.py` – Calculates defensive points allowed.  
- `drop_multicollinearity.py` – Removes redundant correlated features.  
- `fantasy_points_calc.py` – Computes fantasy points based on various scoring methods.  
- `final_dataset_calc.py` – Collects data from 2018 - 2024.  
- `final_dataset_long_format.py` - Unions the 2018 - 2024 data in a long format.
- `player_experience.py` - Collects player experience and merges in long format data.
- `requirements.txt` – Project requirements

### `EDA` Branch  
This branch is dedicated to **data analysis, feature engineering, and preprocessing**.    
**Main Files & Folders**:  
- `data/` – Directory for data storage.
  - `2018/` - 2018 data storage
  - `2019/` - 2019 data storage
  - `2020/` - 2020 data storage
  - `2021/` - 2021 data storage
  - `2022/` - 2022 data storage
  - `2023/` - 2023 data storage
  - `2024/` - 2024 data storage
  - `final_data/` - final data storage  
- `.gitignore` – Ignore unnecessary files in commits. 
- `CategoricalandUnivariateAnalysis.ipynb` – Univariate and categorical data analysis.  
- `EDA1.ipynb` – Exploratory analysis on player statistics.  
- `EDA2.ipynb` / `EDA3.ipynb` – Additional exploratory analysis.  
- `EDA4.ipynb` - Descriptive analysis
- `LICENSE` - MIT licensing
- `README.md` – Documentation for the entire project.  
- `ReportVisualizations.ipynb` – Various data visualizations for reporting.  
- `ScalingDistributionsInvestigation.ipynb` – Scaling distributions for player stats.  
- `VIFInvestigation1.ipynb` / `VIFInvestigation2.ipynb` – Correlation and Variance Inflation Factor (VIF) analysis.  
- `investigate_3tm_2tm.ipynb` – Investigating multi-team players.  
- `requirements.txt` – Project requirements

### `modeling` Branch
This branch contains our ML models and the results of their training and testing.  
**Main Files & Folders**:
- `data/` – Directory for data storage.
  - `2018/` - 2018 data storage
  - `2019/` - 2019 data storage
  - `2020/` - 2020 data storage
  - `2021/` - 2021 data storage
  - `2022/` - 2022 data storage
  - `2023/` - 2023 data storage
  - `2024/` - 2024 data storage
  - `final_data/` - final data storage   
- `deep_neural_network/` - Deep Neural Network model and hyperparameter tuner.
  - `joblib_files/` - Folder to hold trained models
  - `dnn_model_PPR.py` - PPR scoring dnn model
  - `dnn_model_Standard.py` - Standard scoring dnn model
  - `hyperparameter_tuning.py` - Tuning file
- `graphs/` - Graphs related to modeling.
  - `deep_neural_network/` - Graphs related to dnn modeling
  - `lightgbm_regression/` - Graphs related to lightgbm modeling
  - `linear_regression/` - Graphs related to linear modeling
  - `mlp_regression/` - Graphs related to mlp modeling
  - `stacked_model/` - Graphs related to stacked modeling
  - `xgboost_regression/` - Graphs related to xgboost modeling
- `lightgbm_regression/` - LightGBM Regression model and hyperparameter tuner.
  - `joblib_files/` - Folder to hold trained models
  - `lightgbm_model_PPR.py` - PPR scoring lightgbm model
  - `lightgbm_model_Standard.py` - Standard scoring lightgbm model
- `logs/` - Results for model runs.
  - `deep_neural_network/` - Logs related to dnn modeling
  - `lightgbm_regression/` - Logs related to lightgbm modeling
  - `linear_regression/` - Logs related to linear modeling
  - `mlp_regression/` - Logs related to mlp modeling
  - `stacked_model/` - Logs related to stacked modeling
  - `xgboost_regression/` - Logs related to xgboost modeling
- `mlp_regression/` - Multilayer Perceptron Regressor model and hyperparameter tuner.
  - `joblib_files/` - Folder to hold trained models
  - `mlp_model_PPR.py` - PPR scoring mlp model
  - `mlp_model_Standard.py` - Standard scoring mlp model
  - `hyperparameter_tuning.py` - Tuning file
- `stacked_model/` - A stacked XGBoost + MLP model and hyperparameter tuner.
  - `joblib_files/` - Folder to hold trained models
  - `stacked_model_PPR.py` - PPR scoring stacked model
  - `stacked_model_Standard.py` - Standard scoring stacked model
- `utils/` - Utility functions/files.
  - `__init__.py` - File for class initialization.
  - `data_processing.py` - Class for data processing.
  - `model_evaluation.py` - Class for model evaluation.
  - `plotting.py` - Class for plotting.
- `xgboost_regression/` - XGBoost regression model and hyperparameter tuner.
  - `joblib_files/` - Folder to hold trained models
  - `xgboost_model_PPR.py` - PPR scoring xgboost model
  - `xgboost_model_Standard.py` - Standard scoring xgboost model
  - `hyperparameter_tuning.py` - Tuning file
- `.gitignore` - Used to ignore pycache, DS Store, etc.
- `LICENSE` - MIT licensing
- `README.md` – Documentation for the entire project.  
- `requirements.txt` - Project requirements


### `recommender_system` Branch
This branch contains the recommender system function and dashboard app.
- `data/` – Directory for data storage.
  - `2018/` - 2018 data storage
  - `2019/` - 2019 data storage
  - `2020/` - 2020 data storage
  - `2021/` - 2021 data storage
  - `2022/` - 2022 data storage
  - `2023/` - 2023 data storage
  - `2024/` - 2024 data storage
  - `final_data/` - final data storage
- `lightgbm_regression/` - LightGBM Regression model and hyperparameter tuner.
  - `joblib_files/` - Folder to hold trained models
  - `lightgbm_model_PPR.py` - PPR scoring lightgbm model
  - `lightgbm_model_Standard.py` - Standard scoring lightgbm model
  - `lightgbm_regressor.py` - Lightgbm class to call in recommender system.
- `logs/mlp_regression` - Results for MLP regression model.
- `mlp_regression/` - Multilayer Perceptron Regressor model and hyperparameter tuner.
  - `joblib_files/` - Folder to hold trained models
  - `mlp_model_PPR.py` - PPR scoring mlp model
  - `mlp_model_Standard.py` - Standard scoring mlp model
- `stacked_model/` - A stacked XGBoost + MLP model and hyperparameter tuner.
  - `joblib_files/` - Folder to hold trained models
  - `stacked_model_PPR.py` - PPR scoring stacked model
  - `stacked_model_Standard.py` - Standard scoring stacked model
- `.gitignore` - Used to ignore pycache, DS Store, etc.
- `LICENSE` - MIT licensing
- `README.md` – Documentation for the entire project.
- `app.py` - Dashboard app.
- `compare_stats.py` - File containing a custom visualization function.
- `recommender_system.py` - Recommender system.
- `requirements.txt` - Project requirements

## Steps Throughout Capstone

### 1. Data Acquisition & Preprocessing Pipeline
1. Run **final_dataset_calc.py** to pull 2018-2024 data:
   - Calls **MainScraper.py** to collect NFL player statistics from 2018-2024
   - Calls **fantasy_points_calc.py** to compute fantasy points for each season

2. Run **final_dataset_long_format.py**:
   - Combines all seasons into a single long-format dataset (one row per player-season)

3. Run **add_context_variables.py**:
   - Adds deltas, year-over-year metrics, and other contextual features

4. Run **player_experience.py**:
   - Scrapes and adds player years of experience data

5. Run **drop_multicollinearity.py**:
   - Removes redundant and collinear features to improve model performance

```sh
# Execute the full pipeline in order:
python final_dataset_calc.py
python final_dataset_long_format.py
python add_context_variables.py
python player_experience.py
python drop_multicollinearity.py
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
Run the **drop_multicollinearity.py** script (see step 1) to refine the dataset by:  
- Removing **redundant correlated variables** using Variance Inflation Factor (VIF) analysis.  
- Creating **new derived features** such as composite player performance metrics.  

### 4. Machine Learning Models  
- Model Training & Evaluation
  - MLP Regression, XGBoost, Linear Regression, Deep Neural Network, LightGBM, and Stacked (XGBoost + MLP) models.
- Automated Hyperparameter Tuning and Feature Selection.

```sh
python linear_regression/linear_model_PPR.py
# Can change model type and scoring type.
```

### 5. Recommender System Construction  
- Implemented a hybrid recommendation engine that combines:
  - Machine learning model predictions (MLP + XGBoost)
  - Rule-based logic grounded in fantasy draft heuristics (e.g., positional scarcity, best player available, roster limits)
- Dynamically adapts to draft inputs (round, roster needs, scoring format) to provide updated recommendations

### 6. Dashboard Development  
- Building an **interactive web-based UI** that allows users to input draft context and view top player recommendations in real time.  

```sh
python app.py
# Runs the dashboard
```

## Dashboard Demos

**Dashboard Functionality Demo**  
[Watch Here](https://youtu.be/S1IaT5Xy95Q)

**Full Draft Walkthrough**  
[Watch Here](https://youtu.be/gsA_mNDSpPE)

## Installation & Usage  

**1. Clone the repository**  
```sh
git clone https://github.com/caleb10miller/FantasyFootballHelper.git
cd FantasyFootballHelper
```  
**2. Switch to the desired branch**  
- **For Data Acquisition (Web Scraping):**  
  ```sh
  git checkout acquisition-and-preprocessing
  ```  
- **For EDA & Feature Engineering:**  
  ```sh
  git checkout EDA
  ```
- **For Model Training and Hyperparameter Tuning:**  
  ```sh
  git checkout modeling
  ```  
- **For Recommender System and Dashboard App:**  
  ```sh
  git checkout recommender_system
  ```  
  
**Install dependencies**  
```sh
pip install -r requirements.txt
```  

**Run Data Processing Scripts in Order**  
```sh
python final_dataset_calc.py
python final_dataset_long_format.py
python add_context_variables.py
python player_experience.py
python drop_multicollinearity.py
``` 

**Train and Test Models**  
```sh
# Linear Regression
python linear_regression/linear_model_PPR.py
python linear_regression/linear_model_Standard.py

# MLP Regression
python mlp_regression/mlp_model_PPR.py
python mlp_regression/mlp_model_Standard.py

# XGBoost
python xgboost_regression/xgboost_model_PPR.py
python xgboost_regression/xgboost_model_Standard.py

# Stacked
python stacked_model/stacked_model_PPR.py
python stacked_model/stacked_model_Standard.py

# LightGBM
python lightgbm_regression/lightgbm_model_PPR.py
python lightgbm_regression/lightgbm_model_Standard.py

# Deep Neural Network
python deep_neural_network/dnn_model_PPR.py
python deep_neural_network/dnn_model_Standard.py
```

**Run Dashboard App**  
```sh
python app.py
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
