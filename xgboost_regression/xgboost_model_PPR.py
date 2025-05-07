import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import sys
import os
from datetime import datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.plotting import plot_actual_vs_predicted
from utils.data_processing import prepare_data, prepare_next_season_data
from utils.model_evaluation import evaluate_model, save_results, save_model

# === CONFIGURATION ===
input = "1"
scoring_type = "PPR" if input == "1" else "Standard"
input_file = "data/final_data/nfl_stats_long_format_with_context_filtered.csv"   
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Create directories if they don't exist
os.makedirs("logs", exist_ok=True)
os.makedirs("logs/xgboost_regression", exist_ok=True)  
os.makedirs(f"logs/xgboost_regression/{timestamp}", exist_ok=True)  
os.makedirs("xgboost_regression/joblib_files", exist_ok=True)  
os.makedirs("graphs/xgboost_regression", exist_ok=True)  

# === LOAD DATA ===
df = pd.read_csv(input_file)

# === SELECT TARGET ===
target_col = "Target_PPR" if scoring_type == "PPR" else "Target_Standard"

# === FILTER TRAIN AND TEST ===
df_model = df[df["Season"].between(2018, 2023)].copy()
df_model = df_model[df_model[target_col].notna()]
df_next_season = df[df["Season"] == 2024].copy()  # Use 2024 stats to predict 2025

# === DEFINE FEATURES ===
exclude_cols = ["Player Name", "Season", "Target_PPR", "Target_Standard", "PPR Fantasy Points Scored", "Standard Fantasy Points Scored",
                "Delta_PPR_Fantasy_Points" if scoring_type == "Standard" else "Delta_Standard_Fantasy_Points"]
feature_cols = [col for col in df.columns if col not in exclude_cols]
categorical_cols = ["Team", "Position"]

# === PREPARE DATA ===
X_train, X_test, y_train, y_test, preprocessor = prepare_data(
    df_model, target_col, feature_cols, categorical_cols,
    scaler=StandardScaler()
)

# === CREATE PIPELINE ===
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("xgb", XGBRegressor(random_state=42))
])

# === PARAMETER GRID ===
param_grid = {
    'xgb__n_estimators': [200],  
    'xgb__max_depth': [4],  
    'xgb__learning_rate': [0.05],  
    'xgb__subsample': [0.85],  
    'xgb__colsample_bytree': [0.95],  
    'xgb__min_child_weight': [2],  
    'xgb__gamma': [0.01],  
    'xgb__reg_alpha': [0.05],  
    'xgb__reg_lambda': [10]  
}

# === GRID SEARCH ===
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1,  
    verbose=0  
)

grid_search.fit(X_train, y_train)

# === PRINT BEST PARAMETERS ===
print("\nBest parameters found:")
for param, value in grid_search.best_params_.items():
    print(f"{param}: {value}")

# === EVALUATE BEST MODEL ===
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Evaluate model
mse, rmse, r2 = evaluate_model(y_test, y_pred, scoring_type)

# Plot actual vs predicted values
plot_actual_vs_predicted(
    y_test, 
    y_pred,
    title=f"XGBoost Model - {scoring_type} Scoring",
    xlabel=f"Actual {scoring_type} Targets",
    ylabel=f"Predicted {scoring_type} Targets",
    save_path=f"graphs/xgboost_regression/actual_vs_predicted_{scoring_type}_{timestamp}.png"
)

# === GENERATE PREDICTIONS FOR 2025 SEASON ===
if len(df_next_season) > 0:
    X_next_season = prepare_next_season_data(df_next_season, feature_cols, categorical_cols)
    next_season_predictions = best_model.predict(X_next_season)
else:
    next_season_predictions = None

# === SAVE RESULTS ===
results_file = save_results(
    grid_search, y_test, y_pred, df.loc[X_test.index], 
    df_next_season, next_season_predictions, scoring_type, timestamp
)

# === SAVE MODEL ===
model_file = save_model(best_model, scoring_type, timestamp)

print(f"\nResults saved to {results_file}")
print(f"Best model saved to {model_file}")