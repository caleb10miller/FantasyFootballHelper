import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
import sys
import os
from datetime import datetime

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.model_evaluation import evaluate_model, save_results, save_model
from utils.data_processing import prepare_data, prepare_next_season_data
from utils.plotting import plot_actual_vs_predicted
import joblib

# === CONFIGURATION ===
input = "0"
scoring_type = "PPR" if input == "1" else "Standard"
input_file = "data/final_data/nfl_stats_long_format_with_context_filtered.csv"   
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Create directories if they don't exist
os.makedirs("logs/linear_regression", exist_ok=True)  
os.makedirs(f"logs/linear_regression/{timestamp}", exist_ok=True)  
os.makedirs("linear_regression/joblib_files", exist_ok=True)  

# === LOAD DATA ===
df = pd.read_csv(input_file)

# === SELECT TARGET ===
target_col = "Target_PPR" if scoring_type == "PPR" else "Target_Standard"

# === FILTER TRAIN AND TEST ===
df_model = df[df["Season"].between(2018, 2023)].copy()
df_model = df_model[df_model[target_col].notna()]
df_next_season = df[df["Season"] == 2024].copy()  # Use 2024 stats to predict 2025

# === DEFINE FEATURES ===
exclude_cols = [
    "Player Name", 
    "Season", 
    "Target_PPR", 
    "Target_Standard", 
    "PPR Fantasy Points Scored", 
    "Standard Fantasy Points Scored",
    "Delta_PPR_Fantasy_Points" if scoring_type == "Standard" else "Delta_Standard_Fantasy_Points"
]
feature_cols = [col for col in df.columns if col not in exclude_cols]
categorical_cols = ["Team", "Position"]

# === PREPARE DATA ===
X_train, X_test, y_train, y_test, preprocessor = prepare_data(
    df=df_model,
    target_col=target_col,
    feature_cols=feature_cols,
    categorical_cols=categorical_cols,
    scaler=MinMaxScaler()
)

# === CREATE PIPELINE ===
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("linear", LinearRegression())
])

# === PARAMETER GRID ===
param_grid = {
    'linear__fit_intercept': [False],
    'linear__positive': [False]
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

# Evaluate model performance
evaluate_model(y_test, y_pred, scoring_type)

# Plot actual vs predicted values
plot_actual_vs_predicted(
    y_test, 
    y_pred,
    title=f"Linear Regression Model - {scoring_type} Scoring",
    xlabel=f"Actual {scoring_type} Targets",
    ylabel=f"Predicted {scoring_type} Targets",
    save_path=f"graphs/linear_regression/actual_vs_predicted_{scoring_type}_{timestamp}.png"
)

# === SAVE RESULTS ===
results_file = save_results(
    model=grid_search,
    y_test=y_test,
    y_pred=y_pred,
    df_test=df_model.loc[X_test.index],
    df_next_season=df_next_season,
    next_season_predictions=None,  # Will be generated below
    scoring_type=scoring_type,
    timestamp=timestamp
)

# === GENERATE PREDICTIONS FOR 2025 SEASON ===
if len(df_next_season) > 0:
    # Prepare next season data
    X_next_season = prepare_next_season_data(
        df_next_season=df_next_season,
        feature_cols=feature_cols,
        categorical_cols=categorical_cols
    )
    
    # Generate predictions
    next_season_predictions = best_model.predict(X_next_season)
    
    # Update results with next season predictions
    save_results(
        model=grid_search,
        y_test=y_test,
        y_pred=y_pred,
        df_test=df_model.loc[X_test.index],
        df_next_season=df_next_season,
        next_season_predictions=next_season_predictions,
        scoring_type=scoring_type,
        timestamp=timestamp
    )

# === SAVE BEST MODEL ===
model_file = save_model(best_model, scoring_type, timestamp)

print(f"\nResults saved to {results_file}")
print(f"Best model saved to {model_file}")