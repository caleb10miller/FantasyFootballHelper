import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
import joblib
from datetime import datetime
import os
import sys

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.model_evaluation import evaluate_model, save_results, save_model
from utils.data_processing import prepare_data, prepare_next_season_data
from utils.plotting import plot_actual_vs_predicted

# === CONFIGURATION ===
input = input("Enter the scoring type (1 for PPR, 0 for Standard): ")
scoring_type = "PPR" if input == "1" else "Standard"
input_file = "data/final_data/nfl_stats_long_format_with_context_filtered.csv"   
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Create directories if they don't exist
os.makedirs("logs/linear_regression", exist_ok=True)  
os.makedirs("logs/linear_regression/hyperparameter_tuning", exist_ok=True)  
os.makedirs("linear_regression/joblib_files", exist_ok=True)  
os.makedirs("graphs/linear_regression", exist_ok=True)

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

# === PARAMETER GRID ===
param_grid = {
    'linear__fit_intercept': [True, False],
    'linear__positive': [True, False],
    'linear__n_jobs': [-1, 1]
}

# === TRY DIFFERENT SCALERS ===
scalers = {
    'StandardScaler': StandardScaler(),
    'MinMaxScaler': MinMaxScaler(),
    'RobustScaler': RobustScaler(),
    'Normalizer': Normalizer()
}

# Store results for each scaler
scaler_results = {}

print("\n=== COMPARING DIFFERENT SCALERS ===")
for scaler_name, scaler in scalers.items():
    print(f"\nTrying {scaler_name}...")
    
    # Prepare data with current scaler
    X_train, X_test, y_train, y_test, preprocessor = prepare_data(
        df=df_model,
        target_col=target_col,
        feature_cols=feature_cols,
        categorical_cols=categorical_cols,
        scaler=scaler
    )
    
    # Create df_test from X_test
    df_test = df_model.loc[X_test.index]
    
    # Create pipeline
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("linear", LinearRegression())
    ])
    
    # Grid search
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring='r2',
        n_jobs=-1,  
        verbose=0  
    )
    
    grid_search.fit(X_train, y_train)
    
    # Evaluate best model
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    # Evaluate model performance
    evaluate_model(y_test, y_pred, scoring_type)
    
    # Plot actual vs predicted values
    plot_actual_vs_predicted(
        y_test, 
        y_pred,
        title=f"Linear Regression Model - {scoring_type} Scoring ({scaler_name})",
        xlabel=f"Actual {scoring_type} Targets",
        ylabel=f"Predicted {scoring_type} Targets",
        save_path=f"graphs/linear_regression/actual_vs_predicted_{scoring_type}_{scaler_name}_{timestamp}.png"
    )
    
    # Store results
    scaler_results[scaler_name] = {
        'model': grid_search,
        'y_test': y_test,
        'y_pred': y_pred,
        'df_test': df_test,
        'best_params': grid_search.best_params_
    }

# Find best scaler
best_scaler_name = max(scaler_results, key=lambda x: scaler_results[x]['model'].best_score_)
best_scaler_result = scaler_results[best_scaler_name]

print(f"\n=== BEST SCALER: {best_scaler_name} ===")
print("\nBest parameters:")
for param, value in best_scaler_result['best_params'].items():
    print(f"{param}: {value}")

# === GENERATE PREDICTIONS FOR 2025 SEASON ===
if len(df_next_season) > 0:
    X_next_season = prepare_next_season_data(
        df_next_season=df_next_season,
        feature_cols=feature_cols,
        categorical_cols=categorical_cols
    )
    next_season_predictions = best_scaler_result['model'].best_estimator_.predict(X_next_season)
else:
    next_season_predictions = None

# === SAVE RESULTS ===
results_file = save_results(
    model=best_scaler_result['model'],
    y_test=best_scaler_result['y_test'],
    y_pred=best_scaler_result['y_pred'],
    df_test=best_scaler_result['df_test'],
    df_next_season=df_next_season,
    next_season_predictions=next_season_predictions,
    scoring_type=scoring_type,
    timestamp=timestamp,
    subdirectory="hyperparameter_tuning"
)

# === SAVE BEST MODEL ===
model_file = save_model(
    model=best_scaler_result['model'],
    scoring_type=scoring_type,
    timestamp=timestamp,
    subdirectory="hyperparameter_tuning"
)

print(f"\nResults saved to {results_file}")
print(f"Best model saved to {model_file}") 