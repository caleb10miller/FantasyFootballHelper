import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.neural_network import MLPRegressor
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
input = "0"
scoring_type = "PPR" if input == "1" else "Standard"
input_file = "data/final_data/nfl_stats_long_format_with_context_filtered.csv"   
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Create directories if they don't exist
os.makedirs("logs/deep_neural_network", exist_ok=True)  
os.makedirs(f"logs/deep_neural_network/{timestamp}", exist_ok=True)  
os.makedirs("deep_neural_network/joblib_files", exist_ok=True)  
os.makedirs("graphs/deep_neural_network", exist_ok=True)

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
    df=df_model,
    target_col=target_col,
    feature_cols=feature_cols,
    categorical_cols=categorical_cols,
    scaler=MinMaxScaler()  # Override default StandardScaler with MinMaxScaler
)

# Create df_test from X_test
df_test = df_model.loc[X_test.index]

# === CREATE PIPELINE ===
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("mlp", MLPRegressor(random_state=42))
])

# === PARAMETER GRID ===
param_grid = {
    'mlp__hidden_layer_sizes': [
        (256, 128, 64),         
    ],
    'mlp__activation': ['relu'],  
    'mlp__alpha': [0.125],  
    'mlp__batch_size': [32],     
    'mlp__learning_rate': ['adaptive'],  
    'mlp__learning_rate_init': [0.001],  
    'mlp__max_iter': [2000],     
    'mlp__early_stopping': [True],
    'mlp__validation_fraction': [0.1],
    'mlp__n_iter_no_change': [15]
}

# === GRID SEARCH ===
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1,  
    verbose=2  
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
    title=f"DNN Model - {scoring_type} Scoring",
    xlabel=f"Actual {scoring_type} Targets",
    ylabel=f"Predicted {scoring_type} Targets",
    save_path=f"graphs/deep_neural_network/actual_vs_predicted_{scoring_type}_{timestamp}.png"
)

# === GENERATE PREDICTIONS FOR 2025 SEASON ===
if len(df_next_season) > 0:
    X_next_season = prepare_next_season_data(
        df_next_season=df_next_season,
        feature_cols=feature_cols,
        categorical_cols=categorical_cols
    )
    next_season_predictions = best_model.predict(X_next_season)
else:
    next_season_predictions = None

# === SAVE RESULTS ===
results_file = save_results(
    model=grid_search,
    y_test=y_test,
    y_pred=y_pred,
    df_test=df_test,
    df_next_season=df_next_season,
    next_season_predictions=next_season_predictions,
    scoring_type=scoring_type,
    timestamp=timestamp
)

# === SAVE BEST MODEL ===
model_file = save_model(
    model=grid_search,
    scoring_type=scoring_type,
    timestamp=timestamp
)

print(f"\nResults saved to {results_file}")
print(f"Best model saved to {model_file}")
