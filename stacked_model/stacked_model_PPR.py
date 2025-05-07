import pandas as pd
import numpy as np
from sklearn.ensemble import StackingRegressor
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
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
os.makedirs("logs/stacked_model", exist_ok=True)  
os.makedirs(f"logs/stacked_model/{timestamp}", exist_ok=True)  
os.makedirs("stacked_model/joblib_files", exist_ok=True)  
os.makedirs("graphs/stacked_model", exist_ok=True)  

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
    scaler=MinMaxScaler()  # Override default StandardScaler with MinMaxScaler
)

# === BASE MODELS ===
mlp = MLPRegressor(
    hidden_layer_sizes=(125,),
    activation='tanh',
    alpha=0.125,
    batch_size=32,
    early_stopping=True,
    max_iter=1000,
    validation_fraction=0.1,
    random_state=42
)

xgb = XGBRegressor(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    min_child_weight=1,
    gamma=0,
    reg_alpha=0.1,
    reg_lambda=10,
    random_state=42,
    verbosity=0
)

# === STACKING REGRESSOR ===
stacked_model = StackingRegressor(
    estimators=[("mlp", mlp), ("xgb", xgb)],
    final_estimator=XGBRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    ),
    passthrough=True,  # feeds original features to final model
    n_jobs=-1
)

# === FULL PIPELINE ===
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("stack", stacked_model)
])

# === FIT MODEL ===
pipeline.fit(X_train, y_train)

# === EVALUATE ===
y_pred = pipeline.predict(X_test)

# Evaluate model
mse, rmse, r2 = evaluate_model(y_test, y_pred, scoring_type)

# Plot actual vs predicted values
plot_actual_vs_predicted(
    y_test, 
    y_pred,
    title=f"Stacked Model - {scoring_type} Scoring",
    xlabel=f"Actual {scoring_type} Targets",
    ylabel=f"Predicted {scoring_type} Targets",
    save_path=f"graphs/stacked_model/actual_vs_predicted_{scoring_type}_{timestamp}.png"
)

# === GENERATE PREDICTIONS FOR 2025 SEASON ===
if len(df_next_season) > 0:
    X_next_season = prepare_next_season_data(df_next_season, feature_cols, categorical_cols)
    next_season_predictions = pipeline.predict(X_next_season)
else:
    next_season_predictions = None

# === SAVE RESULTS ===
results_file = save_results(
    pipeline, y_test, y_pred, df.loc[X_test.index], 
    df_next_season, next_season_predictions, scoring_type, timestamp
)

# === SAVE MODEL ===
model_file = save_model(pipeline, scoring_type, timestamp)

print(f"\nResults saved to {results_file}")
print(f"Model saved to {model_file}")
