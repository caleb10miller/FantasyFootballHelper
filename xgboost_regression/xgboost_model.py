import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from datetime import datetime
import os

# === CONFIGURATION ===
input = input("Enter the scoring type (1 for PPR, 0 for Standard): ")
scoring_type = "PPR" if input == "1" else "Standard"
input_file = "data/final_data/nfl_stats_long_format_filtered.csv"   
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Create directories if they don't exist
os.makedirs("logs", exist_ok=True)
os.makedirs("logs/xgboost_regression", exist_ok=True)  
os.makedirs(f"logs/xgboost_regression/{timestamp}", exist_ok=True)  
os.makedirs("xgboost_regression/joblib_files", exist_ok=True)  

# === LOAD DATA ===
df = pd.read_csv(input_file)

# === SELECT TARGET ===
target_col = "Target_PPR" if scoring_type == "PPR" else "Target_Standard"

# === FILTER TRAIN AND TEST ===
df_train = df[df["Season"].between(2018, 2022)].copy()
df_train = df_train[df_train[target_col].notna()]
df_test = df[df["Season"] == 2023].copy()
df_test = df_test[df_test[target_col].notna()]

# === DEFINE FEATURES ===
exclude_cols = ["Player Name", "Season", "Target_PPR", "Target_Standard", "PPR Fantasy Points Scored", "Standard Fantasy Points Scored"]
feature_cols = [col for col in df.columns if col not in exclude_cols]

X_train = df_train[feature_cols].copy()
X_test = df_test[feature_cols].copy()
y_train = df_train[target_col]
y_test = df_test[target_col]

# === FILL NA ===
X_train = X_train.fillna(0)
X_test = X_test.fillna(0)

# === ONE-HOT ENCODING for categorical columns ===
categorical_cols = ["Team", "Position"]
numerical_cols = [col for col in feature_cols if col not in categorical_cols]

# Convert categorical columns to string type to ensure consistent data types
for col in categorical_cols:
    X_train[col] = X_train[col].astype(str)
    X_test[col] = X_test[col].astype(str)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols)
    ]
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
    'xgb__subsample': [0.9],  
    'xgb__colsample_bytree': [1.0],  
    'xgb__min_child_weight': [2],  
    'xgb__gamma': [0],  
    'xgb__reg_alpha': [0.1],  
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

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nBest Model Performance:")
print(f"R^2: {r2:.3f}")
print(f"RMSE: {rmse:.1f}")

# === GENERATE PREDICTIONS FOR 2024 SEASON ===
print("\nGenerating predictions for 2024 season...")

# Filter data for 2024 season
df_next_season = df[df["Season"] == 2024].copy()

# === SAVE RESULTS ===
results_file = f"logs/xgboost_regression/{timestamp}/xgb_results_{scoring_type}_{timestamp}.txt"

with open(results_file, 'w') as f:
    f.write("Grid Search Results\n")
    f.write("==================\n\n")
    f.write("Best Parameters:\n")
    for param, value in grid_search.best_params_.items():
        f.write(f"{param}: {value}\n")
    
    f.write("\nTest Set Performance:\n")
    f.write(f"R^2: {r2:.3f}\n")
    f.write(f"RMSE: {rmse:.1f}\n")
    
    # Add 2024 season predictions to the log file
    f.write("\n\n2024 Season Predictions\n")
    f.write("=====================\n\n")
    
    if len(df_next_season) == 0:
        f.write("No data found for 2024 season. Please check your data.\n")
        print("No data found for 2024 season. Please check your data.")
    else:
        # Prepare features for prediction
        X_next_season = df_next_season[feature_cols].copy()
        X_next_season = X_next_season.fillna(0)
        
        # Convert categorical columns to string type
        for col in categorical_cols:
            X_next_season[col] = X_next_season[col].astype(str)
        
        # Generate predictions
        next_season_predictions = best_model.predict(X_next_season)
        
        # Create a DataFrame with predictions
        df_next_season_predictions = df_next_season[["Player Name", "Position", "Team", target_col]].copy()
        df_next_season_predictions["Predicted_Target"] = next_season_predictions
        
        # Sort by predicted target in descending order
        df_next_season_predictions = df_next_season_predictions.sort_values("Predicted_Target", ascending=False)
        
        # Get top 20 players
        top_20_players = df_next_season_predictions.head(20)
        
        # Write top 20 players to log file
        f.write("Top 20 Players for 2024 Season:\n")
        f.write("==============================\n\n")
        f.write("Rank | Player Name | Position | Team | Predicted Target\n")
        f.write("-----|-------------|----------|------|-----------------\n")
        
        for i, (_, row) in enumerate(top_20_players.iterrows(), 1):
            f.write(f"{i:4d} | {row['Player Name']:11s} | {row['Position']:8s} | {row['Team']:4s} | {row['Predicted_Target']:.1f}\n")
        
        # Save all predictions to a CSV file
        predictions_file = f"logs/xgboost_regression/{timestamp}/predictions_{scoring_type}_{timestamp}.csv"
        df_next_season_predictions.to_csv(predictions_file, index=False)
        print(f"All predictions saved to {predictions_file}")

# === SAVE BEST MODEL ===
model_file = f"xgboost_regression/joblib_files/xgb_pipeline_{scoring_type}_{timestamp}.pkl"
joblib.dump(best_model, model_file)

print(f"\nResults saved to {results_file}")
print(f"Best model saved to {model_file}")