import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from datetime import datetime
import os

# === CONFIGURATION ===
input = input("Enter the scoring type (1 for PPR, 0 for Standard): ")
scoring_type = "PPR" if input == "1" else "Standard"
input_file = "data/final_data/nfl_stats_long_format_with_context_filtered.csv"   

# Create directories if they don't exist
os.makedirs("logs", exist_ok=True)
os.makedirs("logs/linear_regression", exist_ok=True)  
os.makedirs("logs/linear_regression/hyperparameter_tuning", exist_ok=True)  
os.makedirs("linear_regression/joblib_files", exist_ok=True)  

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
exclude_cols = ["Player Name", 
                "Season", 
                "Target_PPR", 
                "Target_Standard", 
                "PPR Fantasy Points Scored", 
                "Standard Fantasy Points Scored", 
                "Delta_PPR_Fantasy_Points" if scoring_type == "Standard" else "Delta_Standard_Fantasy_Points"]
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
    
    # Create preprocessor with current scaler
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", scaler, numerical_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols)
        ]
    )
    
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
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"{scaler_name} Performance:")
    print(f"R^2: {r2:.3f}")
    print(f"RMSE: {rmse:.1f}")
    
    # Store results
    scaler_results[scaler_name] = {
        'r2': r2,
        'rmse': rmse,
        'best_params': grid_search.best_params_,
        'best_model': best_model
    }

# Find best scaler
best_scaler_name = max(scaler_results, key=lambda x: scaler_results[x]['r2'])
best_scaler_result = scaler_results[best_scaler_name]
best_model = best_scaler_result['best_model']

print(f"\n=== BEST SCALER: {best_scaler_name} ===")
print(f"R^2: {best_scaler_result['r2']:.3f}")
print(f"RMSE: {best_scaler_result['rmse']:.1f}")
print("\nBest parameters:")
for param, value in best_scaler_result['best_params'].items():
    print(f"{param}: {value}")

# === GENERATE PREDICTIONS FOR 2024 SEASON ===
print("\nGenerating predictions for 2024 season...")

# Filter data for 2024 season
df_next_season = df[df["Season"] == 2024].copy()

# === SAVE RESULTS ===
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_file = f"logs/linear_regression/hyperparameter_tuning/hyperparameter_tuning_{scoring_type}_{timestamp}.txt"

with open(results_file, 'w') as f:
    f.write("Scaler Comparison Results\n")
    f.write("=======================\n\n")
    
    for scaler_name, result in scaler_results.items():
        f.write(f"{scaler_name}:\n")
        f.write(f"R^2: {result['r2']:.3f}\n")
        f.write(f"RMSE: {result['rmse']:.1f}\n\n")
    
    f.write(f"\nBest Scaler: {best_scaler_name}\n")
    f.write("==================\n\n")
    f.write("Best Parameters:\n")
    for param, value in best_scaler_result['best_params'].items():
        f.write(f"{param}: {value}\n")
    
    f.write("\nTest Set Performance:\n")
    f.write(f"R^2: {best_scaler_result['r2']:.3f}\n")
    f.write(f"RMSE: {best_scaler_result['rmse']:.1f}\n")
    
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
        df_next_season_predictions = df_next_season[["Player Name", "Position", "Team"]].copy()
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
        predictions_file = f"logs/linear_regression/hyperparameter_tuning/predictions_{scoring_type}_{timestamp}.csv"
        df_next_season_predictions.to_csv(predictions_file, index=False)
        print(f"All predictions saved to {predictions_file}")

# === SAVE BEST MODEL ===
model_file = f"linear_regression/joblib_files/linear_pipeline_{best_scaler_name}_{scoring_type}_{timestamp}.pkl"
joblib.dump(best_model, model_file)

print(f"\nResults saved to {results_file}")
print(f"Best model saved to {model_file}") 