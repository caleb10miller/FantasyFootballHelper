import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.neural_network import MLPRegressor
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
os.makedirs("logs/mlp_regression", exist_ok=True)  
os.makedirs("logs/mlp_regression/hyperparameter_tuning", exist_ok=True)  
os.makedirs("mlp_regression/joblib_files", exist_ok=True)  

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

# Split train/test (80/20)
X = df_model[feature_cols].copy()
y = df_model[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
df_train = df_model.loc[X_train.index]
df_test = df_model.loc[X_test.index]

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
    'mlp__hidden_layer_sizes': [(150,)],  
    'mlp__activation': ['relu'],  
    'mlp__alpha': [0.009, 0.01, 0.011],  
    'mlp__learning_rate_init': [0.001],  
    'mlp__batch_size': [32],  
    'mlp__max_iter': [1000],
    'mlp__early_stopping': [True],
    'mlp__validation_fraction': [0.1],
    'mlp__n_iter_no_change': [10],
    'mlp__solver': ['adam']
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
        ("mlp", MLPRegressor(random_state=42))
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

# === SAVE RESULTS ===
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_file = f"logs/mlp_regression/hyperparameter_tuning/hyperparameter_tuning_{scoring_type}_{timestamp}.txt"

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
    
    # Add test set predictions to the log file
    f.write("\n\nTest Set Predictions\n")
    f.write("===================\n\n")
    
    df_predictions = pd.DataFrame({
        'Player Name': df_test['Player Name'],
        'Position': df_test['Position'],
        'Team': df_test['Team'],
        'Actual': y_test,
        'Predicted': best_model.predict(X_test)
    })
    df_predictions = df_predictions.sort_values('Predicted', ascending=False)
    
    f.write("Top 20 Test Set Predictions:\n")
    f.write("---------------------------\n\n")
    f.write("Rank | Player Name | Position | Team | Actual | Predicted\n")
    f.write("-----|-------------|----------|------|---------|----------\n")
    
    for i, (_, row) in enumerate(df_predictions.head(20).iterrows(), 1):
        f.write(f"{i:4d} | {row['Player Name']:11s} | {row['Position']:8s} | {row['Team']:4s} | {row['Actual']:6.1f} | {row['Predicted']:8.1f}\n")
    
    predictions_file = f"logs/mlp_regression/hyperparameter_tuning/test_predictions_{timestamp}.csv"
    df_predictions.to_csv(predictions_file, index=False)
    print(f"Test set predictions saved to {predictions_file}")
    
    # === GENERATE PREDICTIONS FOR 2025 SEASON ===
    f.write("\n\n2025 Season Predictions\n")
    f.write("=======================\n\n")
    
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
        df_2025_pred = df_next_season[["Player Name", "Position", "Team"]].copy()
        df_2025_pred["Predicted_Target"] = next_season_predictions
        
        # Sort by predicted target in descending order
        df_2025_pred = df_2025_pred.sort_values("Predicted_Target", ascending=False)
        
        # Write top 20 players to log file
        f.write("Top 20 Players for 2025 Season:\n")
        f.write("==============================\n\n")
        f.write("Rank | Player Name | Position | Team | Predicted Target\n")
        f.write("-----|-------------|----------|------|-----------------\n")
        
        for i, (_, row) in enumerate(df_2025_pred.head(20).iterrows(), 1):
            f.write(f"{i:4d} | {row['Player Name']:11s} | {row['Position']:8s} | {row['Team']:4s} | {row['Predicted_Target']:.1f}\n")
        
        # Save all predictions to a CSV file
        predictions_file_2025 = f"logs/mlp_regression/hyperparameter_tuning/predictions_2025.csv"
        df_2025_pred.to_csv(predictions_file_2025, index=False)
        print(f"2025 season predictions saved to {predictions_file_2025}")

# === SAVE BEST MODEL ===
model_file = f"mlp_regression/joblib_files/mlp_pipeline_{best_scaler_name}_{scoring_type}_{timestamp}.pkl"
joblib.dump(best_model, model_file)

print(f"\nResults saved to {results_file}")
print(f"Best model saved to {model_file}") 