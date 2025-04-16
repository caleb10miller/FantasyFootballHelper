from sklearn.ensemble import StackingRegressor
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime

# === CONFIGURATION ===
input = input("Enter the scoring type (1 for PPR, 0 for Standard): ")
scoring_type = "PPR" if input == "1" else "Standard"
input_file = "data/final_data/nfl_stats_long_format_filtered.csv"   
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Create directories if they don't exist
os.makedirs("logs/stacked_model", exist_ok=True)  
os.makedirs(f"logs/stacked_model/{timestamp}", exist_ok=True)  
os.makedirs("stacked_model/joblib_files", exist_ok=True)  

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

# Convert categorical columns to string type
for col in categorical_cols:
    X_train[col] = X_train[col].astype(str)
    X_test[col] = X_test[col].astype(str)

# === PREPROCESSOR ===
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols)
    ]
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
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nStacked Model Performance:")
print(f"R^2: {r2:.3f}")
print(f"RMSE: {rmse:.1f}")

# === GENERATE PREDICTIONS FOR 2024 SEASON ===
print("\nGenerating predictions for 2024 season...")

# Filter data for 2024 season
df_next_season = df[df["Season"] == 2024].copy()

# === SAVE RESULTS ===
results_file = f"logs/stacked_model/{timestamp}/stacked_results_{scoring_type}_{timestamp}.txt"

with open(results_file, 'w') as f:

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
        next_season_predictions = pipeline.predict(X_next_season)
        
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
        predictions_file = f"logs/stacked_model/{timestamp}/predictions_{scoring_type}_{timestamp}.csv"
        df_next_season_predictions.to_csv(predictions_file, index=False)
        print(f"All predictions saved to {predictions_file}")

# === SAVE MODEL ===
model_path = f"stacked_model/joblib_files/stacked_model_{scoring_type}_{timestamp}.pkl"
joblib.dump(pipeline, model_path)
print(f"Stacked model saved to: {model_path}")
