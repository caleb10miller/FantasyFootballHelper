import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestRegressor
import joblib
from datetime import datetime
import os
import matplotlib.pyplot as plt

# === CONFIGURATION ===
input = "1"
scoring_type = "PPR" if input == "1" else "Standard"
input_file = "data/final_data/nfl_stats_long_format_with_context_filtered.csv"   
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Create directories if they don't exist
os.makedirs("logs", exist_ok=True)
os.makedirs("logs/mlp_regression", exist_ok=True)  
os.makedirs(f"logs/mlp_regression/{timestamp}", exist_ok=True)  
os.makedirs("mlp_regression/joblib_files", exist_ok=True)  
os.makedirs("graphs", exist_ok=True)

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
exclude_cols = ["Player Name", "Season", "Target_PPR", "Target_Standard", "PPR Fantasy Points Scored", "Standard Fantasy Points Scored", "Team",
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
categorical_cols = ["Position"] #["Team", "Position"]
numerical_cols = [col for col in feature_cols if col not in categorical_cols]

# Convert categorical columns to string type to ensure consistent data types
for col in categorical_cols:
    X_train[col] = X_train[col].astype(str)
    X_test[col] = X_test[col].astype(str)

# === CREATE PREPROCESSING PIPELINE ===
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols)
    ]
)

# === FEATURE SELECTION WITH RFECV ===
# We'll use RandomForest as the estimator for RFECV since it's more stable than MLP
rfecv = RFECV(
    estimator=RandomForestRegressor(n_estimators=100, random_state=42),
    step=1,
    cv=5,
    scoring='r2',
    n_jobs=-1,
    verbose=2
)

# === CREATE PIPELINE ===
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("feature_selector", rfecv),
    ("mlp", MLPRegressor(random_state=42))
])

# === PARAMETER GRID ===
param_grid = {
    'mlp__activation': ['relu'],
    'mlp__alpha': [0.01],
    'mlp__batch_size': [32],
    'mlp__early_stopping': [True],
    'mlp__hidden_layer_sizes': [(150,)],
    'mlp__learning_rate_init': [0.001],
    'mlp__max_iter': [1000],
    'mlp__n_iter_no_change': [10],
    'mlp__validation_fraction': [0.1],
    'mlp__solver': ['adam']
}

# === GRID SEARCH ===
print("\nStarting Grid Search with RFECV feature selection...")
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

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nBest Model Performance:")
print(f"R^2: {r2:.3f}")
print(f"RMSE: {rmse:.1f}")

# === PLOT RFECV RESULTS ===
feature_selector = best_model.named_steps['feature_selector']
n_features_selected = feature_selector.n_features_
print(f"\nOptimal number of features selected: {n_features_selected}")

# Plot number of features vs. cross-validated R-squared
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(feature_selector.cv_results_['mean_test_score']) + 1),
         feature_selector.cv_results_['mean_test_score'], 'b-', label='CV Score')
plt.xlabel('Number of Features')
plt.ylabel('R-squared Score')
plt.title('Feature Selection (RFECV)\nNumber of Features vs. Performance')
plt.grid(True)
plt.legend(loc='best')
plt.tight_layout()
plt.savefig(f'graphs/rfecv_feature_selection_{scoring_type}_{timestamp}.png')
plt.close()

# Get selected feature names
cat_cols = categorical_cols
cat_categories = best_model.named_steps['preprocessor'].named_transformers_['cat'].categories_
all_feature_names = (
    numerical_cols +
    [f"{col}_{val}" for col, cats in zip(cat_cols, cat_categories) for val in cats]
)
selected_features = [name for name, selected in 
                    zip(all_feature_names, feature_selector.support_) if selected]

# === GENERATE PREDICTIONS FOR 2024 SEASON ===
print("\nGenerating predictions for 2024 season...")

# Filter data for 2024 season
df_next_season = df[df["Season"] == 2024].copy()

# === SAVE RESULTS ===
results_file = f"logs/mlp_regression/{timestamp}/mlp_results_{scoring_type}_{timestamp}.txt"

with open(results_file, 'w') as f:
    f.write("Grid Search Results\n")
    f.write("==================\n\n")
    f.write("Best Parameters:\n")
    for param, value in grid_search.best_params_.items():
        f.write(f"{param}: {value}\n")
    
    f.write("\nFeature Selection Results:\n")
    f.write(f"Optimal number of features: {n_features_selected}\n\n")
    f.write("Selected Features:\n")
    for feature in selected_features:
        f.write(f"- {feature}\n")
    
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
        df_next_season_predictions = df_next_season[["Player Name", "Position", target_col]].copy()
        df_next_season_predictions["Predicted_Target"] = next_season_predictions
        
        # Sort by predicted target in descending order
        df_next_season_predictions = df_next_season_predictions.sort_values("Predicted_Target", ascending=False)
        
        # Get top 20 players
        top_20_players = df_next_season_predictions.head(20)
        
        # Write top 20 players to log file
        f.write("Top 20 Players for 2024 Season:\n")
        f.write("==============================\n\n")
        f.write("Rank | Player Name | Position | Predicted Target\n")
        f.write("-----|-------------|----------|-----------------\n")
        
        for i, (_, row) in enumerate(top_20_players.iterrows(), 1):
            f.write(f"{i:4d} | {row['Player Name']:11s} | {row['Position']:8s} | {row['Predicted_Target']:.1f}\n")
        
        # Save all predictions to a CSV file
        predictions_file = f"logs/mlp_regression/{timestamp}/predictions_{scoring_type}_{timestamp}.csv"
        df_next_season_predictions.to_csv(predictions_file, index=False)
        print(f"All predictions saved to {predictions_file}")

# === SAVE BEST MODEL ===
model_file = f"mlp_regression/joblib_files/mlp_pipeline_{scoring_type}_{timestamp}.pkl"
joblib.dump(best_model, model_file)

print(f"\nResults saved to {results_file}")
print(f"Best model saved to {model_file}")
print(f"Feature selection plot saved to graphs/rfecv_feature_selection_{scoring_type}_{timestamp}.png")
