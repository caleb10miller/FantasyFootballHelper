import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LinearRegression
import sys
import os
from datetime import datetime
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.plotting import plot_actual_vs_predicted
from utils.data_processing import prepare_data, prepare_next_season_data
from utils.model_evaluation import evaluate_model, save_results, save_model

# === CONFIGURATION ===
input = "1"
scoring_type = "PPR" if input == "1" else "Standard"
input_file = "data/final_data/nfl_stats_long_format_with_context_filtered_with_experience.csv"   
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Create directories if they don't exist
os.makedirs("logs", exist_ok=True)
os.makedirs("logs/mlp_regression", exist_ok=True)  
os.makedirs(f"logs/mlp_regression/{timestamp}", exist_ok=True)  
os.makedirs("mlp_regression/joblib_files", exist_ok=True)  
os.makedirs("graphs", exist_ok=True)
os.makedirs("graphs/mlp_regression", exist_ok=True)

# === LOAD DATA ===
df = pd.read_csv(input_file)

# === SELECT TARGET ===
target_col = "Target_PPR" if scoring_type == "PPR" else "Target_Standard"

# === FILTER TRAIN AND TEST ===
df_model = df[df["Season"].between(2018, 2023)].copy()
df_model = df_model[df_model[target_col].notna()]
df_next_season = df[df["Season"] == 2024].copy()  # Use 2024 stats to predict 2025

# === DEFINE FEATURES ===
exclude_cols = ["Player Name", "Season", "Target_PPR", "Target_Standard", "PPR Fantasy Points Scored", "Standard Fantasy Points Scored", "Team",
                "Delta_PPR_Fantasy_Points" if scoring_type == "Standard" else "Delta_Standard_Fantasy_Points"]
feature_cols = [col for col in df.columns if col not in exclude_cols]
categorical_cols = ["Position"]

# === PREPARE DATA ===
X_train, X_test, y_train, y_test, preprocessor = prepare_data(
    df_model, target_col, feature_cols, categorical_cols
)

# === CREATE PIPELINE ===
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("feature_selector", RFECV(
        estimator=LinearRegression(), 
        step=1, 
        cv=3,    
        scoring='r2',
        n_jobs=-1,
        verbose=2,
        min_features_to_select=10 
    )),
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

# Evaluate model
mse, rmse, r2 = evaluate_model(y_test, y_pred, scoring_type)

# Plot actual vs predicted values
plot_actual_vs_predicted(
    y_test, 
    y_pred,
    title=f"MLP Model with RFECV - {scoring_type} Scoring",
    xlabel=f"Actual {scoring_type} Targets",
    ylabel=f"Predicted {scoring_type} Targets",
    save_path=f"graphs/mlp_regression/actual_vs_predicted_rfecv_{scoring_type}_{timestamp}.png"
)

# Plot RFECV results
feature_selector = best_model.named_steps['feature_selector']
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(feature_selector.cv_results_['mean_test_score']) + 1),
         feature_selector.cv_results_['mean_test_score'], 'b-', label='CV Score')
plt.xlabel('Number of Features')
plt.ylabel('R-squared Score')
plt.title('Feature Selection (RFECV)\nNumber of Features vs. Performance')
plt.grid(True)
plt.legend(loc='best')
plt.tight_layout()
plt.savefig(f'graphs/mlp_regression/rfecv_feature_selection_{scoring_type}_{timestamp}.png')
plt.close()

# Get selected feature names
cat_cols = categorical_cols
cat_categories = best_model.named_steps['preprocessor'].named_transformers_['cat'].categories_
all_feature_names = (
    [col for col in feature_cols if col not in categorical_cols] +
    [f"{col}_{val}" for col, cats in zip(cat_cols, cat_categories) for val in cats]
)
selected_features = [name for name, selected in 
                    zip(all_feature_names, feature_selector.support_) if selected]

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

# Add selected features to results file
with open(results_file, 'a') as f:
    f.write("\nFeature Selection Results:\n")
    f.write("========================\n\n")
    f.write(f"Optimal number of features: {feature_selector.n_features_}\n\n")
    f.write("Selected Features:\n")
    for feature in selected_features:
        f.write(f"- {feature}\n")

# === SAVE BEST MODEL ===
model_file = save_model(best_model, scoring_type, timestamp)

print(f"\nResults saved to {results_file}")
print(f"Best model saved to {model_file}")
print(f"Feature selection plot saved to graphs/mlp_regression/rfecv_feature_selection_{scoring_type}_{timestamp}.png")