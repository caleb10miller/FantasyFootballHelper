import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from datetime import datetime
import os
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
import lightgbm as lgb

def evaluate_model(y_test, y_pred, scoring_type):
    """
    Evaluate model performance and return metrics.
    
    Parameters:
    -----------
    y_test : array-like
        True target values
    y_pred : array-like
        Predicted values
    scoring_type : str
        Type of scoring (e.g., "PPR" or "Standard")
        
    Returns:
    --------
    tuple
        mse, rmse, r2
    """
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print("\nModel Performance:")
    print(f"R^2: {r2:.3f}")
    print(f"RMSE: {rmse:.1f}")
    
    return mse, rmse, r2

def get_model_type(model):
    """
    Determine the type of model based on its components.
    
    Parameters:
    -----------
    model : estimator or GridSearchCV
        Fitted model object
        
    Returns:
    --------
    str
        Model type ('mlp_regression', 'deep_neural_network', 'xgboost_regression', 'linear_regression', 'lightgbm_regression', or 'stacked_model')
    """
    if hasattr(model, 'best_estimator_'):
        # For GridSearchCV objects
        estimator = model.best_estimator_
    else:
        # For Pipeline objects
        estimator = model
    
    # Get the last step of the pipeline
    if hasattr(estimator, 'steps'):
        last_step = estimator.steps[-1][1]
        if isinstance(last_step, XGBRegressor):
            return 'xgboost_regression'
        elif isinstance(last_step, LinearRegression):
            return 'linear_regression'
        elif hasattr(last_step, 'hidden_layer_sizes'):  # MLPRegressor
            # Check if it's a deep neural network (multiple hidden layers)
            if isinstance(last_step.hidden_layer_sizes, tuple) and len(last_step.hidden_layer_sizes) > 1:
                return 'deep_neural_network'
            else:
                return 'mlp_regression'
        elif isinstance(last_step, lgb.LGBMRegressor) or hasattr(last_step, 'model') and isinstance(last_step.model, lgb.Booster):
            return 'lightgbm_regression'
        else:
            return 'stacked_model'
    else:
        # If it's not a pipeline, check the estimator directly
        if isinstance(estimator, XGBRegressor):
            return 'xgboost_regression'
        elif isinstance(estimator, LinearRegression):
            return 'linear_regression'
        elif hasattr(estimator, 'hidden_layer_sizes'):  # MLPRegressor
            # Check if it's a deep neural network (multiple hidden layers)
            if isinstance(estimator.hidden_layer_sizes, tuple) and len(estimator.hidden_layer_sizes) > 1:
                return 'deep_neural_network'
            else:
                return 'mlp_regression'
        elif isinstance(estimator, lgb.LGBMRegressor) or hasattr(estimator, 'model') and isinstance(estimator.model, lgb.Booster):
            return 'lightgbm_regression'
        else:
            return 'stacked_model'

def save_results(model, y_test, y_pred, df_test, df_next_season, 
                next_season_predictions, scoring_type, timestamp, subdirectory=None):
    """
    Save model results and predictions to files.
    
    Parameters:
    -----------
    model : estimator or GridSearchCV
        Fitted model object (Pipeline or GridSearchCV)
    y_test : array-like
        True test values
    y_pred : array-like
        Predicted test values
    df_test : pandas.DataFrame
        Test dataframe
    df_next_season : pandas.DataFrame
        Next season dataframe
    next_season_predictions : array-like
        Predictions for next season
    scoring_type : str
        Type of scoring (e.g., "PPR" or "Standard")
    timestamp : str
        Timestamp string for file naming
    subdirectory : str, optional
        Subdirectory to save results in (e.g., "hyperparameter_tuning")
    """
    # Determine model type and set appropriate directories
    model_type = get_model_type(model)
    results_dir = f"logs/{model_type}"
    if subdirectory:
        results_dir = f"{results_dir}/{subdirectory}"
    results_dir = f"{results_dir}/{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Save results
    results_file = f"{results_dir}/{model_type}_results_{scoring_type}_{timestamp}.txt"
    
    with open(results_file, 'w') as f:
        if hasattr(model, 'best_params_'):
            f.write("Grid Search Results\n")
            f.write("==================\n\n")
            f.write("Best Parameters:\n")
            for param, value in model.best_params_.items():
                f.write(f"{param}: {value}\n")
        else:
            f.write("Model Parameters\n")
            f.write("===============\n\n")
            for param, value in model.get_params().items():
                f.write(f"{param}: {value}\n")
        
        # Save test set predictions
        df_predictions = pd.DataFrame({
            'Player Name': df_test['Player Name'],
            'Position': df_test['Position'],
            'Team': df_test['Team'],
            'Actual': y_test,
            'Predicted': y_pred
        })
        df_predictions = df_predictions.sort_values('Predicted', ascending=False)
        
        f.write("\n\nTest Set Predictions\n")
        f.write("===================\n\n")
        f.write("Top 20 Test Set Predictions:\n")
        f.write("---------------------------\n\n")
        f.write("Rank | Player Name | Position | Team | Actual | Predicted\n")
        f.write("-----|-------------|----------|------|---------|----------\n")
        
        for i, (_, row) in enumerate(df_predictions.head(20).iterrows(), 1):
            f.write(f"{i:4d} | {row['Player Name']:11s} | {row['Position']:8s} | {row['Team']:4s} | {row['Actual']:6.1f} | {row['Predicted']:8.1f}\n")
        
        predictions_file = f"{results_dir}/test_predictions_{timestamp}.csv"
        df_predictions.to_csv(predictions_file, index=False)
        
        # Save next season predictions
        if len(df_next_season) > 0 and next_season_predictions is not None:
            df_2025_pred = df_next_season[["Player Name", "Position", "Team"]].copy()
            df_2025_pred["Predicted_Target"] = next_season_predictions
            df_2025_pred = df_2025_pred.sort_values("Predicted_Target", ascending=False)
            
            f.write("\n\n2025 Season Predictions\n")
            f.write("=======================\n\n")
            f.write("Top 20 Players for 2025 Season:\n")
            f.write("==============================\n\n")
            f.write("Rank | Player Name | Position | Team | Predicted Target\n")
            f.write("-----|-------------|----------|------|-----------------\n")
            
            for i, (_, row) in enumerate(df_2025_pred.head(20).iterrows(), 1):
                f.write(f"{i:4d} | {row['Player Name']:11s} | {row['Position']:8s} | {row['Team']:4s} | {row['Predicted_Target']:.1f}\n")
            
            predictions_file_2025 = f"{results_dir}/predictions_2025.csv"
            df_2025_pred.to_csv(predictions_file_2025, index=False)
    
    return results_file

def save_model(model, scoring_type, timestamp, subdirectory=None):
    """
    Save the trained model to a file.
    
    Parameters:
    -----------
    model : estimator
        Trained model to save
    scoring_type : str
        Type of scoring (e.g., "PPR" or "Standard")
    timestamp : str
        Timestamp string for file naming
    subdirectory : str, optional
        Subdirectory to save model in (e.g., "hyperparameter_tuning")
        
    Returns:
    --------
    str
        Path to saved model file
    """
    # Determine model type and set appropriate directory
    model_type = get_model_type(model)
    model_dir = f"{model_type}/joblib_files"
    if subdirectory:
        model_dir = f"{model_dir}/{subdirectory}"
    os.makedirs(model_dir, exist_ok=True)
    
    model_file = f"{model_dir}/{model_type}_pipeline_{scoring_type}_{timestamp}.pkl"
    joblib.dump(model, model_file)
    
    return model_file 