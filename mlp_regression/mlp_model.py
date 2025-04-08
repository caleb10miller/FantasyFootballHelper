import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

def load_data(data_path):
    """
    Load the long format data from CSV file.
    
    Args:
        data_path (str): Path to the data file
        
    Returns:
        pandas.DataFrame: Loaded data
    """
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"Data loaded with shape: {df.shape}")
    return df

def split_data(df):
    """
    Split data into train (2018-2022) and test (2023) sets.
    
    Args:
        df (pandas.DataFrame): Input data
        
    Returns:
        tuple: (train_df, test_df)
    """
    print("Splitting data into train (2018-2022) and test (2023) sets...")
    train_df = df[df['Season'].isin([2018, 2019, 2020, 2021, 2022])]
    test_df = df[df['Season'] == 2023]
    
    # Remove rows where target is NaN from training data
    train_df = train_df[train_df['Target_PPR'].notna()]
    
    print(f"Training data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")
    
    return train_df, test_df

def prepare_features(df, train_df, test_df, scoring_type):
    """
    Prepare features for model training and prediction.
    
    Args:
        df (pandas.DataFrame): Original data
        train_df (pandas.DataFrame): Training data
        test_df (pandas.DataFrame): Test data
        scoring_type (int): 0 for standard, 1 for PPR
        
    Returns:
        tuple: (X_train, y_train, X_test, y_test, feature_cols)
    """
    print("Preparing features...")
    
    # Prepare features (excluding target variables, non-feature columns, and Standard Fantasy Points)
    feature_cols = [col for col in df.columns if col not in 
                    ['Player Name', 'Season', 'Target_PPR', 'Target_Standard']]
    
    # Split into features and target based on scoring type
    X_train = train_df[feature_cols]
    y_train = train_df['Target_PPR'] if scoring_type == 1 else train_df['Target_Standard']
    X_test = test_df[feature_cols]
    y_test = test_df['Target_PPR'] if scoring_type == 1 else test_df['Target_Standard']
    
    # Fill NaN values with 0 (since NaN in football stats typically means 0)
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)
    
    # Handle categorical variables
    categorical_cols = ['Position', 'Team']
    X_train = pd.get_dummies(X_train, columns=categorical_cols)
    X_test = pd.get_dummies(X_test, columns=categorical_cols)
    
    # Ensure test set has all columns from training set
    missing_cols = set(X_train.columns) - set(X_test.columns)
    for col in missing_cols:
        X_test[col] = 0
    X_test = X_test[X_train.columns]
    
    print(f"Training features shape: {X_train.shape}")
    print(f"Test features shape: {X_test.shape}")
    
    return X_train, y_train, X_test, y_test, feature_cols

def scale_features(X_train, X_test):
    """
    Scale the features using StandardScaler.
    
    Args:
        X_train (pandas.DataFrame): Training features
        X_test (pandas.DataFrame): Test features
        
    Returns:
        tuple: (X_train_scaled, X_test_scaled, scaler)
    """
    print("Scaling features...")
    
    # Scale the features
    scaler = Normalizer()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, scaler

def train_model(X_train_scaled, y_train):
    """
    Train the MLP Regressor model.
    
    Args:
        X_train_scaled (numpy.ndarray): Scaled training features
        y_train (pandas.Series): Training target
        
    Returns:
        MLPRegressor: Trained model
    """
    print("Training MLP Regressor model...")
    
    model = MLPRegressor(
        hidden_layer_sizes=(100, 50),
        alpha=0.01,                  # <-- Regularization strength (try 0.001–0.1)
        early_stopping=True,         # <-- Stop when validation score stops improving
        validation_fraction=0.1,     # <-- Fraction of training data used as validation set
        n_iter_no_change=10,         # <-- Wait 10 epochs before stopping
        max_iter=1000,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)
    
    return model

def evaluate_model(model, X_train_scaled, y_train, X_test_scaled, y_test):
    """
    Evaluate the model on training data and test data.
    
    Args:
        model (MLPRegressor): Trained model
        X_train_scaled (numpy.ndarray): Scaled training features
        y_train (pandas.Series): Training target
        X_test_scaled (numpy.ndarray): Scaled test features
        y_test (pandas.Series): Test target
        
    Returns:
        tuple: (y_pred_train, mse_train, rmse_train, r2_train, y_pred_test, mse_test, rmse_test, r2_test)
    """
    print("Evaluating model on training data...")
    
    # Evaluate on training data
    y_pred_train = model.predict(X_train_scaled)
    mse_train = mean_squared_error(y_train, y_pred_train)
    rmse_train = np.sqrt(mse_train)
    r2_train = r2_score(y_train, y_pred_train)
    
    print("\nModel Performance on Training Data (2018-2022):")
    print(f"Mean Squared Error: {mse_train:.2f}")
    print(f"Root Mean Squared Error: {rmse_train:.2f}")
    print(f"R² Score: {r2_train:.2f}")

    # Handle NaN values in test data
    if np.isnan(y_test).any():
        print("Warning: NaN values found in test target. Filling with 0.")
        y_test = y_test.fillna(0)
    
    # Evaluate on test set
    y_pred_test = model.predict(X_test_scaled)
    mse_test = mean_squared_error(y_test, y_pred_test)
    rmse_test = np.sqrt(mse_test)
    r2_test = r2_score(y_test, y_pred_test)
    
    print("\nModel Performance on Test Set:")
    print(f"Mean Squared Error: {mse_test:.2f}")
    print(f"Root Mean Squared Error: {rmse_test:.2f}")
    print(f"R² Score: {r2_test:.2f}")
    
    return y_pred_train, mse_train, rmse_train, r2_train, y_pred_test, mse_test, rmse_test, r2_test

def make_predictions(model, X_test_scaled, test_df, scoring_type):
    """
    Make predictions for test data.
    
    Args:
        model (MLPRegressor): Trained model
        X_test_scaled (numpy.ndarray): Scaled test features
        test_df (pandas.DataFrame): Test data
        scoring_type (int): 0 for standard, 1 for PPR
        
    Returns:
        tuple: (y_pred, predictions_df)
    """
    
    # Make predictions for 2023 data (to predict 2024)
    y_pred = model.predict(X_test_scaled)
    
    # Create a copy of test_df to avoid SettingWithCopyWarning
    predictions_df = test_df.copy()
    predictions_df['Predicted_2024_PPR' if scoring_type == 1 else 'Predicted_2024_Standard'] = y_pred
    
    # Sort by predicted points and show top 10
    top_10_predicted = predictions_df.sort_values('Predicted_2024_PPR' if scoring_type == 1 else 'Predicted_2024_Standard', ascending=False).head(10)
    print("\nTop 10 Predicted Fantasy Points for 2024:")
    print(top_10_predicted[['Player Name', 'Position', 'Team', 'Target_PPR' if scoring_type == 1 else 'Target_Standard', 'Predicted_2024_PPR' if scoring_type == 1 else 'Predicted_2024_Standard']].to_string(index=False))

    top_10_actual = predictions_df.sort_values('Target_PPR' if scoring_type == 1 else 'Target_Standard', ascending=False).head(10)
    print("\nTop 10 Actual Fantasy Points for 2024:")
    print(top_10_actual[['Player Name', 'Position', 'Team', 'Target_PPR' if scoring_type == 1 else 'Target_Standard', 'Predicted_2024_PPR' if scoring_type == 1 else 'Predicted_2024_Standard']].to_string(index=False))
    

    return y_pred, predictions_df

def get_feature_importance(model, X_train):
    """
    Get feature importance from the model.
    
    Args:
        model (MLPRegressor): Trained model
        X_train (pandas.DataFrame): Training features
        
    Returns:
        pandas.DataFrame: Feature importance
    """
    print("Calculating feature importance...")
    
    # Get feature importance (using the absolute values of the weights in the first layer)
    feature_importance = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': np.abs(model.coefs_[0]).mean(axis=1)
    })
    feature_importance = feature_importance.sort_values('Importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10))
    
    return feature_importance

def create_visualizations(y_train, y_pred_train, y_pred, feature_importance, output_dir, scoring_type):
    """
    Create visualizations for model evaluation and predictions.
    
    Args:
        y_train (pandas.Series): Training target
        y_pred_train (numpy.ndarray): Training predictions
        y_pred (numpy.ndarray): Test predictions
        feature_importance (pandas.DataFrame): Feature importance
        output_dir (str): Output directory for visualizations
        scoring_type (int): 0 for standard, 1 for PPR
    """
    print(f"Creating visualizations in {output_dir}...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set scoring type string for filenames
    scoring_str = "ppr" if scoring_type == 1 else "standard"
    
    # Plot training predictions distribution
    plt.figure(figsize=(10, 6))
    plt.hist(y_pred_train, bins=50, alpha=0.75)
    plt.xlabel('Predicted Fantasy Points')
    plt.ylabel('Number of Players')
    plt.title('Distribution of Predicted Fantasy Points (Training Data)')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/mlp_predictions_training_{scoring_str}.png')
    plt.close()
    
    # Plot predicted vs actual for training data
    plt.figure(figsize=(10, 8))
    plt.scatter(y_train, y_pred_train, alpha=0.5)
    plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
    plt.xlabel('Actual Fantasy Points')
    plt.ylabel('Predicted Fantasy Points')
    plt.title('Predicted vs Actual Fantasy Points (Training Data)')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/mlp_predicted_vs_actual_training_{scoring_str}.png')
    plt.close()
    
    # Plot predictions distribution
    plt.figure(figsize=(10, 6))
    plt.hist(y_pred, bins=50, alpha=0.75)
    plt.xlabel('Predicted 2024 Fantasy Points')
    plt.ylabel('Number of Players')
    plt.title('Distribution of Predicted 2024 Fantasy Points')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/mlp_predictions_test_{scoring_str}.png')
    plt.close()
    
    # Plot top 10 most important features
    plt.figure(figsize=(12, 6))
    sns.barplot(data=feature_importance.head(10), x='Importance', y='Feature')
    plt.title('Top 10 Most Important Features for Fantasy Points Prediction')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/mlp_feature_importance_2024_{scoring_str}.png')
    plt.close()
    
    print("Visualizations created successfully.")

def save_model(model, scaler, output_dir, scoring_type):
    """
    Save the model and scaler.
    
    Args:
        model (MLPRegressor): Trained model
        scaler (StandardScaler): Fitted scaler
        output_dir (str): Output directory for model files
        scoring_type (int): 0 for standard, 1 for PPR
    """
    print(f"Saving model and scaler to {output_dir}...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set scoring type string for filenames
    scoring_str = "ppr" if scoring_type == 1 else "standard"
    
    # Save the model and scaler
    joblib.dump(model, f'{output_dir}/mlp_model_2024_{scoring_str}.joblib')
    joblib.dump(scaler, f'{output_dir}/mlp_scaler_2024_{scoring_str}.joblib')
    
    print("Model and scaler saved successfully.")

def main():
    """
    Main function to run the MLP Regression model pipeline.
    """
    # Get user input for scoring type
    while True:
        try:
            scoring_type = int(input("Enter scoring type (0 for standard, 1 for PPR): "))
            if scoring_type in [0, 1]:
                break
            else:
                print("Please enter 0 for standard or 1 for PPR.")
        except ValueError:
            print("Please enter a valid integer (0 or 1).")
    
    # Set scoring type string for messages
    scoring_str = "PPR" if scoring_type == 1 else "Standard"
    print(f"\nRunning MLP Regression model pipeline for {scoring_str} scoring...")
    
    # Set paths
    data_path = 'data/final_data/nfl_stats_long_format.csv'
    graphs_dir = 'graphs/mlp_regression'
    model_dir = 'mlp_regression/joblib_files'
    
    # Create graphs directory if it doesn't exist
    os.makedirs(graphs_dir, exist_ok=True)
    
    # Run pipeline
    df = load_data(data_path)
    train_df, test_df = split_data(df)
    X_train, y_train, X_test, y_test, feature_cols = prepare_features(df, train_df, test_df, scoring_type)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    model = train_model(X_train_scaled, y_train)
    y_pred_train, mse_train, rmse_train, r2_train, y_pred_test, mse_test, rmse_test, r2_test = evaluate_model(model, X_train_scaled, y_train, X_test_scaled, y_test)
    y_pred, predictions_df = make_predictions(model, X_test_scaled, test_df, scoring_type)
    feature_importance = get_feature_importance(model, X_train)
    create_visualizations(y_train, y_pred_train, y_pred, feature_importance, graphs_dir, scoring_type)
    save_model(model, scaler, model_dir, scoring_type)
    
    print(f"\nMLP Regression model pipeline for {scoring_str} scoring completed successfully.")

if __name__ == "__main__":
    main() 