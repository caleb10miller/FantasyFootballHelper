import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def prepare_data(df, target_col, feature_cols, categorical_cols, test_size=0.2, random_state=42, scaler=None):
    """
    Prepare data for model training by splitting and preprocessing.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    target_col : str
        Name of the target column
    feature_cols : list
        List of feature column names
    categorical_cols : list
        List of categorical column names
    test_size : float, optional
        Proportion of data to use for testing
    random_state : int, optional
        Random seed for reproducibility
    scaler : object, optional
        Scaler to use for numerical features. If None, uses StandardScaler
        
    Returns:
    --------
    tuple
        X_train, X_test, y_train, y_test, preprocessor
    """
    # Split features and target
    X = df[feature_cols].copy()
    y = df[target_col]
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Fill NA values
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)
    
    # Convert categorical columns to string type
    for col in categorical_cols:
        X_train[col] = X_train[col].astype(str)
        X_test[col] = X_test[col].astype(str)
    
    # Create preprocessor
    numerical_cols = [col for col in feature_cols if col not in categorical_cols]
    if scaler is None:
        scaler = StandardScaler()
        
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", scaler, numerical_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols)
        ]
    )
    
    return X_train, X_test, y_train, y_test, preprocessor

def prepare_next_season_data(df_next_season, feature_cols, categorical_cols):
    """
    Prepare next season's data for prediction.
    
    Parameters:
    -----------
    df_next_season : pandas.DataFrame
        DataFrame containing next season's data
    feature_cols : list
        List of feature column names
    categorical_cols : list
        List of categorical column names
        
    Returns:
    --------
    pandas.DataFrame
        Prepared features for prediction
    """
    X_next_season = df_next_season[feature_cols].copy()
    X_next_season = X_next_season.fillna(0)
    
    # Convert categorical columns to string type
    for col in categorical_cols:
        X_next_season[col] = X_next_season[col].astype(str)
    
    return X_next_season 