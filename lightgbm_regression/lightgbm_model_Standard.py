import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import r2_score, mean_squared_error
import lightgbm as lgb
import joblib
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LightGBMRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=6, 
                 num_leaves=31, min_child_samples=20, subsample=0.8, 
                 colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1,
                 min_child_weight=1, min_split_gain=0.0):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.num_leaves = num_leaves
        self.min_child_samples = min_child_samples
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.min_child_weight = min_child_weight
        self.min_split_gain = min_split_gain
        self.model = None
        self.scaler = StandardScaler()
        
    def get_params(self, deep=True):
        return {
            'n_estimators': self.n_estimators,
            'learning_rate': self.learning_rate,
            'max_depth': self.max_depth,
            'num_leaves': self.num_leaves,
            'min_child_samples': self.min_child_samples,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'reg_alpha': self.reg_alpha,
            'reg_lambda': self.reg_lambda,
            'min_child_weight': self.min_child_weight,
            'min_split_gain': self.min_split_gain
        }
        
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
        
    def fit(self, X, y):
        # Scale numerical features
        X_scaled = self.scaler.fit_transform(X)
        
        # Create LightGBM dataset
        train_data = lgb.Dataset(X_scaled, label=y)
        
        # Set up parameters
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'n_estimators': self.n_estimators,
            'learning_rate': self.learning_rate,
            'max_depth': self.max_depth,
            'num_leaves': self.num_leaves,
            'min_child_samples': self.min_child_samples,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'reg_alpha': self.reg_alpha,
            'reg_lambda': self.reg_lambda,
            'min_child_weight': self.min_child_weight,
            'min_split_gain': self.min_split_gain,
            'verbose': -1
        }
        
        # Train the model
        self.model = lgb.train(params, train_data)
        return self
    
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def score(self, X, y):
        y_pred = self.predict(X)
        return r2_score(y, y_pred)

def load_data():
    """Load and preprocess the data"""
    logger.info("Loading data...")
    data_path = os.path.join('data', 'final_data', 'nfl_stats_long_format_with_context_filtered.csv')
    df = pd.read_csv(data_path)
    
    # Drop rows with missing target values
    df = df.dropna(subset=['Target_Standard'])
    
    # Create position-specific features
    df['QB_Features'] = df['Position'].apply(lambda x: 1 if x == 'QB' else 0)
    df['RB_Features'] = df['Position'].apply(lambda x: 1 if x == 'RB' else 0)
    df['WR_Features'] = df['Position'].apply(lambda x: 1 if x == 'WR' else 0)
    df['TE_Features'] = df['Position'].apply(lambda x: 1 if x == 'TE' else 0)
    
    # Create interaction features
    df['QB_Passing_Yards'] = df['QB_Features'] * df['Yards per Completion']
    df['RB_Rushing_Yards'] = df['RB_Features'] * df['Yards per Carry']
    df['WR_Receiving_Yards'] = df['WR_Features'] * df['Yards per Reception']
    df['TE_Receiving_Yards'] = df['TE_Features'] * df['Yards per Reception']
    
    # Select features and target
    feature_columns = [
        'Age', 'Games Played', 'Games Started',
        'Yards per Completion', 'Yards per Passing Touchdown',
        'Attempts per Completion', 'Passing Touchdowns',
        'Interceptions Thrown', 'Rushing Attempts',
        'Yards per Rushing Touchdown', 'Yards per Carry',
        'Rushing Touchdowns', 'Targets', 'Yards per Reception',
        'Yards per Receiving Touchdown', 'Catch Percentage',
        'Receiving Touchdowns', 'Fumbles', 'Field Goals Made',
        'Field Goal Percentage', 'Total Touchdowns Allowed',
        'Special Teams Impact', 'ST_Safeties',
        'ST_Special Teams Touchdowns', 'XP2',
        'Average ADP', 'Positional ADP',
        'Delta_Passing_Yards', 'Delta_Rushing_Yards',
        'Delta_Receiving_Yards', 'Delta_Passing_Touchdowns',
        'Delta_Rushing_Touchdowns', 'Delta_Receiving_Touchdowns',
        'Delta_Interceptions_Thrown', 'Delta_Fumbles',
        'Delta_Field_Goals_Made', 'Delta_Extra_Points_Made',
        'Delta_Average_ADP', 'Rolling_3_Year_PPR_Fantasy_Points',
        'Rolling_3_Year_Standard_Fantasy_Points',
        'QB_Features', 'RB_Features', 'WR_Features', 'TE_Features',
        'QB_Passing_Yards', 'RB_Rushing_Yards', 'WR_Receiving_Yards', 'TE_Receiving_Yards'
    ]
    
    # Convert categorical variables
    df['Position'] = pd.Categorical(df['Position']).codes
    
    X = df[feature_columns]
    y = df['Target_Standard']
    
    return X, y

def main():
    # Load data
    X, y = load_data()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Define parameter grid for grid search
    param_grid = {
        'n_estimators': [100],
        'learning_rate': [0.05],
        'max_depth': [6],
        'num_leaves': [35],
        'min_child_samples': [10],
        'subsample': [0.8],
        'colsample_bytree': [0.8],
        'reg_alpha': [0.1],
        'reg_lambda': [0.1]
    }
    
    # Create base model
    base_model = LightGBMRegressor()
    
    # Perform grid search
    logger.info("Starting grid search...")
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=3,  # Reduced from 5 to 3 folds
        scoring='r2',
        n_jobs=-1,
        verbose=2
    )
    
    grid_search.fit(X_train, y_train)
    
    # Get best parameters and model
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_
    
    # Evaluate on test set
    y_pred = best_model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    # Log results
    logger.info(f"Best parameters: {best_params}")
    logger.info(f"R² score: {r2:.3f}")
    logger.info(f"RMSE: {rmse:.2f}")
    
    # Save the model
    model_path = os.path.join('lightgbm_regression', 'lightgbm_model.joblib')
    joblib.dump(best_model, model_path)
    logger.info(f"Model saved to {model_path}")

if __name__ == "__main__":
    main() 