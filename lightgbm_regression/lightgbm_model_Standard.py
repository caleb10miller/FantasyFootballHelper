import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, RegressorMixin
import lightgbm as lgb
import joblib
import os
import logging
from datetime import datetime
import sys
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.model_evaluation import evaluate_model, save_results, save_model
from utils.data_processing import prepare_data, prepare_next_season_data
from utils.plotting import plot_actual_vs_predicted

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === CONFIGURATION ===
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
scoring_type = "Standard"

# Create directories if they don't exist
os.makedirs("logs/lightgbm_regression", exist_ok=True)
os.makedirs(f"logs/lightgbm_regression/{timestamp}", exist_ok=True)
os.makedirs("lightgbm_regression/joblib_files", exist_ok=True)
os.makedirs("graphs/lightgbm_regression", exist_ok=True)

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
        self.preprocessor = None
        
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
        # Get numerical and categorical columns
        numerical_cols = [col for col in X.columns if col != 'Position']
        categorical_cols = ['Position']
        
        # Create preprocessor
        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", self.scaler, numerical_cols),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols)
            ]
        )
        
        # Transform features
        X_processed = self.preprocessor.fit_transform(X)
        
        # Create LightGBM dataset
        train_data = lgb.Dataset(X_processed, label=y)
        
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
        X_processed = self.preprocessor.transform(X)
        return self.model.predict(X_processed)
    
    def score(self, X, y):
        y_pred = self.predict(X)
        return r2_score(y, y_pred)

def main():
    # Load and prepare data
    df = pd.read_csv('data/final_data/nfl_stats_long_format_with_context_filtered_with_experience.csv')
    
    # Remove rows with NaN in target variable
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
    
    # Define feature columns
    feature_cols = [
        'Age', 'Games Played', 'Games Started', 'Years_of_Experience',
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
        'Position',  # Add Position back to features
        'QB_Features', 'RB_Features', 'WR_Features', 'TE_Features',
        'QB_Passing_Yards', 'RB_Rushing_Yards', 'WR_Receiving_Yards', 'TE_Receiving_Yards'
    ]
    
    # Prepare data using utility function
    X_train, X_test, y_train, y_test, preprocessor = prepare_data(
        df=df,
        target_col='Target_Standard',
        feature_cols=feature_cols,
        categorical_cols=['Position'],
        scaler=StandardScaler()
    )
    
    # Create df_test from X_test
    df_test = df.loc[X_test.index]
    
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
        cv=3,
        scoring='r2',
        n_jobs=-1,
        verbose=2
    )
    
    grid_search.fit(X_train, y_train)
    
    # Get best model
    best_model = grid_search.best_estimator_
    
    # Evaluate on test set
    y_pred = best_model.predict(X_test)
    
    # Evaluate model performance
    evaluate_model(y_test, y_pred, scoring_type)
    
    # Plot actual vs predicted values
    plot_actual_vs_predicted(
        y_test, 
        y_pred,
        title=f"LightGBM Model - {scoring_type} Scoring",
        xlabel=f"Actual {scoring_type} Targets",
        ylabel=f"Predicted {scoring_type} Targets",
        save_path=f"graphs/lightgbm_regression/actual_vs_predicted_{scoring_type}_{timestamp}.png"
    )
    
    # Prepare next season data
    df_next_season = df[df['Season'] == df['Season'].max()].copy()
    X_next_season = prepare_next_season_data(
        df_next_season=df_next_season,
        feature_cols=feature_cols,
        categorical_cols=['Position']
    )
    
    # Generate predictions for next season
    next_season_predictions = best_model.predict(X_next_season)
    
    # Save results and model
    results_file = save_results(
        model=grid_search,
        y_test=y_test,
        y_pred=y_pred,
        df_test=df_test,
        df_next_season=df_next_season,
        next_season_predictions=next_season_predictions,
        scoring_type=scoring_type,
        timestamp=timestamp
    )
    
    model_file = save_model(
        model=grid_search,
        scoring_type=scoring_type,
        timestamp=timestamp
    )
    
    logger.info(f"Results saved to {results_file}")
    logger.info(f"Best model saved to {model_file}")

if __name__ == "__main__":
    main() 