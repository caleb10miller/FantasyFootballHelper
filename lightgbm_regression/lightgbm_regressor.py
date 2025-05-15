import lightgbm as lgb
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score

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