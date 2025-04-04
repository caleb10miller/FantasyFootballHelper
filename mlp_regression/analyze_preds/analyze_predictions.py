import pandas as pd
import joblib

# Load predictions
test_df = pd.read_csv('../../data/final_data/nfl_stats_long_format.csv')
test_df = test_df[test_df['Season'] == 2024]

# Load model and scaler
model = joblib.load('../joblib_files/mlp_model_2025_ppr.joblib')

# Get feature columns
feature_cols = [col for col in test_df.columns if col not in 
                ['Player Name', 'Season', 'Target_PPR', 'Target_Standard', 
                 'PPR Fantasy Points Scored', 'Standard Fantasy Points Scored']]

# Prepare features
X_test = test_df[feature_cols]
X_test = X_test.fillna(0)
X_test = pd.get_dummies(X_test, columns=['Position', 'Team'])

# Load scaler
scaler = joblib.load('../joblib_files/scaler_2025_ppr.joblib')

# Ensure all columns exist
missing_cols = set(scaler.feature_names_in_) - set(X_test.columns)
for col in missing_cols:
    X_test[col] = 0
X_test = X_test[scaler.feature_names_in_]

# Make predictions
X_test_scaled = scaler.transform(X_test)
y_pred = model.predict(X_test_scaled)

# Add predictions to dataframe
predictions_df = test_df.copy()
predictions_df['Predicted_2025_PPR'] = y_pred

# Look at players predicted around 60 points
around_60 = predictions_df[
    (predictions_df['Predicted_2025_PPR'] > 55) & 
    (predictions_df['Predicted_2025_PPR'] < 65)
].sort_values('Predicted_2025_PPR')

print('\nNumber of players predicted between 55-65 PPR points:', len(around_60))
print('\nSample of players predicted around 60 PPR points:')
print(around_60[['Player Name', 'Position', 'Team', 'PPR Fantasy Points Scored', 'Predicted_2025_PPR']].head(10).to_string())

# Print position distribution for these players
print('\nPosition distribution for players predicted between 55-65 PPR points:')
print(around_60['Position'].value_counts())

# Print average stats for these players
print('\nAverage stats for players predicted between 55-65 PPR points:')
stats_cols = ['Games Played', 'Average ADP', 'Positional ADP']
print(around_60[stats_cols].mean()) 