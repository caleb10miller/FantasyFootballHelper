import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

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

# Get top 20 highest predicted players
top_20 = predictions_df.sort_values('Predicted_2025_PPR', ascending=False).head(20)
print("\nTop 20 Predicted PPR Fantasy Points for 2025:")
print(top_20[['Player Name', 'Position', 'Team', 'PPR Fantasy Points Scored', 'Predicted_2025_PPR']].to_string(index=False))

# Calculate the increase in points from 2024 to predicted 2025
top_20['Points_Increase'] = top_20['Predicted_2025_PPR'] - top_20['PPR Fantasy Points Scored']
print("\nTop 20 Players with Largest Predicted Increase in PPR Points:")
print(top_20.sort_values('Points_Increase', ascending=False)[['Player Name', 'Position', 'Team', 'PPR Fantasy Points Scored', 'Predicted_2025_PPR', 'Points_Increase']].to_string(index=False))

# Analyze key stats for top predicted players
print("\nAverage Stats for Top 20 Predicted Players:")
key_stats = ['Age', 'Games Played', 'Games Started', 'Average ADP', 'Positional ADP', 
             'Carries*Yards', 'Receptions*Yards', 'Total Passing', 'Rushing Touchdowns', 
             'Receiving Touchdowns', 'Fumbles']
print(top_20[key_stats].mean())

# Compare with league averages
print("\nLeague Averages for Comparison:")
league_avg = test_df[key_stats].mean()
print(league_avg)

# Calculate how many standard deviations above average the top players are
z_scores = (top_20[key_stats].mean() - league_avg) / test_df[key_stats].std()
print("\nZ-Scores (Standard Deviations Above Average) for Top 20 Players:")
print(z_scores)

# Look at position distribution of top 20
print("\nPosition Distribution in Top 20 Predicted Players:")
print(top_20['Position'].value_counts())

# Analyze team distribution
print("\nTeam Distribution in Top 20 Predicted Players:")
print(top_20['Team'].value_counts())

# Create a visualization of the top 10 predicted players
plt.figure(figsize=(12, 8))
top_10 = top_20.head(10)
bars = plt.bar(top_10['Player Name'], top_10['Predicted_2025_PPR'])
plt.xticks(rotation=45, ha='right')
plt.ylabel('Predicted 2025 PPR Points')
plt.title('Top 10 Predicted PPR Fantasy Points for 2025')

# Add actual 2024 points as a comparison
plt.plot(top_10['Player Name'], top_10['PPR Fantasy Points Scored'], 'ro-', label='2024 Actual Points')
plt.legend()

# Add value labels on the bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 5,
             f'{height:.1f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('../../graphs/top_10_predicted_players.png')
plt.close()

# Analyze feature importance for top predicted players
# Get the weights from the first layer of the neural network
weights = model.coefs_[0]
feature_importance = pd.DataFrame({
    'Feature': X_test.columns,
    'Importance': np.abs(weights).mean(axis=1)
})
feature_importance = feature_importance.sort_values('Importance', ascending=False)

print("\nTop 10 Most Important Features for High Predictions:")
print(feature_importance.head(10))

# Create a visualization of feature importance
plt.figure(figsize=(12, 8))
sns.barplot(data=feature_importance.head(15), x='Importance', y='Feature')
plt.title('Top 15 Most Important Features for PPR Prediction')
plt.tight_layout()
plt.savefig('../../graphs/feature_importance_high_predictions.png')
plt.close() 