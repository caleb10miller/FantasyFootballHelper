import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load predictions
test_df = pd.read_csv('../../data/final_data/nfl_stats_long_format.csv')
test_df = test_df[test_df['Season'] == 2024]

# Calculate starter status (60% of games started)
test_df['Games_Started_Pct'] = test_df['Games Started'] / test_df['Games Played']
test_df['Is_Starter'] = test_df['Games_Started_Pct'] >= 0.6

# Load model and scaler
model = joblib.load('../joblib_files/mlp_model_2025_ppr.joblib')

# Get feature columns
feature_cols = [col for col in test_df.columns if col not in 
                ['Player Name', 'Season', 'Target_PPR', 'Target_Standard', 
                 'PPR Fantasy Points Scored', 'Standard Fantasy Points Scored',
                 'Games_Started_Pct', 'Is_Starter']]

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
predictions_df['Points_Increase'] = predictions_df['Predicted_2025_PPR'] - predictions_df['PPR Fantasy Points Scored']

# Analyze predictions by starter status
print("\nPrediction Analysis by Starter Status:")
print(f"Number of Starters: {predictions_df['Is_Starter'].sum()}")
print(f"Number of Non-Starters: {(~predictions_df['Is_Starter']).sum()}")

# Top 20 predicted players with starter status
top_20 = predictions_df.sort_values('Predicted_2025_PPR', ascending=False).head(20)
print("\nTop 20 Predicted Players with Starter Status:")
print(top_20[['Player Name', 'Position', 'Team', 'PPR Fantasy Points Scored', 
              'Predicted_2025_PPR', 'Points_Increase', 'Games Started', 
              'Games Played', 'Games_Started_Pct', 'Is_Starter']].to_string(index=False))

# Analyze average predictions by starter status
starter_stats = predictions_df.groupby('Is_Starter').agg({
    'PPR Fantasy Points Scored': 'mean',
    'Predicted_2025_PPR': 'mean',
    'Points_Increase': 'mean',
    'Games Started': 'mean',
    'Games Played': 'mean',
    'Games_Started_Pct': 'mean'
}).round(2)

print("\nAverage Stats by Starter Status:")
print(starter_stats)

# Analyze position distribution by starter status
print("\nPosition Distribution for Starters:")
print(predictions_df[predictions_df['Is_Starter']]['Position'].value_counts())
print("\nPosition Distribution for Non-Starters:")
print(predictions_df[~predictions_df['Is_Starter']]['Position'].value_counts())

# Analyze largest increases by starter status
print("\nTop 10 Largest Increases for Starters:")
starter_increases = predictions_df[predictions_df['Is_Starter']].nlargest(10, 'Points_Increase')
print(starter_increases[['Player Name', 'Position', 'Team', 'PPR Fantasy Points Scored', 
                        'Predicted_2025_PPR', 'Points_Increase', 'Games Started', 
                        'Games Played', 'Games_Started_Pct']].to_string(index=False))

print("\nTop 10 Largest Increases for Non-Starters:")
non_starter_increases = predictions_df[~predictions_df['Is_Starter']].nlargest(10, 'Points_Increase')
print(non_starter_increases[['Player Name', 'Position', 'Team', 'PPR Fantasy Points Scored', 
                            'Predicted_2025_PPR', 'Points_Increase', 'Games Started', 
                            'Games Played', 'Games_Started_Pct']].to_string(index=False))

# Create visualizations
plt.figure(figsize=(12, 6))
sns.boxplot(data=predictions_df, x='Is_Starter', y='Points_Increase')
plt.title('Distribution of Predicted Point Increases by Starter Status')
plt.xlabel('Is Starter')
plt.ylabel('Predicted Point Increase')
plt.savefig('../../graphs/point_increase_by_starter.png')
plt.close()

# Plot average predictions by position and starter status
plt.figure(figsize=(15, 8))
position_starter_avg = predictions_df.groupby(['Position', 'Is_Starter'])['Predicted_2025_PPR'].mean().unstack()
position_starter_avg.plot(kind='bar')
plt.title('Average Predicted 2025 PPR Points by Position and Starter Status')
plt.xlabel('Position')
plt.ylabel('Average Predicted PPR Points')
plt.legend(['Non-Starter', 'Starter'])
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('../../graphs/predictions_by_position_starter.png')
plt.close()

# Analyze the relationship between Games Started percentage and predicted increase
plt.figure(figsize=(10, 6))
plt.scatter(predictions_df['Games_Started_Pct'], predictions_df['Points_Increase'], alpha=0.5)
plt.xlabel('Games Started Percentage')
plt.ylabel('Predicted Point Increase')
plt.title('Relationship Between Games Started Percentage and Predicted Point Increase')
plt.savefig('../../graphs/games_started_vs_increase.png')
plt.close()

# Print correlation between Games Started percentage and predicted increase
correlation = predictions_df['Games_Started_Pct'].corr(predictions_df['Points_Increase'])
print(f"\nCorrelation between Games Started Percentage and Predicted Point Increase: {correlation:.3f}")

# NEW ANALYSIS: Investigate why starters are predicted to decrease more often
print("\n--- ADDITIONAL ANALYSIS: WHY STARTERS MIGHT BE PREDICTED TO DECREASE ---")

# Check if there's a correlation between 2024 points and predicted decrease
correlation_2024_points = predictions_df['PPR Fantasy Points Scored'].corr(predictions_df['Points_Increase'])
print(f"Correlation between 2024 Points and Predicted Point Increase: {correlation_2024_points:.3f}")

# Analyze regression to the mean effect
print("\nRegression to the Mean Analysis:")
print("Top 10 highest scoring players in 2024 and their predicted changes:")
top_scorers_2024 = predictions_df.nlargest(10, 'PPR Fantasy Points Scored')
print(top_scorers_2024[['Player Name', 'Position', 'Team', 'PPR Fantasy Points Scored', 
                        'Predicted_2025_PPR', 'Points_Increase', 'Is_Starter']].to_string(index=False))

# Analyze age/experience effect if available
if 'Age' in predictions_df.columns:
    print("\nAge Analysis:")
    age_correlation = predictions_df['Age'].corr(predictions_df['Points_Increase'])
    print(f"Correlation between Age and Predicted Point Increase: {age_correlation:.3f}")
    
    # Group by age ranges
    predictions_df['Age_Group'] = pd.cut(predictions_df['Age'], bins=[0, 22, 25, 28, 31, 100], 
                                        labels=['22 or younger', '23-25', '26-28', '29-31', '32+'])
    age_group_stats = predictions_df.groupby('Age_Group')['Points_Increase'].mean().round(2)
    print("\nAverage Point Increase by Age Group:")
    print(age_group_stats)
    
    # Visualize age effect
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=predictions_df, x='Age_Group', y='Points_Increase')
    plt.title('Distribution of Predicted Point Increases by Age Group')
    plt.xlabel('Age Group')
    plt.ylabel('Predicted Point Increase')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('../../graphs/point_increase_by_age.png')
    plt.close()

# Analyze position-specific trends
print("\nPosition-Specific Analysis:")
position_correlation = predictions_df.groupby('Position')['Games_Started_Pct'].corr(predictions_df['Points_Increase']).round(3)
print("\nCorrelation between Games Started Percentage and Point Increase by Position:")
print(position_correlation)

# NEW: Analyze Jayden Daniels specifically
print("\n--- JAYDEN DANIELS ANALYSIS ---")
jayden_daniels = predictions_df[predictions_df['Player Name'] == 'Jayden Daniels']
if not jayden_daniels.empty:
    print("\nJayden Daniels Stats:")
    print(jayden_daniels[['Player Name', 'Position', 'Team', 'PPR Fantasy Points Scored', 
                          'Predicted_2025_PPR', 'Points_Increase', 'Games Started', 
                          'Games Played', 'Games_Started_Pct', 'Is_Starter']].to_string(index=False))
    
    # Compare with other young QBs
    young_qbs = predictions_df[(predictions_df['Position'] == 'QB') & 
                              (predictions_df['Age'] <= 25) if 'Age' in predictions_df.columns else 
                              (predictions_df['Position'] == 'QB')]
    
    print("\nComparison with Other Young QBs:")
    print(young_qbs[['Player Name', 'Team', 'PPR Fantasy Points Scored', 
                     'Predicted_2025_PPR', 'Points_Increase', 'Games Started', 
                     'Games Played', 'Games_Started_Pct', 'Is_Starter']].sort_values('PPR Fantasy Points Scored', ascending=False).to_string(index=False))
    
    # Analyze QBs with similar 2024 production
    similar_production_qbs = predictions_df[(predictions_df['Position'] == 'QB') & 
                                           (predictions_df['PPR Fantasy Points Scored'] >= 300) & 
                                           (predictions_df['PPR Fantasy Points Scored'] <= 380)]
    
    print("\nQBs with Similar 2024 Production (300-380 points):")
    print(similar_production_qbs[['Player Name', 'Team', 'PPR Fantasy Points Scored', 
                                 'Predicted_2025_PPR', 'Points_Increase', 'Games Started', 
                                 'Games Played', 'Games_Started_Pct', 'Is_Starter']].sort_values('PPR Fantasy Points Scored', ascending=False).to_string(index=False))
else:
    print("Jayden Daniels not found in the dataset.")

# NEW: Analyze QBs with unusual predictions
print("\n--- QBs WITH UNUSUAL PREDICTIONS ---")
unusual_qbs = predictions_df[(predictions_df['Position'] == 'QB') & 
                            ((predictions_df['Points_Increase'] < -200) | 
                             (predictions_df['Points_Increase'] > 150))]
print("\nQBs with Unusual Predictions (Large Decrease or Increase):")
print(unusual_qbs[['Player Name', 'Team', 'PPR Fantasy Points Scored', 
                   'Predicted_2025_PPR', 'Points_Increase', 'Games Started', 
                   'Games Played', 'Games_Started_Pct', 'Is_Starter']].sort_values('Points_Increase', ascending=False).to_string(index=False))

# NEW: Analyze team changes or situational factors
print("\n--- TEAM CHANGES ANALYSIS ---")
# Check if there's a 'Team_Change' column or similar
if 'Team_Change' in predictions_df.columns:
    team_change_correlation = predictions_df['Team_Change'].corr(predictions_df['Points_Increase'])
    print(f"Correlation between Team Change and Predicted Point Increase: {team_change_correlation:.3f}")
    
    # Group by team change status
    team_change_stats = predictions_df.groupby('Team_Change')['Points_Increase'].mean().round(2)
    print("\nAverage Point Increase by Team Change Status:")
    print(team_change_stats)
else:
    print("Team change data not available for analysis.")

# Analyze players with unusual predictions (high increase for non-starters or low increase for starters)
print("\nPotential Anomalies:")
anomalies = predictions_df[
    ((~predictions_df['Is_Starter']) & (predictions_df['Points_Increase'] > 100)) |
    (predictions_df['Is_Starter'] & (predictions_df['Points_Increase'] < -50))
].sort_values('Points_Increase', ascending=False)

print(anomalies[['Player Name', 'Position', 'Team', 'PPR Fantasy Points Scored', 
                 'Predicted_2025_PPR', 'Points_Increase', 'Games Started', 
                 'Games Played', 'Games_Started_Pct', 'Is_Starter']].to_string(index=False))

# NEW: Analyze the relationship between starter status and draft value
print("\n--- DRAFT VALUE ANALYSIS ---")

# Calculate a simple draft value metric (predicted points / current ADP if available)
# For this example, we'll use a placeholder approach
if 'ADP' in predictions_df.columns:
    predictions_df['Draft_Value'] = predictions_df['Predicted_2025_PPR'] / predictions_df['ADP']
    
    # Compare draft value by starter status
    draft_value_by_starter = predictions_df.groupby('Is_Starter')['Draft_Value'].mean().round(2)
    print("\nAverage Draft Value by Starter Status:")
    print(draft_value_by_starter)
    
    # Visualize draft value distribution
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=predictions_df, x='Is_Starter', y='Draft_Value')
    plt.title('Distribution of Draft Value by Starter Status')
    plt.xlabel('Is Starter')
    plt.ylabel('Draft Value (Predicted Points / ADP)')
    plt.savefig('../../graphs/draft_value_by_starter.png')
    plt.close()
else:
    print("\nADP data not available for draft value analysis.") 