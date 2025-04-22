import pandas as pd
import numpy as np

# Load the data
df = pd.read_csv("data/final_data/nfl_stats_long_format_with_injuries.csv")

def add_rookie_status(df):
    # Sort by Player Name and Season to find first appearance
    df = df.sort_values(['Player Name', 'Season'])
    
    # Group by player and get their first season
    first_seasons = df.groupby('Player Name')['Season'].first()
    
    # Create rookie flag
    df['Is_Rookie'] = df.apply(lambda row: 1 if row['Season'] == first_seasons[row['Player Name']] else 0, axis=1)
    
    return df

df = add_rookie_status(df)

# Sort by player and season to ensure correct order for calculations
df = df.sort_values(['Player Name', 'Season'])

# Initialize new columns for deltas
delta_columns = [
    'Delta_PPR_Fantasy_Points',
    'Delta_Standard_Fantasy_Points',
    'Delta_Passing_Yards',
    'Delta_Rushing_Yards',
    'Delta_Receiving_Yards',
    'Delta_Passing_Touchdowns',
    'Delta_Rushing_Touchdowns',
    'Delta_Receiving_Touchdowns',
    'Delta_Interceptions_Thrown',
    'Delta_Fumbles',
    'Delta_Field_Goals_Made',
    'Delta_Extra_Points_Made',
    'Delta_Average_ADP'
]

# Initialize new columns for rolling averages
rolling_columns = [
    'Rolling_2_Year_PPR_Fantasy_Points',
    'Rolling_2_Year_Standard_Fantasy_Points',
    'Rolling_3_Year_PPR_Fantasy_Points',
    'Rolling_3_Year_Standard_Fantasy_Points'
]

# Initialize all new columns with NaN
for col in delta_columns + rolling_columns:
    df[col] = np.nan

# Calculate deltas and rolling averages for each player
for player in df['Player Name'].unique():
    player_mask = df['Player Name'] == player
    player_data = df[player_mask].copy()
    
    if len(player_data) > 1:
        # Calculate deltas
        df.loc[player_mask, 'Delta_PPR_Fantasy_Points'] = player_data['PPR Fantasy Points Scored'].diff()
        df.loc[player_mask, 'Delta_Standard_Fantasy_Points'] = player_data['Standard Fantasy Points Scored'].diff()
        df.loc[player_mask, 'Delta_Passing_Yards'] = player_data['Passing Yards'].diff()
        df.loc[player_mask, 'Delta_Rushing_Yards'] = player_data['Rushing Yards'].diff()
        df.loc[player_mask, 'Delta_Receiving_Yards'] = player_data['Receiving Yards'].diff()
        df.loc[player_mask, 'Delta_Passing_Touchdowns'] = player_data['Passing Touchdowns'].diff()
        df.loc[player_mask, 'Delta_Rushing_Touchdowns'] = player_data['Rushing Touchdowns'].diff()
        df.loc[player_mask, 'Delta_Receiving_Touchdowns'] = player_data['Receiving Touchdowns'].diff()
        df.loc[player_mask, 'Delta_Interceptions_Thrown'] = player_data['Interceptions Thrown'].diff()
        df.loc[player_mask, 'Delta_Fumbles'] = player_data['Fumbles'].diff()
        df.loc[player_mask, 'Delta_Field_Goals_Made'] = player_data['Field Goals Made'].diff()
        df.loc[player_mask, 'Delta_Extra_Points_Made'] = player_data['Extra Points Made'].diff()
        df.loc[player_mask, 'Delta_Average_ADP'] = player_data['Average ADP'].diff()
        
        # Calculate rolling averages
        df.loc[player_mask, 'Rolling_2_Year_PPR_Fantasy_Points'] = player_data['PPR Fantasy Points Scored'].rolling(window=2, min_periods=1).mean()
        df.loc[player_mask, 'Rolling_2_Year_Standard_Fantasy_Points'] = player_data['Standard Fantasy Points Scored'].rolling(window=2, min_periods=1).mean()
        df.loc[player_mask, 'Rolling_3_Year_PPR_Fantasy_Points'] = player_data['PPR Fantasy Points Scored'].rolling(window=3, min_periods=1).mean()
        df.loc[player_mask, 'Rolling_3_Year_Standard_Fantasy_Points'] = player_data['Standard Fantasy Points Scored'].rolling(window=3, min_periods=1).mean()

# Save to new file
df.to_csv("data/final_data/nfl_stats_long_format_with_context_with_injuries.csv", index=False)

print("Context variables have been added and saved to nfl_stats_long_format_with_context_with_injuries.csv")
print("\nNew columns added:")
print("\nDelta columns:")
for col in delta_columns:
    print(f"- {col}")
print("\nRolling average columns:")
for col in rolling_columns:
    print(f"- {col}") 