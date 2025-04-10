import pandas as pd
import numpy as np

# Load the wide-format dataset
df = pd.read_csv('data/final_data/nfl_merged_wide_format_no_multicollinearity.csv')

# List of consistent stats that appear in all three years
consistent_stats = [
    'Age', 'Average ADP', 'Carries*Yards', 'Extra Points Made', 
    'Field Goals Made', 'Fumbles', 'Games Played', 'Games Started',
    'Interceptions Thrown', 'PPR Fantasy Points Scored', 'Position',
    'Positional ADP', 'Receiving Touchdowns', 'Receptions*Yards',
    'Rushing Touchdowns', 'ST_Interceptions', 'ST_Safeties',
    'ST_Special Teams Touchdowns', 'Standard Fantasy Points Scored',
    'Team', 'Total Passing', 'Touchdowns Allowed', 'XP2'
]

# Years to process
years = ['2018', '2019', '2020', '2021', '2022', '2023', '2024']

# Melt each stat column per year into long format
dfs = []
for year in years:
    # Create list of columns for this year
    stat_cols = [f"{year} {stat}" for stat in consistent_stats]
    
    # Select only the columns that exist in the dataset
    existing_cols = [col for col in stat_cols if col in df.columns]
    
    # Create subset with Player Name and existing stat columns
    subset = df[['Player Name'] + existing_cols].copy()
    
    # Rename columns to remove year prefix
    rename_dict = {f"{year} {stat}": stat for stat in consistent_stats if f"{year} {stat}" in existing_cols}
    subset = subset.rename(columns=rename_dict)
    
    # Add year column
    subset['Season'] = int(year)
    
    dfs.append(subset)

# Combine all dataframes
long_df = pd.concat(dfs, axis=0).reset_index(drop=True)

# Sort by Player Name and Season
long_df = long_df.sort_values(by=['Player Name', 'Season'])

# Identify players who did not play in each season
# We'll use a few key stats that would indicate the player played
key_stats = ['Games Played', 'Total Passing', 'Carries*Yards', 'Receptions*Yards', 
             'Field Goals Made', 'Extra Points Made', 'Touchdowns Allowed']

# Create a mask for rows where the player did not play
did_not_play_mask = pd.Series(True, index=long_df.index)

# For each key stat, check if it has a non-null value
for stat in key_stats:
    if stat in long_df.columns:
        # If any stat has a non-null value, the player played
        did_not_play_mask = did_not_play_mask & pd.isna(long_df[stat])

# Remove rows where the player did not play
long_df = long_df[~did_not_play_mask].reset_index(drop=True)

# Initialize target columns
long_df['Target_PPR'] = np.nan
long_df['Target_Standard'] = np.nan

# Create target variables for next season's fantasy points
targets_df = []
for name, group in long_df.groupby('Player Name'):
    # Sort by season to ensure correct order
    group = group.sort_values('Season')
    
    # Only set target if there's a next season's data
    if len(group) > 1:
        for i in range(len(group) - 1):
            current_season = group.iloc[i]
            next_season = group.iloc[i + 1]
            
            # Set target for next season's fantasy points
            group.iloc[i, group.columns.get_loc('Target_PPR')] = next_season['PPR Fantasy Points Scored']
            group.iloc[i, group.columns.get_loc('Target_Standard')] = next_season['Standard Fantasy Points Scored']
    
    targets_df.append(group)

# Combine back into single dataframe
long_df = pd.concat(targets_df, axis=0).reset_index(drop=True)

# Drop rows with null target values for all seasons except 2024
initial_rows = len(long_df)
long_df = long_df[
    (long_df['Season'] == 2024) |  # Keep all 2024 rows
    ((long_df['Target_PPR'].notna()) & (long_df['Target_Standard'].notna()))  # For other years, require non-null targets
].reset_index(drop=True)

# Print information about dropped rows
dropped_rows = initial_rows - len(long_df)
print(f"\nDropped {dropped_rows} rows with null target values for seasons before 2024")
print(f"Remaining rows: {len(long_df)}")

# Reorder columns to put Season and targets in desired positions
cols = ['Player Name', 'Season', 'Target_PPR', 'Target_Standard'] + [col for col in long_df.columns 
      if col not in ['Player Name', 'Season', 'Target_PPR', 'Target_Standard', 'PPR Fantasy Points Scored', 'Standard Fantasy Points Scored']]
long_df = long_df[cols]

# Save the long format data
long_df.to_csv('data/final_data/nfl_stats_long_format.csv', index=False)

print("Data has been converted to long format and saved to 'nfl_stats_long_format.csv'")
print(f"Shape of long format data: {long_df.shape}")
print("\nColumns in long format:")
print(long_df.columns.tolist())

# Print some example rows to verify the target variables
print("\nExample rows showing current and target fantasy points:")
print(long_df[['Player Name', 'Season', 'Target_PPR', 'Target_Standard']].head(10))

# Print count of players per season
print("\nCount of players per season:")
print(long_df.groupby('Season').size()) 