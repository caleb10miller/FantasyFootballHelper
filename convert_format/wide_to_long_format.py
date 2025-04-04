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
years = ['2022', '2023', '2024']

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
    
    # Only keep rows where the player has actual stats (using Games Played as indicator)
    subset = subset[~pd.isna(subset['Games Played'])]
    
    dfs.append(subset)

# Combine all dataframes
long_df = pd.concat(dfs, axis=0).reset_index(drop=True)

# Sort by Player Name and Season
long_df = long_df.sort_values(by=['Player Name', 'Season'])

# Initialize target columns
long_df['Target_PPR'] = np.nan
long_df['Target_Standard'] = np.nan

# Create target variables for next season's fantasy points
# We'll do this by player to ensure we're not creating targets for players who didn't play
targets_df = []
for name, group in long_df.groupby('Player Name'):
    # Sort by season to ensure correct order
    group = group.sort_values('Season')
    
    # Only set target if there's a next season's data
    if len(group) > 1:
        for i in range(len(group) - 1):
            current_season = group.iloc[i]
            next_season = group.iloc[i + 1]
            
            # Only set target if next season has actual stats
            if not pd.isna(next_season['PPR Fantasy Points Scored']):
                group.iloc[i, group.columns.get_loc('Target_PPR')] = next_season['PPR Fantasy Points Scored']
                group.iloc[i, group.columns.get_loc('Target_Standard')] = next_season['Standard Fantasy Points Scored']
    
    targets_df.append(group)

# Combine back into single dataframe
long_df = pd.concat(targets_df, axis=0).reset_index(drop=True)

# Reorder columns to put Season and targets in desired positions
cols = ['Player Name', 'Season', 'Target_PPR', 'Target_Standard'] + [col for col in long_df.columns 
      if col not in ['Player Name', 'Season', 'Target_PPR', 'Target_Standard']]
long_df = long_df[cols]

# Save the long format data
long_df.to_csv('data/final_data/nfl_stats_long_format.csv', index=False)

print("Data has been converted to long format and saved to 'nfl_stats_long_format.csv'")
print(f"Shape of long format data: {long_df.shape}")
print("\nColumns in long format:")
print(long_df.columns.tolist())

# Print some example rows to verify the target variables
print("\nExample rows showing current and target fantasy points:")
print(long_df[['Player Name', 'Season', 'PPR Fantasy Points Scored', 'Target_PPR']].head(10)) 