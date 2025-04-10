import pandas as pd
import numpy as np

df = pd.read_csv("data/final_data/nfl_stats_long_format.csv") 

df['Yards per Completion'] = df['Passing Yards'] / df['Passing Completions']

df["Attempts per Completion"] = np.where(
    df["Passing Attempts"] > 0,
    df["Passing Completions"] / df["Passing Attempts"],
    np.nan)

df["Yards per Carry"] = np.where(
    df["Rushing Attempts"] > 0,
    df["Rushing Yards"] / df["Rushing Attempts"],
    np.nan)

df["Yards per Reception"] = np.where(
    df["Receptions"] > 0,
    df["Receiving Yards"] / df["Receptions"],
    np.nan)

df['Total Touchdowns Allowed'] = df['Passing Touchdowns Allowed'] + df['Rushing Touchdowns Allowed']

df['Special Teams Impact'] = df['ST_Sacks'] + df['ST_Interceptions'] + df['ST_Fumble Recoveries'] + df['ST_Forced Fumbles']


columns_to_keep = [
    "Player Name", "Age", "Season", "Position", "Team",
    "Games Played","Games Started","Yards per Completion", "Attempts per Completion",
    "Passing Touchdowns", "Interceptions Thrown", "Rushing Attempts",
    "Yards per Carry", "Rushing Touchdowns", "Targets", 'Yards per Reception',
    "Receiving Touchdowns", "Fumbles","Field Goals Made","Extra Points Made",
    "Total Touchdowns Allowed","Special Teams Impact","ST_Safeties",
    "ST_Special Teams Touchdowns","XP2","Average ADP","Positional ADP",
    "PPR Fantasy Points Scored","Standard Fantasy Points Scored"
]

long_df = df[columns_to_keep].copy()

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

long_df.to_csv("data/final_data/nfl_stats_long_format_filtered.csv", index=False)
