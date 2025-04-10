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

columns_to_keep = [
    "Age", "Games Played", "Games Started", "Yards per Completion", "Attempts per Completion",
    "Passing Touchdowns", "Interceptions Thrown", "Rushing Attempts",
    "Yards per Carry", "Rushing Touchdowns", "Targets", 'Yards per Reception',
    "Receiving Touchdowns", "Fumbles" , "XP2", "ESPN ADP", "Sleeper ADP", "NFL ADP",
    "RTSports ADP", "Average ADP", "Positional ADP"
]

df_filtered = df[columns_to_keep].copy()

df_filtered.to_csv("data/final_data/nfl_stats_long_format_.csv", index=False)

