import pandas as pd
import numpy as np

def drop_multicollinearity(path="/Users/calebmiller/Library/CloudStorage/OneDrive-Personal/School/MSDS/Q6/DSCI 591/Capstone/data/final_data/nfl_merged_wide_format.csv",years=[2022,2023,2024]):

    df = pd.read_csv(path)

    for year in years:

        df[f"{year} Total Passing"] = df[f"{year} Passing Yards"] * (df[f"{year} Passing Touchdowns"] + 1)
        df[f"{year} Touchdowns Allowed"] = df[f"{year} Passing Touchdowns Allowed"] + df[f"{year} Rushing Touchdowns Allowed"]
        df[f"{year} Receptions*Yards"] = df[f"{year} Receptions"] * (df[f"{year} Receiving Yards"] + 1)
        df[f"{year} Carries*Yards"] = (df[f"{year} Rushing Yards"] + 1) * df[f"{year} Rushing Attempts"]

        df.drop(columns=[
            f"{year} ESPN ADP", 
            f"{year} RTSports ADP", 
            f"{year} Sleeper ADP", 
            f"{year} NFL ADP", 
            f"{year} Rushing Attempts", 
            f"{year} Field Goals Attempted 0-39", 
            f"{year} Field Goals Attempted 40-49", 
            f"{year} Field Goals Made 50+", 
            f"{year} Field Goals Attempted 50+", 
            f"{year} Extra Points Attempted", 
            f"{year} Total Yards Allowed", 
            f"{year} Passing Yards Allowed", 
            f"{year} Rushing Yards Allowed", 
            f"{year} ST_Fumble Recoveries", 
            f"{year} Fantasy Points From Points", 
            f"{year} Percent Drives Takeaway", 
            f"{year} First Downs Allowed", 
            f"{year} Penalties Committed", 
            f"{year} Passing Yards", 
            f"{year} Passing Touchdowns", 
            f"{year} Receptions", 
            f"{year} Passing Attempts", 
            f"{year} Targets", 
            f"{year} Passing Completions", 
            f"{year} Field Goals Made 0-39", 
            f"{year} Field Goals Made 40-49",  
            f"{year} Total Plays", 
            f"{year} Percent Drives Scored On", 
            f"{year} ST_Forced Fumbles", 
            f"{year} Penalty Yards", 
            f"{year} First Downs by Penalty", 
            f"{year} ST_Sacks", 
            f"{year} Passing Touchdowns Allowed", 
            f"{year} Rushing Touchdowns Allowed", 
            f"{year} Takeaways", 
            f"{year} Field Goals Attempted", 
            f"{year} Receiving Yards", 
            f"{year} Rushing Yards"
        ], inplace=True)

        for index, row in df.iterrows():
                if row[f"{year} Total Passing"] == 0 and row[f"{year} Interceptions Thrown"] == 0:
                    df.at[index, f"{year} Interceptions Thrown"] = np.nan
                    df.at[index, f"{year} Total Passing"] = np.nan
                if row[f"{year} Carries*Yards"] == 0 and row[f"{year} Rushing Touchdowns"] == 0:
                    df.at[index, f"{year} Rushing Touchdowns"] = np.nan
                    df.at[index, f"{year} Carries*Yards"] = np.nan
                if row[f"{year} Receptions*Yards"] == 0 and row[f"{year} Receiving Touchdowns"] == 0:
                    df.at[index, f"{year} Receiving Touchdowns"] = np.nan
                    df.at[index, f"{year} Receptions*Yards"] = np.nan

    ordered_columns = [
        "Player Name",  

        # 2022 Stats 
        "2022 Age", "2022 Position", "2022 Team", "2022 Games Played", "2022 Games Started",
        "2022 Total Passing", "2022 Interceptions Thrown",  
        "2022 Rushing Touchdowns", "2022 Carries*Yards",  
        "2022 Receiving Touchdowns", "2022 Receptions*Yards",  
        "2022 Fumbles", "2022 Field Goals Made", "2022 Extra Points Made",  
        "2022 ST_Interceptions", "2022 ST_Safeties", "2022 ST_Special Teams Touchdowns", "2022 XP2",  
        "2022 Touchdowns Allowed", "2022 Average ADP", "2022 Positional ADP",   
        "2022 PPR Fantasy Points Scored", "2022 Standard Fantasy Points Scored",  

        # 2023 Stats
        "2023 Age", "2023 Position", "2023 Team", "2023 Games Played", "2023 Games Started",
        "2023 Total Passing", "2023 Interceptions Thrown",  
        "2023 Rushing Touchdowns", "2023 Carries*Yards",  
        "2023 Receiving Touchdowns", "2023 Receptions*Yards",  
        "2023 Fumbles", "2023 Field Goals Made", "2023 Extra Points Made",  
        "2023 ST_Interceptions", "2023 ST_Safeties", "2023 ST_Special Teams Touchdowns", "2023 XP2",  
        "2023 Touchdowns Allowed", "2023 Average ADP", "2023 Positional ADP",  
        "2023 PPR Fantasy Points Scored", "2023 Standard Fantasy Points Scored",  
          

        # 2024 Stats
        "2024 Age", "2024 Position", "2024 Team", "2024 Games Played", "2024 Games Started",  
        "2024 Total Passing", "2024 Interceptions Thrown",  
        "2024 Rushing Touchdowns", "2024 Carries*Yards",  
        "2024 Receiving Touchdowns", "2024 Receptions*Yards",  
        "2024 Fumbles", "2024 Field Goals Made", "2024 Extra Points Made",  
        "2024 ST_Interceptions", "2024 ST_Safeties", "2024 ST_Special Teams Touchdowns", "2024 XP2",  
        "2024 Touchdowns Allowed", "2024 Average ADP", "2024 Positional ADP",  
        "2024 PPR Fantasy Points Scored", "2024 Standard Fantasy Points Scored",  
    ]

    df = df[ordered_columns]

    df.to_csv("/Users/calebmiller/Library/CloudStorage/OneDrive-Personal/School/MSDS/Q6/DSCI 591/Capstone/data/final_data/nfl_merged_wide_format_no_multicollinearity.csv", index=False)

    return "File saved as nfl_merged_wide_format_no_multicollinearity.csv"

if __name__ == "__main__":
    drop_multicollinearity()