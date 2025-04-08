import pandas as pd
import numpy as np

def drop_multicollinearity(path="data/final_data/nfl_merged_wide_format.csv",years=[2022,2023,2024]):

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

        # 2018 Stats
        "2018 Age", "2018 Position", "2018 Team", "2018 Games Played", "2018 Games Started",
        "2018 Total Passing", "2018 Interceptions Thrown",  
        "2018 Rushing Touchdowns", "2018 Carries*Yards",  
        "2018 Receiving Touchdowns", "2018 Receptions*Yards",  
        "2018 Fumbles", "2018 Field Goals Made", "2018 Extra Points Made",  
        "2018 ST_Interceptions", "2018 ST_Safeties", "2018 ST_Special Teams Touchdowns", "2018 XP2",  
        "2018 Touchdowns Allowed", "2018 Average ADP", "2018 Positional ADP",   
        "2018 PPR Fantasy Points Scored", "2018 Standard Fantasy Points Scored",  

        # 2019 Stats
        "2019 Age", "2019 Position", "2019 Team", "2019 Games Played", "2019 Games Started",
        "2019 Total Passing", "2019 Interceptions Thrown",  
        "2019 Rushing Touchdowns", "2019 Carries*Yards",  
        "2019 Receiving Touchdowns", "2019 Receptions*Yards",  
        "2019 Fumbles", "2019 Field Goals Made", "2019 Extra Points Made",  
        "2019 ST_Interceptions", "2019 ST_Safeties", "2019 ST_Special Teams Touchdowns", "2019 XP2",  
        "2019 Touchdowns Allowed", "2019 Average ADP", "2019 Positional ADP",   
        "2019 PPR Fantasy Points Scored", "2019 Standard Fantasy Points Scored",  

        # 2020 Stats
        "2020 Age", "2020 Position", "2020 Team", "2020 Games Played", "2020 Games Started",
        "2020 Total Passing", "2020 Interceptions Thrown",  
        "2020 Rushing Touchdowns", "2020 Carries*Yards",  
        "2020 Receiving Touchdowns", "2020 Receptions*Yards",  
        "2020 Fumbles", "2020 Field Goals Made", "2020 Extra Points Made",  
        "2020 ST_Interceptions", "2020 ST_Safeties", "2020 ST_Special Teams Touchdowns", "2020 XP2",  
        "2020 Touchdowns Allowed", "2020 Average ADP", "2020 Positional ADP",   
        "2020 PPR Fantasy Points Scored", "2020 Standard Fantasy Points Scored",  

        # 2021 Stats
        "2021 Age", "2021 Position", "2021 Team", "2021 Games Played", "2021 Games Started",
        "2021 Total Passing", "2021 Interceptions Thrown",  
        "2021 Rushing Touchdowns", "2021 Carries*Yards",  
        "2021 Receiving Touchdowns", "2021 Receptions*Yards",  
        "2021 Fumbles", "2021 Field Goals Made", "2021 Extra Points Made",  
        "2021 ST_Interceptions", "2021 ST_Safeties", "2021 ST_Special Teams Touchdowns", "2021 XP2",  
        "2021 Touchdowns Allowed", "2021 Average ADP", "2021 Positional ADP",   
        "2021 PPR Fantasy Points Scored", "2021 Standard Fantasy Points Scored",  

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

    df.to_csv("data/final_data/nfl_merged_wide_format_no_multicollinearity.csv", index=False)

    return "File saved as nfl_merged_wide_format_no_multicollinearity.csv"

if __name__ == "__main__":
    drop_multicollinearity('data/final_data/nfl_merged_wide_format.csv', [2018, 2019, 2020, 2021, 2022, 2023, 2024])