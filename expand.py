#!/usr/bin/env python3
"""
Scrapes NFL stats for 2022 from Pro-Football-Reference:
 - Passing
 - Rushing
 - Receiving
 - Kicking
 - Team Defense (opp)

1) Right after scraping each offensive table, removes single-team rows if the player 
   has a multi-team row ("2TM"/"3TM"/"4TM").
2) Transforms each to a 35-col schema but keeps missing stats as NaN.
3) Merges all offensive DataFrames by Player Name, coalescing NaN columns.
4) Removes single-team rows if a multi-team row is present (final safety check).
5) Appends team defense rows and excludes unwanted defense rows.
6) Sets final_df["Season"] = year so every row has the correct season.
"""

import requests
import pandas as pd
from bs4 import BeautifulSoup
from io import StringIO
import time
import numpy as np
from collections import defaultdict

##############################
# 0) GLOBAL SETTINGS
##############################

FINAL_COLUMNS = [
    "Season", "Player ID", "Player Name", "Position", "Team",
    "Games Played", "Games Started",
    "Passing Attempts", "Passing Completions", "Passing Yards", "Passing Touchdowns", "Interceptions Thrown",
    "Rushing Attempts", "Rushing Yards", "Rushing Touchdowns",
    "Targets", "Receptions", "Receiving Yards", "Receiving Touchdowns",
    "Fumbles", "Fumbles Lost", "Two Point Conversions",
    "Field Goals Made", "Field Goals Attempted", "Extra Points Made", "Extra Points Attempted", 
    "Total Yards Allowed", "Total Plays", "Takeaways", "Def Fumbles Lost", "First Downs Allowed", 
    "Passing Yards Allowed", "Passing Touchdowns Allowed", "Rushing Yards Allowed", "Rushing Touchdowns Allowed",
    "Penalties Committed", "Penalty Yards", 
    "First Downs by Penalty", "Percent Drives Scored On", "Percent Drives Takeaway"
]

##############################
# 1) HELPER FUNCTIONS
##############################

def _clean_header_rows(df):
    """Remove repeated header rows labeled 'Player' or 'Rk' within the DataFrame body."""
    if "Player" in df.columns:
        df = df[df["Player"] != "Player"]
    if "Rk" in df.columns:
        df = df[df["Rk"] != "Rk"]
    return df

def _drop_multiindex(df):
    """Drop one level if df.columns is a MultiIndex (PFR often has multi-level table headers)."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(0)
    return df

def drop_single_teams_if_multi_team_exists(df, player_col="Player", team_col="Tm"):
    """
    If a player has a multi-team row (Team in [2TM, 3TM, 4TM]),
    remove any single-team rows for that same player.
    This ensures we keep only the combined row for multi-team players.
    """
    if player_col not in df.columns or team_col not in df.columns:
        return df  # can't do anything if these columns don't exist

    def filter_group(g):
        multi_mask = g[team_col].isin(["2TM", "3TM", "4TM"])
        if multi_mask.any():
            return g[multi_mask]
        else:
            return g

    # Updated to exclude grouping columns in the operation
    return df.groupby(player_col, group_keys=False).apply(filter_group).reset_index(drop=True)

##############################
# 2) SCRAPING FUNCTIONS
##############################

def get_passing_stats(year=2022):
    url = f"https://www.pro-football-reference.com/years/{year}/passing.htm"
    print(f"[Scrape] Passing stats: {url}")
    resp = requests.get(url)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    table = soup.find("table", {"id": "passing"})
    if table is None:
        raise ValueError("Could not find passing table on PFR.")

    df_list = pd.read_html(StringIO(str(table)))
    if not df_list:
        raise ValueError("Could not parse passing table with pandas.")

    df = df_list[0]
    df = _clean_header_rows(df)
    df = _drop_multiindex(df)

    # Remove single-team rows if a multi-team row (2TM/3TM/4TM) exists for that player
    df = drop_single_teams_if_multi_team_exists(df, player_col="Player", team_col="Tm")

    return df

def get_rushing_stats(year=2022):
    url = f"https://www.pro-football-reference.com/years/{year}/rushing.htm"
    print(f"[Scrape] Rushing stats: {url}")
    resp = requests.get(url)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    table = soup.find("table", {"id": "rushing"})
    if table is None:
        raise ValueError("Could not find rushing table on PFR.")

    df_list = pd.read_html(StringIO(str(table)))
    if not df_list:
        raise ValueError("Could not parse rushing table with pandas.")

    df = df_list[0]
    df = _clean_header_rows(df)
    df = _drop_multiindex(df)

    # Remove single-team rows if multi-team row exists
    df = drop_single_teams_if_multi_team_exists(df, player_col="Player", team_col="Tm")

    return df

def get_receiving_stats(year=2022):
    url = f"https://www.pro-football-reference.com/years/{year}/receiving.htm"
    print(f"[Scrape] Receiving stats: {url}")
    resp = requests.get(url)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    table = soup.find("table", {"id": "receiving"})
    if table is None:
        raise ValueError("Could not find receiving table on PFR.")

    df_list = pd.read_html(StringIO(str(table)))
    if not df_list:
        raise ValueError("Could not parse receiving table with pandas.")

    df = df_list[0]
    df = _clean_header_rows(df)
    df = _drop_multiindex(df)

    # Remove single-team rows if multi-team row exists
    df = drop_single_teams_if_multi_team_exists(df, player_col="Player", team_col="Tm")

    return df

def get_kicking_stats(year=2022):
    url = f"https://www.pro-football-reference.com/years/{year}/kicking.htm"
    print(f"[Scrape] Kicking stats: {url}")
    resp = requests.get(url)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    table = soup.find("table", {"id": "kicking"})
    if table is None:
        raise ValueError("Could not find kicking table on PFR.")

    df_list = pd.read_html(StringIO(str(table)))
    if not df_list:
        raise ValueError("Could not parse kicking table with pandas.")

    df = df_list[0]
    df = _clean_header_rows(df)
    df = _drop_multiindex(df)

    # -------------------------------------
    # The key lines to keep last duplicates
    df = df.iloc[:, ::-1]  # Reverse columns
    df = df.loc[:, ~df.columns.duplicated(keep='first')]
    df = df.iloc[:, ::-1]  # Reverse back
    # -------------------------------------

    # Remove single-team rows if multi-team row exists
    df = drop_single_teams_if_multi_team_exists(df, player_col="Player", team_col="Tm")

    return df

def get_team_defense_stats(year=2022):
    url = f"https://www.pro-football-reference.com/years/{year}/opp.htm"
    print(f"[Scrape] Team defense stats (opp): {url}")
    resp = requests.get(url)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    table = soup.find("table", {"id": "team_stats"})
    if table is None:
        raise ValueError("Could not find team defense table on PFR (opp.htm).")

    df_list = pd.read_html(StringIO(str(table)))
    if not df_list:
        raise ValueError("Could not parse team defense table with pandas.")

    df = df_list[0]
    # Clean repeated header rows if they exist
    if "Tm" in df.columns:
        df = df[df["Tm"] != "Tm"]
    if "Rk" in df.columns:
        df = df[df["Rk"] != "Rk"]

    df = _drop_multiindex(df)

    # Remove "League Total" or any total rows
    if "Tm" in df.columns:
        df = df[~df["Tm"].str.contains("League Total", na=False)]

    if "G" in df.columns:
        df.loc[:, "G"] = pd.to_numeric(df["G"], errors="coerce")  # Use .loc to avoid SettingWithCopyWarning

    df.rename(columns={"Tm": "Team"}, inplace=True)

    # **Rename duplicate '1stD', 'Yds', and 'TD' columns with suffixes**
    columns = df.columns.tolist()
    new_columns = []
    count = defaultdict(int)

    for col in columns:
        if col in ["1stD", "Yds", "TD"]:
            count[col] += 1
            new_col = f"{col}_{count[col]}"
            new_columns.append(new_col)
        else:
            new_columns.append(col)

    df.columns = new_columns

    # **Optional:** Print columns to verify their new names
    print("Team Defense Columns:", df.columns.tolist())

    return df.reset_index(drop=True)

##############################
# 3) TRANSFORM FUNCTIONS (KEEP NaN)
##############################

def transform_passing(df_passing, year=2022):
    out = pd.DataFrame(columns=FINAL_COLUMNS)
    out["Season"] = year
    out["Player ID"] = None
    out["Player Name"] = df_passing["Player"]
    out["Position"] = df_passing.get("Pos", pd.Series(dtype=object))
    out["Team"] = df_passing.get("Team", pd.Series(dtype=object))

    # Convert to numeric if they exist
    # We'll keep them as NaN if missing
    out["Games Played"] = pd.to_numeric(df_passing.get("G", pd.Series(dtype=float)), errors="coerce")
    out["Games Started"] = pd.to_numeric(df_passing.get("GS", pd.Series(dtype=float)), errors="coerce")

    out["Passing Attempts"] = pd.to_numeric(df_passing.get("Att", pd.Series(dtype=float)), errors="coerce")
    out["Passing Completions"] = pd.to_numeric(df_passing.get("Cmp", pd.Series(dtype=float)), errors="coerce")
    out["Passing Yards"] = pd.to_numeric(df_passing.get("Yds", pd.Series(dtype=float)), errors="coerce")
    out["Passing Touchdowns"] = pd.to_numeric(df_passing.get("TD", pd.Series(dtype=float)), errors="coerce")
    out["Interceptions Thrown"] = pd.to_numeric(df_passing.get("Int", pd.Series(dtype=float)), errors="coerce")

    # Rushing placeholders: no data from passing table
    out["Rushing Attempts"] = np.nan
    out["Rushing Yards"] = np.nan
    out["Rushing Touchdowns"] = np.nan

    # Receiving placeholders
    out["Targets"] = np.nan
    out["Receptions"] = np.nan
    out["Receiving Yards"] = np.nan
    out["Receiving Touchdowns"] = np.nan

    # Other placeholders
    out["Fumbles"] = np.nan
    out["Fumbles Lost"] = np.nan
    out["Two Point Conversions"] = np.nan
    out["Field Goals Made"] = np.nan
    out["Field Goals Attempted"] = np.nan
    out["Extra Points Made"] = np.nan
    out["Extra Points Attempted"] = np.nan

    # >>> NO CHANGES HERE FOR NEW DEF STATS
    return out

def transform_rushing(df_rushing, year=2022):
    out = pd.DataFrame(columns=FINAL_COLUMNS)
    out["Season"] = year
    out["Player ID"] = None
    out["Player Name"] = df_rushing["Player"]
    out["Position"] = df_rushing.get("Pos", pd.Series(dtype=object))
    out["Team"] = df_rushing.get("Team", pd.Series(dtype=object))

    out["Games Played"] = pd.to_numeric(df_rushing.get("G", pd.Series(dtype=float)), errors="coerce")
    out["Games Started"] = pd.to_numeric(df_rushing.get("GS", pd.Series(dtype=float)), errors="coerce")

    # Passing placeholders
    out["Passing Attempts"] = np.nan
    out["Passing Completions"] = np.nan
    out["Passing Yards"] = np.nan
    out["Passing Touchdowns"] = np.nan
    out["Interceptions Thrown"] = np.nan

    # Rushing stats
    out["Rushing Attempts"] = pd.to_numeric(df_rushing.get("Att", pd.Series(dtype=float)), errors="coerce")
    out["Rushing Yards"] = pd.to_numeric(df_rushing.get("Yds", pd.Series(dtype=float)), errors="coerce")
    out["Rushing Touchdowns"] = pd.to_numeric(df_rushing.get("TD", pd.Series(dtype=float)), errors="coerce")

    # Receiving placeholders
    out["Targets"] = np.nan
    out["Receptions"] = np.nan
    out["Receiving Yards"] = np.nan
    out["Receiving Touchdowns"] = np.nan

    # Other placeholders
    out["Fumbles"] = pd.to_numeric(df_rushing.get("Fmb", pd.Series(dtype=float)), errors="coerce")
    out["Fumbles Lost"] = np.nan
    out["Two Point Conversions"] = np.nan
    out["Field Goals Made"] = np.nan
    out["Field Goals Attempted"] = np.nan
    out["Extra Points Made"] = np.nan
    out["Extra Points Attempted"] = np.nan

    # >>> NO CHANGES HERE FOR NEW DEF STATS
    return out

def transform_receiving(df_receiving, year=2022):
    out = pd.DataFrame(columns=FINAL_COLUMNS)
    out["Season"] = year
    out["Player ID"] = None
    out["Player Name"] = df_receiving["Player"]
    out["Position"] = df_receiving.get("Pos", pd.Series(dtype=object))
    out["Team"] = df_receiving.get("Team", pd.Series(dtype=object))

    out["Games Played"] = pd.to_numeric(df_receiving.get("G", pd.Series(dtype=float)), errors="coerce")
    out["Games Started"] = pd.to_numeric(df_receiving.get("GS", pd.Series(dtype=float)), errors="coerce")

    # Passing placeholders
    out["Passing Attempts"] = np.nan
    out["Passing Completions"] = np.nan
    out["Passing Yards"] = np.nan
    out["Passing Touchdowns"] = np.nan
    out["Interceptions Thrown"] = np.nan

    # Rushing placeholders
    out["Rushing Attempts"] = np.nan
    out["Rushing Yards"] = np.nan
    out["Rushing Touchdowns"] = np.nan

    # Receiving
    out["Targets"] = pd.to_numeric(df_receiving.get("Tgt", pd.Series(dtype=float)), errors="coerce")
    out["Receptions"] = pd.to_numeric(df_receiving.get("Rec", pd.Series(dtype=float)), errors="coerce")
    out["Receiving Yards"] = pd.to_numeric(df_receiving.get("Yds", pd.Series(dtype=float)), errors="coerce")
    out["Receiving Touchdowns"] = pd.to_numeric(df_receiving.get("TD", pd.Series(dtype=float)), errors="coerce")

    # Other placeholders
    out["Fumbles"] = pd.to_numeric(df_receiving.get("Fmb", pd.Series(dtype=float)), errors="coerce")
    out["Fumbles Lost"] = np.nan
    out["Two Point Conversions"] = np.nan
    out["Field Goals Made"] = np.nan
    out["Field Goals Attempted"] = np.nan
    out["Extra Points Made"] = np.nan
    out["Extra Points Attempted"] = np.nan

    # >>> NO CHANGES HERE FOR NEW DEF STATS
    return out

def transform_kicking(df_kick, year=2022):
    out = pd.DataFrame(columns=FINAL_COLUMNS)
    out["Season"] = year
    out["Player ID"] = None
    out["Player Name"] = df_kick["Player"]
    out["Position"] = df_kick.get("Pos", pd.Series(dtype=object))
    out["Team"] = df_kick.get("Tm", pd.Series(dtype=object))

    out["Games Played"] = pd.to_numeric(df_kick.get("G", pd.Series(dtype=float)), errors="coerce")
    out["Games Started"] = pd.to_numeric(df_kick.get("GS", pd.Series(dtype=float)), errors="coerce")

    # Passing placeholders
    out["Passing Attempts"] = np.nan
    out["Passing Completions"] = np.nan
    out["Passing Yards"] = np.nan
    out["Passing Touchdowns"] = np.nan
    out["Interceptions Thrown"] = np.nan

    # Rushing placeholders
    out["Rushing Attempts"] = np.nan
    out["Rushing Yards"] = np.nan
    out["Rushing Touchdowns"] = np.nan

    # Receiving placeholders
    out["Targets"] = np.nan
    out["Receptions"] = np.nan
    out["Receiving Yards"] = np.nan
    out["Receiving Touchdowns"] = np.nan

    out["Fumbles"] = np.nan
    out["Fumbles Lost"] = np.nan
    out["Two Point Conversions"] = np.nan

    out["Field Goals Made"] = pd.to_numeric(df_kick.get("FGM", pd.Series(dtype=float)), errors="coerce")
    out["Field Goals Attempted"] = pd.to_numeric(df_kick.get("FGA", pd.Series(dtype=float)), errors="coerce")
    out["Extra Points Made"] = pd.to_numeric(df_kick.get("XPM", pd.Series(dtype=float)), errors="coerce")
    out["Extra Points Attempted"] = pd.to_numeric(df_kick.get("XPA", pd.Series(dtype=float)), errors="coerce")

    # >>> NO CHANGES HERE FOR NEW DEF STATS
    return out

def transform_team_defense(df_def, year=2022):
    # ------------------------------------------------------
    # 1) Read each row's defensive fields directly by column name
    # 2) Map them to the desired schema
    # 3) Remove unwanted 'Avg Team Defense' rows
    # ------------------------------------------------------
    out_rows = []
    for _, row in df_def.iterrows():
        team_name = row["Team"] if "Team" in row else "Unknown"
        g = row["G"] if "G" in row else 0

        # Extract required columns by their unique names
        total_yards_allowed = row.get("Yds_1", np.nan)       # Total Yards Allowed
        passing_yards_allowed = row.get("Yds_2", np.nan)     # Passing Yards Allowed
        rushing_yards_allowed = row.get("Yds_3", np.nan)     # Rushing Yards Allowed
        penalty_yards = row.get("Yds_4", np.nan)             # Penalty Yards
        first_downs_allowed = row.get("1stD_1", np.nan)      # First Downs Allowed
        passing_touchdowns_allowed = row.get("TD_1", np.nan)    # Passing Touchdowns Allowed
        rushing_touchdowns_allowed = row.get("TD_2", np.nan)    # Rushing Touchdowns Allowed

        data_dict = {
            "Season": year,
            "Player ID": None,
            "Player Name": f"{team_name} Defense",
            "Position": "DEF",
            "Team": team_name,
            "Games Played": g,
            "Games Started": g,
            "Passing Attempts": np.nan,
            "Passing Completions": np.nan,
            "Passing Yards": np.nan,
            "Passing Touchdowns": np.nan,
            "Interceptions Thrown": np.nan,
            "Rushing Attempts": np.nan,
            "Rushing Yards": np.nan,
            "Rushing Touchdowns": np.nan,
            "Targets": np.nan,
            "Receptions": np.nan,
            "Receiving Yards": np.nan,
            "Receiving Touchdowns": np.nan,
            "Fumbles": np.nan,
            "Fumbles Lost": np.nan,            # Offensive placeholder
            "Two Point Conversions": np.nan,
            "Field Goals Made": np.nan,
            "Field Goals Attempted": np.nan,
            "Extra Points Made": np.nan,
            "Extra Points Attempted": np.nan,
            # Updated columns using direct access
            "Total Yards Allowed": total_yards_allowed,
            "Total Plays": row.get("Ply", np.nan),
            "Takeaways": row.get("TO", np.nan),
            "Def Fumbles Lost": row.get("FL", np.nan),  # Ensure 'FL' exists
            "First Downs Allowed": first_downs_allowed,
            "Passing Yards Allowed": passing_yards_allowed,
            "Passing Touchdowns Allowed": passing_touchdowns_allowed,
            "Rushing Yards Allowed": rushing_yards_allowed,
            "Rushing Touchdowns Allowed": rushing_touchdowns_allowed,
            "Penalties Committed": row.get("Pen", np.nan),
            "Penalty Yards": penalty_yards,
            "First Downs by Penalty": row.get("1stPy", np.nan),
            "Percent Drives Scored On": row.get("Sc%", np.nan),
            "Percent Drives Takeaway": row.get("TO%", np.nan),
        }
        out_rows.append(data_dict)

    out_df = pd.DataFrame(out_rows, columns=FINAL_COLUMNS)
    
    # Exclude Average Team Defense rows
    exclude_names = ['Avg Team Defense', 'Avg Tm/G Defense']
    out_df = out_df[~out_df['Player Name'].isin(exclude_names)]

    return out_df

##############################
# 4) MERGING & FILTERING
##############################

def merge_offensive_dataframes_no_agg(dfs):
    """
    Merges multiple DataFrames (in the 35-col schema) by "Player Name" WITHOUT numeric summation.
    If a column is NaN in the main DataFrame but non-NaN in the second (dup), we use the second.
    Otherwise, we keep the main value.
    """
    final = pd.DataFrame(columns=FINAL_COLUMNS)

    for i, df in enumerate(dfs):
        if i == 0:
            final = df.copy()
        else:
            final = pd.merge(
                final,
                df,
                on="Player Name",
                how="outer",
                suffixes=("", "_dup")
            )
            for col in FINAL_COLUMNS:
                col_dup = col + "_dup"
                if col in final.columns and col_dup in final.columns:
                    if pd.api.types.is_numeric_dtype(final[col]):
                        # Use combine_first to prefer non-NaN values from the main column
                        final[col] = final[col].combine_first(final[col_dup])
                        final.drop(columns=[col_dup], inplace=True)
                    else:
                        final[col] = final[col].combine_first(final[col_dup])
                        final.drop(columns=[col_dup], inplace=True)

    return final.reset_index(drop=True)

def keep_combined_multiteam_rows(df):
    """
    Keeps only multi-team rows ("2TM", "3TM", "4TM") if they exist for a player.
    Otherwise, keeps the single-team row.
    """
    def filter_group(g):
        mt_mask = g["Team"].isin(["2TM", "3TM", "4TM"])
        if mt_mask.any():
            return g[mt_mask]
        else:
            return g

    return df.groupby("Player Name", group_keys=False).apply(filter_group).reset_index(drop=True)

def pick_best_multiteam_row(df, stat_col="Rushing Attempts"):
    """
    From multi-team rows, picks the row with the highest value in the specified stat_col.
    """
    def filter_group(g):
        mt_mask = g["Team"].isin(["2TM", "3TM", "4TM"])
        multi_team_rows = g[mt_mask]
        if len(multi_team_rows) <= 1:
            return g
        else:
            # Pick the row with the maximum stat_col value
            best_mt_row = multi_team_rows.loc[multi_team_rows[stat_col].idxmax()]
            # Keep non-multi-team rows
            normal_rows = g[~mt_mask]
            # Combine normal rows with the best multi-team row
            return pd.concat([normal_rows, best_mt_row.to_frame().T], ignore_index=True)

    return df.groupby("Player Name", group_keys=False).apply(filter_group).reset_index(drop=True)

##############################
# 5) CREATE FINAL DATASET
##############################

def create_final_dataset(year=2022):
    """
    1. Scrape passing/rushing/receiving/kicking/team defense
       and remove single-team rows for multi-team players right after scraping.
    2. Transform each to 35-col schema, preserving NaN for missing stats.
    3. Merge all offensive DataFrames by Player Name, coalescing NaN columns.
    4. Remove single-team rows if a multi-team row is present (final safety check),
       then pick the best multi-team row if duplicates remain.
    5. Append defense rows and exclude unwanted defense rows.
    6. Force final_df["Season"] = year so every row has correct year.
    """
    print(f"=== Creating final dataset for year {year} ===\n")

    # 1) Scrape
    df_pass = get_passing_stats(year)
    time.sleep(1)
    df_rush = get_rushing_stats(year)
    time.sleep(1)
    df_recv = get_receiving_stats(year)
    time.sleep(1)
    df_kick = get_kicking_stats(year)
    time.sleep(1)
    df_def_opp = get_team_defense_stats(year)
    time.sleep(1)

    # 2) Transform
    df_pass_final = transform_passing(df_pass, year)
    df_rush_final = transform_rushing(df_rush, year)
    df_recv_final = transform_receiving(df_recv, year)
    df_kick_final = transform_kicking(df_kick, year)
    df_def_final = transform_team_defense(df_def_opp, year)

    # 3) Merge offense (NaN coalescing)
    df_offense = merge_offensive_dataframes_no_agg([df_pass_final, df_rush_final, df_recv_final, df_kick_final])

    # 4) Remove single-team if multi-team row is present, then pick best multi-team row
    df_offense = keep_combined_multiteam_rows(df_offense)
    df_offense = pick_best_multiteam_row(df_offense, stat_col="Rushing Attempts")

    # 5) Append defense
    final_df = pd.concat([df_offense, df_def_final], ignore_index=True)

    # 6) Force Season = year
    final_df["Season"] = year

    return final_df

def main(year=2022, save_csv=True):
    df_final = create_final_dataset(year=year)
    print("\nSample of final data (NaN for missing stats instead of 0):")
    print(df_final.head(30))

    # **Verify 'First Downs Allowed' and other key columns**
    print("\nSample 'First Downs Allowed' Values:")
    print(df_final[['Player Name', 'First Downs Allowed']].head(10))

    print("\nSample 'Total Yards Allowed' Values:")
    print(df_final[['Player Name', 'Total Yards Allowed']].head(10))

    print("\nSample 'Passing Yards Allowed' Values:")
    print(df_final[['Player Name', 'Passing Yards Allowed']].head(10))

    print("\nSample 'Passing Touchdowns Allowed' Values:")
    print(df_final[['Player Name', 'Passing Touchdowns Allowed']].head(10))

    print("\nSample 'Rushing Yards Allowed' Values:")
    print(df_final[['Player Name', 'Rushing Yards Allowed']].head(10))

    print("\nSample 'Rushing Touchdowns Allowed' Values:")
    print(df_final[['Player Name', 'Rushing Touchdowns Allowed']].head(10))

    print("\nSample 'Penalty Yards' Values:")
    print(df_final[['Player Name', 'Penalty Yards']].head(10))

    if save_csv:
        out_file = f"nfl_{year}_final_na.csv"
        df_final.to_csv(out_file, index=False)
        print(f"\nSaved final dataset to {out_file}")

if __name__ == "__main__":
    main(year=2022, save_csv=True)
