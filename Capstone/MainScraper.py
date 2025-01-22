#!/usr/bin/env python3
"""
Scrapes NFL stats for 2022 from Pro-Football-Reference and FantasyPros:
 - Passing
 - Rushing
 - Receiving
 - Kicking
 - Team Defense (opp)
 - Special Teams Stats from FantasyPros

1) Right after scraping each offensive table, removes single-team rows if the player 
   has a multi-team row ("2TM"/"3TM"/"4TM").
2) Transforms each to a 48-col schema but keeps missing stats as NaN.
3) Merges all offensive DataFrames by Player Name, coalescing NaN columns.
4) Removes single-team rows if a multi-team row is present (final safety check).
5) Appends team defense rows and excludes unwanted defense rows.
6) Appends special teams stats without creating unwanted '_st' columns.
7) Sets final_df["Season"] = year so every row has the correct season.
"""

import requests
import pandas as pd
from bs4 import BeautifulSoup
from io import StringIO
import time
import numpy as np
from collections import defaultdict
import re
import os

##############################
# 0) GLOBAL SETTINGS
##############################

FINAL_COLUMNS = [
    "Age", "Season", "Player ID", "Player Name", "Position", "Team",
    "Games Played", "Games Started",
    "Passing Attempts", "Passing Completions", "Passing Yards", "Passing Touchdowns", "Interceptions Thrown",
    "Rushing Attempts", "Rushing Yards", "Rushing Touchdowns",
    "Targets", "Receptions", "Receiving Yards", "Receiving Touchdowns",
    "Fumbles", "Fumbles Lost", "Two Point Conversions",
    # Field Goal Attempts and Makes by Distance
    "Field Goals Attempted 0-19", "Field Goals Made 0-19",
    "Field Goals Attempted 20-29", "Field Goals Made 20-29",
    "Field Goals Attempted 30-39", "Field Goals Made 30-39",
    "Field Goals Attempted 40-49", "Field Goals Made 40-49",
    "Field Goals Attempted 50+", "Field Goals Made 50+",
    # Total Field Goals
    "Field Goals Attempted", "Field Goals Made",
    "Extra Points Made", "Extra Points Attempted",
    "Total Yards Allowed", "Total Plays", "Takeaways", "First Downs Allowed",
    "Passing Yards Allowed", "Passing Touchdowns Allowed", "Rushing Yards Allowed", "Rushing Touchdowns Allowed",
    "Penalties Committed", "Penalty Yards",
    "First Downs by Penalty", "Percent Drives Scored On", "Percent Drives Takeaway",
    "ST_Sacks", "ST_Interceptions", "ST_Fumble Recoveries", "ST_Forced Fumbles",
    "ST_Safeties", "ST_Special Teams Touchdowns"
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
    
    **Modified to keep only the first single-team row if multiple exist.**
    """
    if player_col not in df.columns or team_col not in df.columns:
        return df  # can't do anything if these columns don't exist

    # Group by player
    grouped = df.groupby(player_col)

    # Function to select rows
    def select_rows(g):
        mt_mask = g[team_col].isin(["2TM", "3TM", "4TM"])
        if mt_mask.any():
            return g[mt_mask]
        else:
            return g.iloc[[0]]  # Keep only the first single-team row

    return grouped.apply(select_rows).reset_index(drop=True)

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

    # Handle duplicate FGA and FGM columns by renaming them to match distances
    distance_ranges = ["0-19", "20-29", "30-39", "40-49", "50+", "Total"]
    fga_count = 0
    fgm_count = 0

    new_columns = []
    for col in df.columns:
        if col == "FGA":
            new_columns.append(f"FGA {distance_ranges[fga_count]}")
            fga_count += 1
        elif col == "FGM":
            new_columns.append(f"FGM {distance_ranges[fgm_count]}")
            fgm_count += 1
        else:
            new_columns.append(col)

    # Apply the renamed columns to the DataFrame
    df.columns = new_columns

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
        df = df.copy()  # To avoid SettingWithCopyWarning
        df["G"] = pd.to_numeric(df["G"], errors="coerce")  # Use .loc to avoid SettingWithCopyWarning

    df = df.rename(columns={"Tm": "Team"})

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

    return df.reset_index(drop=True)

def get_special_teams_stats(year=2022):
    url = f"https://www.fantasypros.com/nfl/stats/dst.php?year={year}"
    print(f"[Scrape] Special Teams stats: {url}")
    resp = requests.get(url)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    table = soup.find("table", {"id": "data"})
    if table is None:
        raise ValueError("Could not find special teams table on FantasyPros.")

    df_list = pd.read_html(StringIO(str(table)))
    if not df_list:
        raise ValueError("Could not parse special teams table with pandas.")

    df = df_list[0]
    df = _clean_header_rows(df)

    # Rename columns to match desired stats based on actual headers
    df.rename(columns={
        "Player": "Player Name",
        "Sack": "ST_Sacks",
        "Int": "ST_Interceptions",
        "FR": "ST_Fumble Recoveries",
        "FF": "ST_Forced Fumbles",
        "Def TD": "ST_Defensive Touchdowns",
        "SFTY": "ST_Safeties",
        "SPC TD": "ST_Special Teams Touchdowns"
    }, inplace=True)

    # If the 'Team' column exists, retain it; otherwise, set it as NaN
    if "Team" in df.columns:
        df = df.rename(columns={"Team": "Team"})  # Ensure consistency
    else:
        df["Team"] = np.nan  # Placeholder if 'Team' is not available

    return df

def get_adp_stats(year=2022):
    url = f"https://www.fantasypros.com/nfl/adp/ppr-overall.php?year={year}"
    print(f"[Scrape] Average Draft Position stats: {url}")
    resp = requests.get(url)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    table = soup.find("table", {"id": "data"})

    if table is None:
        raise ValueError("Could not find special teams table on FantasyPros.")

    df_list = pd.read_html(StringIO(str(table)))

    if not df_list:
        raise ValueError("Could not parse special teams table with pandas.")

    df = df_list[0]
    df = _clean_header_rows(df)
    

    return df

def get_and_transform_two_point_conversion_stats(year):
    """
    Fetches NFL 2-point conversion stats for a given year from StatMuse.

    Args:
        year (int): The season year to fetch data for.

    Returns:
        pd.DataFrame: A DataFrame with columns ['Player Name', 'XP2'].
    """
    url = f"https://www.statmuse.com/nfl/ask/most-2-point-conversion-leaders-in-the-nfl-{year}"
    print(f"[Scrape] 2-Point Conversion stats: {url}")

    response = requests.get(url)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Locate the table on the page
    table = soup.find('table')
    if table is None:
        raise ValueError("Could not find the 2-point conversion table on StatMuse.")
    
    # Read the table into a DataFrame
    df_list = pd.read_html(str(table))
    if not df_list:
        raise ValueError("Could not parse the 2-point conversion table with pandas.")
    
    df = df_list[0]

    print(df.columns)

    # Ensure columns 'NAME' and 'XP2' are present
    if 'NAME' not in df.columns or 'XP2' not in df.columns:
        raise ValueError("Required columns 'NAME' or 'XP2' are missing from the data.")

    # Clean up 'Player Name' by removing duplicate name formats
    def clean_player_name(name):
        name = str(name).strip()
        # Regex to detect duplicated format like 'Aaron Jones Sr. A. Jones Sr.'
        pattern = r"^([\w'\.\-\s]+)\s([A-Z]\.[\s\w'\.\-]+)$"
        match = re.match(pattern, name)
        if match:
            full_name, abbrev_name = match.groups()
            # Check if initials in abbrev match the full name
            full_last_name = full_name.split()[-1]
            abbrev_last_name = abbrev_name.split()[-1]
            if full_last_name == abbrev_last_name:
                return full_name  # Return only the full name
        return name

    # Rename 'NAME' to 'Player Name' and clean it
    df.rename(columns={'NAME': 'Player Name'}, inplace=True)
    df['Player Name'] = df['Player Name'].apply(clean_player_name)

    return df[['Player Name', 'XP2']]


##############################
# 3) TRANSFORM FUNCTIONS (KEEP NaN)
##############################

def transform_passing(df_passing, year=2022):
    out = pd.DataFrame(columns=FINAL_COLUMNS)
    out["Age"] = pd.to_numeric(df_passing.get("Age", pd.Series(dtype=float)), errors="coerce")
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
    out["Field Goals Made"] = np.nan
    out["Field Goals Attempted"] = np.nan
    out["Extra Points Made"] = np.nan
    out["Extra Points Attempted"] = np.nan
    out["ST_Sacks"] = np.nan
    out["ST_Interceptions"] = np.nan
    out["ST_Fumble Recoveries"] = np.nan
    out["ST_Forced Fumbles"] = np.nan
    out["ST_Safeties"] = np.nan
    out["ST_Special Teams Touchdowns"] = np.nan
    # "ST_Defensive Touchdowns" removed
    # "Defensive Touchdowns Allowed" removed

    return out

def transform_rushing(df_rushing, year=2022):
    out = pd.DataFrame(columns=FINAL_COLUMNS)
    out["Age"] = pd.to_numeric(df_rushing.get("Age", pd.Series(dtype=float)), errors="coerce")
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
    out["Field Goals Made"] = np.nan
    out["Field Goals Attempted"] = np.nan
    out["Extra Points Made"] = np.nan
    out["Extra Points Attempted"] = np.nan
    out["ST_Sacks"] = np.nan
    out["ST_Interceptions"] = np.nan
    out["ST_Fumble Recoveries"] = np.nan
    out["ST_Forced Fumbles"] = np.nan
    out["ST_Safeties"] = np.nan
    out["ST_Special Teams Touchdowns"] = np.nan
    # "ST_Defensive Touchdowns" removed
    # "Defensive Touchdowns Allowed" removed

    return out

def transform_receiving(df_receiving, year=2022):
    out = pd.DataFrame(columns=FINAL_COLUMNS)
    out["Age"] = pd.to_numeric(df_receiving.get("Age", pd.Series(dtype=float)), errors="coerce")
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
    out["Field Goals Made"] = np.nan
    out["Field Goals Attempted"] = np.nan
    out["Extra Points Made"] = np.nan
    out["Extra Points Attempted"] = np.nan
    out["ST_Sacks"] = np.nan
    out["ST_Interceptions"] = np.nan
    out["ST_Fumble Recoveries"] = np.nan
    out["ST_Forced Fumbles"] = np.nan
    out["ST_Safeties"] = np.nan
    out["ST_Special Teams Touchdowns"] = np.nan
    # "ST_Defensive Touchdowns" removed
    # "Defensive Touchdowns Allowed" removed

    return out

def transform_kicking(df_kick, year=2022):
    """
    Transforms the raw kicking stats DataFrame into the standardized FINAL_COLUMNS format,
    capturing all instances of 'FGA' and 'FGM' by distance.
    """
    out = pd.DataFrame(columns=FINAL_COLUMNS)

    # Basic player information
    out["Age"] = pd.to_numeric(df_kick.get("Age", pd.Series(dtype=float)), errors="coerce")
    out["Season"] = year
    out["Player ID"] = None
    out["Player Name"] = df_kick["Player"]
    out["Position"] = df_kick.get("Pos", pd.Series(dtype=object))
    out["Team"] = df_kick.get("Tm", pd.Series(dtype=object))

    # Games played and started
    out["Games Played"] = pd.to_numeric(df_kick.get("G", pd.Series(dtype=float)), errors="coerce")
    out["Games Started"] = pd.to_numeric(df_kick.get("GS", pd.Series(dtype=float)), errors="coerce")

    # Map FGA and FGM by distance
    distances = ["0-19", "20-29", "30-39", "40-49", "50+"]
    for distance in distances:
        out[f"Field Goals Attempted {distance}"] = pd.to_numeric(df_kick.get(f"FGA {distance}", pd.Series(dtype=float)), errors="coerce").fillna(0)
        out[f"Field Goals Made {distance}"] = pd.to_numeric(df_kick.get(f"FGM {distance}", pd.Series(dtype=float)), errors="coerce").fillna(0)

    # Map Total FGA and FGM
    out["Field Goals Attempted"] = pd.to_numeric(df_kick.get("FGA Total", pd.Series(dtype=float)), errors="coerce").fillna(0)
    out["Field Goals Made"] = pd.to_numeric(df_kick.get("FGM Total", pd.Series(dtype=float)), errors="coerce").fillna(0)

    # Extra Points
    out["Extra Points Made"] = pd.to_numeric(df_kick.get("XPM", pd.Series(dtype=float)), errors="coerce").fillna(0)
    out["Extra Points Attempted"] = pd.to_numeric(df_kick.get("XPA", pd.Series(dtype=float)), errors="coerce").fillna(0)

    # Placeholder stats for non-kicker metrics
    placeholder_columns = [
        "Passing Attempts", "Passing Completions", "Passing Yards", "Passing Touchdowns", "Interceptions Thrown",
        "Rushing Attempts", "Rushing Yards", "Rushing Touchdowns",
        "Targets", "Receptions", "Receiving Yards", "Receiving Touchdowns",
        "Fumbles", "Fumbles Lost", "Two Point Conversions",
        "Total Yards Allowed", "Total Plays", "Takeaways", "First Downs Allowed",
        "Passing Yards Allowed", "Passing Touchdowns Allowed", "Rushing Yards Allowed", "Rushing Touchdowns Allowed",
        "Penalties Committed", "Penalty Yards",
        "First Downs by Penalty", "Percent Drives Scored On", "Percent Drives Takeaway",
        "ST_Sacks", "ST_Interceptions", "ST_Fumble Recoveries", "ST_Forced Fumbles",
        "ST_Safeties", "ST_Special Teams Touchdowns"
    ]
    for col in placeholder_columns:
        out[col] = np.nan

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

        TEAM_NAME_TO_ABBR = {
        'San Francisco 49ers': 'SFO',
        'Buffalo Bills': 'BUF',
        'Baltimore Ravens': 'BAL',
        'New York Jets': 'NYJ',
        'Cincinnati Bengals': 'CIN',
        'Dallas Cowboys': 'DAL',
        'Washington Commanders': 'WAS',
        'Philadelphia Eagles': 'PHI',
        'New Orleans Saints': 'NOR',
        'Pittsburgh Steelers': 'PIT',
        'New England Patriots': 'NWE',
        'Jacksonville Jaguars': 'JAX',
        'Tampa Bay Buccaneers': 'TAM',
        'Denver Broncos': 'DEN',
        'Tennessee Titans': 'TEN',
        'Kansas City Chiefs': 'KAN',
        'Green Bay Packers': 'GNB',
        'New York Giants': 'NYG',
        'Carolina Panthers': 'CAR',
        'Cleveland Browns': 'CLE',
        'Los Angeles Chargers': 'LAC',
        'Los Angeles Rams': 'LAR',
        'Atlanta Falcons': 'ATL',
        'Miami Dolphins': 'MIA',
        'Seattle Seahawks': 'SEA',
        'Las Vegas Raiders': 'LVR',
        'Houston Texans': 'HOU',
        'Minnesota Vikings': 'MIN',
        'Detroit Lions': 'DET',
        'Indianapolis Colts': 'IND',
        'Arizona Cardinals': 'ARI',
        'Chicago Bears': 'CHI'
    }

        team_abbr = TEAM_NAME_TO_ABBR.get(team_name, None)

        g = row["G"] if "G" in row else 0

        # Extract required columns by their unique names
        total_yards_allowed = row.get("Yds_1", np.nan)       # Total Yards Allowed
        passing_yards_allowed = row.get("Yds_2", np.nan)     # Passing Yards Allowed
        rushing_yards_allowed = row.get("Yds_3", np.nan)     # Rushing Yards Allowed
        penalty_yards = row.get("Yds_4", np.nan)             # Penalty Yards
        first_downs_allowed = row.get("1stD_1", np.nan)      # First Downs Allowed
        passing_touchdowns_allowed = row.get("TD_1", np.nan)    # Passing Touchdowns Allowed
        rushing_touchdowns_allowed = row.get("TD_2", np.nan)    # Rushing Touchdowns Allowed

        # Calculate Defensive Touchdowns Allowed
        td1 = row.get("TD_1", 0)
        td2 = row.get("TD_2", 0)
        if pd.isna(td1) and pd.isna(td2):
            defensive_td_allowed = np.nan
        else:
            defensive_td_allowed = (td1 if not pd.isna(td1) else 0) + (td2 if not pd.isna(td2) else 0)

        data_dict = {
            "Season": year,
            "Player ID": None,
            "Player Name": f"{team_name} DST",
            "Position": "DEF",
            "Team": team_abbr,
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
            "Field Goals Made": np.nan,
            "Field Goals Attempted": np.nan,
            "Extra Points Made": np.nan,
            "Extra Points Attempted": np.nan,
            "Total Yards Allowed": total_yards_allowed,
            "Total Plays": row.get("Ply", np.nan),
            "Takeaways": row.get("TO", np.nan),
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
            "ST_Sacks": np.nan,
            "ST_Interceptions": np.nan,
            "ST_Fumble Recoveries": np.nan,
            "ST_Forced Fumbles": np.nan,
            "ST_Safeties": np.nan,
            "ST_Special Teams Touchdowns": np.nan,
            # "ST_Defensive Touchdowns" removed
            # "Defensive Touchdowns Allowed" removed
        }
        out_rows.append(data_dict)

    out_df = pd.DataFrame(out_rows, columns=FINAL_COLUMNS)
    
    # Exclude Average Team Defense rows
    exclude_names = ['Avg Team DST', 'Avg Tm/G DST']
    out_df = out_df[~out_df['Player Name'].isin(exclude_names)]

    return out_df

def transform_special_teams(df_st, year=2022):
    def normalize_player_name(fp_name):
        # Remove the abbreviation in parentheses, e.g., "New England Patriots (NE)" to "New England Patriots"
        if "(" in fp_name:
            team_name = fp_name.split(" (")[0]
        else:
            team_name = fp_name
        # Append " Defense"
        return f"{team_name} Defense"

    # Create a DataFrame with only 'Player Name' and special teams stats
    out = pd.DataFrame()
    out["Player Name"] = df_st["Player Name"].apply(normalize_player_name).str.replace("Defense", "DST")

    # Assign special teams stats with unique column names
    out["ST_Sacks"] = pd.to_numeric(df_st.get("SACK", pd.Series(dtype=float)), errors="coerce")
    out["ST_Interceptions"] = pd.to_numeric(df_st.get("INT", pd.Series(dtype=float)), errors="coerce")
    out["ST_Fumble Recoveries"] = pd.to_numeric(df_st.get("ST_Fumble Recoveries", pd.Series(dtype=float)), errors="coerce")
    out["ST_Forced Fumbles"] = pd.to_numeric(df_st.get("ST_Forced Fumbles", pd.Series(dtype=float)), errors="coerce")
    out["ST_Safeties"] = pd.to_numeric(df_st.get("ST_Safeties", pd.Series(dtype=float)), errors="coerce")
    out["ST_Special Teams Touchdowns"] = pd.to_numeric(df_st.get("ST_Special Teams Touchdowns", pd.Series(dtype=float)), errors="coerce")

    return out

def transform_ADP_stats(df_adp, year=2022):

    # Create a DataFrame with only 'Player Name' and special teams stats
    df = pd.DataFrame(df_adp)
    
    # List of NFL team abbreviations
    nfl_teams = [
        'ARI', 'ATL', 'BAL', 'BUF', 'CAR', 'CHI', 'CIN', 'CLE',
        'DAL', 'DEN', 'DET', 'GB', 'HOU', 'IND', 'JAC', 'KC',
        'LV', 'LAC', 'LAR', 'MIA', 'MIN', 'NE', 'NO', 'NYG',
        'NYJ', 'PHI', 'PIT', 'SEA', 'SF', 'TB', 'TEN', 'WAS'
    ]

    # Create a regex pattern for team abbreviations
    team_pattern = '|'.join(nfl_teams)

    # Complete regex pattern
    pattern = rf"""
        ^(?P<Player>.*?)\s+                      # Player name (non-greedy)
        (?P<Team>{team_pattern})\s*              # Team abbreviation
        \((?P<Bye>\d+)\)                         # Bye week inside parentheses
        (?:\s+(?P<Other>\w+))?$                  # Optional additional info (e.g., 'O')
    """

    # Apply the regex pattern to extract components
    df[['Player_Name', 'Team', 'Bye_Week', 'Other']] = df['Player Team (Bye)'].str.extract(pattern, flags=re.VERBOSE)

    # Define a pattern for entries without team abbreviations
    pattern_no_team = r"^(?P<Player>.+)$"

    # Extract player names where team information is missing
    df[['Player_Name_NoTeam']] = df['Player Team (Bye)'].str.extract(pattern_no_team)

    # Combine the two extraction results
    df['Player_Name'] = df['Player_Name'].combine_first(df['Player_Name_NoTeam'])
    df.drop(['Player_Name_NoTeam', 'Other', 'Bye_Week', 'Team', 'Player Team (Bye)', 'Rank', 'POS', 'FFC'], axis=1, inplace=True)

    pattern = r'\(.*\)$'
    mask = df['Player_Name'].str.contains(pattern, regex=True)
    players_with_parentheses = df[mask]
    df.loc[mask, 'Player_Name'] = df.loc[mask, 'Player_Name'].str.replace(r'\s*\(.*\)$', '', regex=True)

    # Rename columns to match desired stats based on actual headers
    df.rename(columns={
        "Player_Name": "Player Name",
        "ESPN": "ESPN ADP",
        "Sleeper": "Sleeper ADP",
        "NFL": "NFL ADP",
        "RTSports": "RTSports ADP",
        "AVG": "Average ADP"
    }, inplace=True)

    return df

##############################
# 4) MERGING & FILTERING
##############################

def merge_offensive_dataframes_no_agg(dfs):
    """
    Merges multiple DataFrames (in the 48-col schema) by "Player Name" WITHOUT numeric summation.
    If a column is NaN in the main DataFrame but non-NaN in the second (dup), we use the second.
    Otherwise, we keep the main value.
    """
    final = pd.DataFrame(columns=FINAL_COLUMNS)

    for i, df in enumerate(dfs):
        df = df.drop_duplicates(subset=["Player Name"])

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

    # To silence DeprecationWarnings, select only non-grouping columns
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

def add_positional_adp(df):
    """
    Adds a 'Positional ADP' column to the DataFrame, which calculates the ADP rank within each position group.
    """

    # Group by 'Position' and rank the 'Average ADP' within each position
    df['Positional ADP'] = df.groupby('Position')['Average ADP'].rank(method='first', ascending=True)

    # Fill NaN values with a default rank (e.g., 0 or 999)
    df['Positional ADP'] = df['Positional ADP'].fillna(301).astype(int)

    return df

##############################
# 5) CREATE FINAL DATASET
##############################

def create_final_dataset(year=2022):
    """
    1. Scrape passing/rushing/receiving/kicking/team defense/special teams
       and remove single-team rows for multi-team players right after scraping.
    2. Transform each to 48-col schema, preserving NaN for missing stats.
    3. Merge all offensive DataFrames by Player Name, coalescing NaN columns.
    4. Remove single-team rows if multi-team row is present (final safety check),
       then pick the best multi-team row if duplicates remain.
    5. Append defense rows and exclude unwanted defense rows.
    6. Append special teams stats without creating unwanted '_st' columns.
    7. Merge ADP data into the final dataset.
    8. Force final_df["Season"] = year so every row has correct year.
    """
    print(f"=== Creating final dataset for year {year} ===\n")

    # 1) Scrape
    df_pass = get_passing_stats(year)
    time.sleep(2)
    df_rush = get_rushing_stats(year)
    time.sleep(2)
    df_recv = get_receiving_stats(year)
    time.sleep(2)
    df_kick = get_kicking_stats(year)
    time.sleep(2)
    df_def_opp = get_team_defense_stats(year)
    time.sleep(2)
    df_st = get_special_teams_stats(year)
    time.sleep(2)
    df_adp = get_adp_stats(year=year)
    time.sleep(2)
    df_two_point_conversion_final = get_and_transform_two_point_conversion_stats(year)

    # 2) Transform
    df_pass_final = transform_passing(df_pass, year)
    df_rush_final = transform_rushing(df_rush, year)
    df_recv_final = transform_receiving(df_recv, year)
    df_kick_final = transform_kicking(df_kick, year)
    df_def_final = transform_team_defense(df_def_opp, year)
    df_st_final = transform_special_teams(df_st, year)
    df_adp_final = transform_ADP_stats(df_adp, year)

    # 3) Merge offense (NaN coalescing)
    df_offense = merge_offensive_dataframes_no_agg([df_pass_final, df_rush_final, df_recv_final, df_kick_final])

    # 4) Remove single-team if multi-team row is present, then pick best multi-team row
    df_offense = keep_combined_multiteam_rows(df_offense)
    df_offense = pick_best_multiteam_row(df_offense, stat_col="Rushing Attempts")

    # 5) Append defense
    final_df = pd.concat([df_offense, df_def_final], ignore_index=True)

    # 6) Append special teams
    # Use 'update' to merge special teams stats into team defense rows 
    final_df.set_index("Player Name", inplace=True)  # Ensure consistent column name
    df_st_final.set_index("Player Name", inplace=True)
    final_df.update(df_st_final)
    final_df.reset_index(inplace=True)

    points_from_points = pd.read_csv(f"Capstone/data/{year}/fantasy_points_from_points_allowed_{year}.csv")
    final_df = final_df.merge(points_from_points, on='Player Name', how='left')
    final_df = final_df.merge(df_two_point_conversion_final, on='Player Name', how='left')

    # 7) Merge ADP data into final_df
    # Ensure that 'Player Name' is the column name in both dataframes
    # If 'Player Name' in final_df is named differently, adjust accordingly
    if 'Player Name' not in final_df.columns:
        raise KeyError("final_df must contain a 'Player Name' column for merging.")
    
    if 'Player Name' not in df_adp_final.columns:
        raise KeyError("df_adp_final must contain a 'Player Name' column for merging.")

    if df_adp_final['Player Name'].duplicated().any():
        print("Warning: df_adp_final contains duplicate Player Name entries. Aggregating by mean ADP.")
        # Example: Aggregate ADP by mean if duplicates exist
        df_adp_final = df_adp_final.groupby('Player Name', as_index=False).mean()

    final_df = final_df.merge(df_adp_final, on='Player Name', how='left')
    
    final_df = final_df[final_df['Player Name'] != 'League Average']

    final_df = final_df[~final_df['Position'].isin(['DB', 'FS', 'LB', 'LT', 'RG', 'RT', 'S'])]

    # 8) Force Season = year
    final_df["Season"] = year

    final_df = add_positional_adp(final_df)
    final_df['Player ID'] = final_df.reset_index().index + 1

    # Final Cleanup: Drop any unwanted '_x', '_y', or '_dup' columns if they somehow exist
    unwanted_suffixes = ['_x', '_y', '_dup']
    for suffix in unwanted_suffixes:
        cols_to_drop = [col for col in final_df.columns if col.endswith(suffix)]
        if cols_to_drop:
            final_df.drop(columns=cols_to_drop, inplace=True)
            print(f"\nDropped unwanted '{suffix}' suffixed columns: {cols_to_drop}")

    return final_df


def main(year=2022, save_csv=True):

    df_final = create_final_dataset(year=year)

    df_final['ESPN ADP'] = df_final['ESPN ADP'].fillna(301)
    df_final['NFL ADP'] = df_final['NFL ADP'].fillna(301)
    df_final['RTSports ADP'] = df_final['RTSports ADP'].fillna(301)
    df_final['Sleeper ADP'] = df_final['Sleeper ADP'].fillna(301)
    df_final['Average ADP'] = df_final['Average ADP'].fillna(301)

    if save_csv:
        os.makedirs(f"Capstone/data/{year}", exist_ok=True)
        out_file = f"Capstone/data/{year}/nfl_{year}_final_data.csv"
        df_final.to_csv(out_file, index=False)
        print(f"\nSaved final dataset to {out_file}")

if __name__ == "__main__":
    main(year=2022, save_csv=True)
