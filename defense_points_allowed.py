import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import os

TEAM_ABBREVIATIONS = [
    "crd", "atl", "rav", "buf", "car", "chi", "cin", "cle",
    "dal", "den", "det", "gnb", "htx", "clt", "jax", "kan",
    "rai", "sdg", "ram", "mia", "min", "nwe", "nor", "nyg",
    "nyj", "phi", "pit", "sfo", "sea", "tam", "oti", "was"
]

def calculate_fantasy_points(points_allowed: int) -> int:
    """
    Returns the fantasy points based on the points_allowed value:
      0        => +5
      1-6      => +4
      7-13     => +3
      14-17    => +1
      18-27    =>  0
      28-34    => -1
      35-45    => -3
      46+      => -5
    """
    if points_allowed == 0:
        return 5
    elif 1 <= points_allowed <= 6:
        return 4
    elif 7 <= points_allowed <= 13:
        return 3
    elif 14 <= points_allowed <= 17:
        return 1
    elif 18 <= points_allowed <= 27:
        return 0
    elif 28 <= points_allowed <= 34:
        return -1
    elif 35 <= points_allowed <= 45:
        return -3
    elif points_allowed >= 46:
        return -5
    return 0  # fallback for unexpected data

def scrape_team_schedule(team_abbr: str, year: int) -> pd.DataFrame:
    """
    Scrapes the entire 'games' table from Pro-Football-Reference for a given
    team abbreviation (team_abbr) and year, returning a DataFrame with
    only the 'Opp' column plus a 'Fantasy Points' column.
    """
    url = f"https://www.pro-football-reference.com/teams/{team_abbr}/{year}.htm"
    print(f"Scraping {url}")
    
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/58.0.3029.110 Safari/537.36"
        )
    }
    
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    
    soup = BeautifulSoup(response.text, "html.parser")
    
    # Locate the 'games' table
    games_table = soup.find("table", {"id": "games"})
    if not games_table:
        print(f"WARNING: No schedule table found for {team_abbr} in {year}.")
        return pd.DataFrame()  # Return empty DataFrame if not found
    
    # Grab the column headers from the last row in <thead>
    header_rows = games_table.find("thead").find_all("tr")
    col_headers = [th.get_text(strip=True) for th in header_rows[-1].find_all("th")]
    
    # Extract the data rows in <tbody>
    body_rows = games_table.find("tbody").find_all("tr")
    
    # We'll build a list of dicts, then convert it to a DataFrame
    row_dicts = []
    for row in body_rows:
        cells = row.find_all(["th", "td"])
        cell_texts = [cell.get_text(strip=True) for cell in cells]
        
        row_data = {}
        for col_name, col_value in zip(col_headers, cell_texts):
            row_data[col_name] = col_value
        
        row_dicts.append(row_data)
    
    # Convert to DataFrame
    df_team = pd.DataFrame(row_dicts)
    
    # Keep only 'Opp' for demonstration
    if "Opp" in df_team.columns:
        df_team = df_team[["Opp"]]
    else:
        # If the 'Opp' column doesn't exist, just return empty
        return pd.DataFrame()
    
    # 1) Convert Opp to numeric (some rows might be blank or text)
    df_team["Opp"] = pd.to_numeric(df_team["Opp"], errors="coerce")
    
    # 2) Create 'Fantasy Points' column by applying your scoring function
    df_team["Fantasy Points"] = df_team["Opp"].apply(
        lambda x: calculate_fantasy_points(x) if pd.notnull(x) else None
    )

    df_output = pd.DataFrame()
    df_output['Fantasy Points From Points'] = df_team['Fantasy Points']
    
    return df_output


def scrape_all_teams(year: int) -> pd.DataFrame:
    """
    Loops through all team abbreviations, scrapes the full schedule data
    (all columns), and returns one combined DataFrame.
    """
    df_all = pd.DataFrame()
    
    for team in TEAM_ABBREVIATIONS:
        # Scrape a single team's schedule into a DataFrame
        df_team = scrape_team_schedule(team, year)

        df_team_summed = pd.DataFrame(df_team.sum())

        df_team_summed.rename(columns={0: 'Fantasy Points From Points'}, inplace=True)

        df_team_summed['Player Name'] = team
        
        # Concatenate with the master DataFrame
        df_all = pd.concat([df_all, df_team_summed], ignore_index=True)
        
        # Sleep to avoid rate-limiting
        print("Sleeping for 2 seconds...")
        time.sleep(2)

    TEAM_NAME_MAP = {
        "sfo": "San Francisco 49ers DST",
        "buf": "Buffalo Bills DST",
        "rav": "Baltimore Ravens DST",
        "nyj": "New York Jets DST",
        "cin": "Cincinnati Bengals DST",
        "dal": "Dallas Cowboys DST",
        "was": "Washington Commanders DST",
        "phi": "Philadelphia Eagles DST",
        "nor": "New Orleans Saints DST",
        "pit": "Pittsburgh Steelers DST",
        "nwe": "New England Patriots DST",
        "jax": "Jacksonville Jaguars DST",
        "tam": "Tampa Bay Buccaneers DST",
        "den": "Denver Broncos DST",
        "oti": "Tennessee Titans DST",
        "kan": "Kansas City Chiefs DST",
        "gnb": "Green Bay Packers DST",
        "nyg": "New York Giants DST",
        "car": "Carolina Panthers DST",
        "cle": "Cleveland Browns DST",
        "sdg": "Los Angeles Chargers DST",
        "ram": "Los Angeles Rams DST",
        "atl": "Atlanta Falcons DST",
        "mia": "Miami Dolphins DST",
        "sea": "Seattle Seahawks DST",
        "rai": "Las Vegas Raiders DST",
        "htx": "Houston Texans DST",
        "min": "Minnesota Vikings DST",
        "det": "Detroit Lions DST",
        "clt": "Indianapolis Colts DST",
        "crd": "Arizona Cardinals DST",
        "chi": "Chicago Bears DST"
    }

    df_all["Player Name"] = df_all["Player Name"].map(TEAM_NAME_MAP)
    
    return df_all

def points_from_points_allowed(year=2022):
    df_final = scrape_all_teams(year)
    print("\nPreview of final DataFrame:")
    print(df_final.head(10))
    
    # write to CSV
    os.makedirs(f"Capstone/data/{year}", exist_ok=True)
    csv_filename = f"Capstone/data/{year}/fantasy_points_from_points_allowed_{year}.csv"
    df_final.to_csv(csv_filename, index=False)
    print(f"\nSaved to {csv_filename}")

if __name__ == "__main__":
    points_from_points_allowed(2022)
