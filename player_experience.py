import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import urllib.parse
import os

def get_first_season(player_name, max_retries=3, base_delay=3):
    """
    Get a player's first season in the NFL, handling both direct stat tables and disambiguation pages.
    Implements exponential backoff for rate limiting.
    Returns None if there's an error or if the information can't be found.
    """
    for attempt in range(max_retries):
        try:
            # URL encode the player name
            encoded_name = urllib.parse.quote(player_name)
            url = f"https://www.pro-football-reference.com/search/search.fcgi?search={encoded_name}"
            
            # Add exponential delay between attempts
            delay = base_delay * (2 ** attempt)
            time.sleep(delay)
            
            resp = requests.get(url)
            
            # If we get rate limited, wait longer and retry
            if resp.status_code == 429:
                print(f"Rate limited, waiting {delay} seconds before retry...")
                continue
                
            resp.raise_for_status()
            
            soup = BeautifulSoup(resp.text, "html.parser")
            
            # Try to find a stats table first
            tbody = soup.find("tbody")
            if tbody:
                # Get the first row
                first_row = tbody.find("tr")
                if first_row:
                    # Find the season cell (it's in a th element with data-stat="year_id")
                    season_cell = first_row.find("th", {"data-stat": "year_id"})
                    if season_cell:
                        # The year is in the text of the th element
                        return int(season_cell.text)
            # If no stats table, look for a disambiguation list of players
            player_links = soup.select('div.search-item-url a[href^="/players/"]')
            if not player_links:
                # Try the older format: look for the first <a> under the results list
                player_links = soup.select('div#content a[href^="/players/"]')
            if player_links:
                # Follow the first player link
                player_url = 'https://www.pro-football-reference.com' + player_links[0]['href']
                # Fetch the player's page
                time.sleep(1)
                player_resp = requests.get(player_url)
                player_resp.raise_for_status()
                player_soup = BeautifulSoup(player_resp.text, "html.parser")
                stats_tbody = player_soup.find("tbody")
                if stats_tbody:
                    first_row = stats_tbody.find("tr")
                    if first_row:
                        season_cell = first_row.find("th", {"data-stat": "year_id"})
                        if season_cell:
                            return int(season_cell.text)
            return None
            
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:  # Last attempt
                print(f"Error getting first season for {player_name} after {max_retries} attempts: {str(e)}")
                return None
            continue
            
    return None

def scrape_first_seasons(df, output_file="data/final_data/player_first_seasons.csv"):
    """
    Scrape first seasons for all unique players in the dataset and save to CSV.
    """
    print("Scraping first seasons for all players...")
    
    # Get unique players
    unique_players = df["Player Name"].unique()
    total_players = len(unique_players)
    
    # Create DataFrame to store results
    results = []
    
    # Add a longer initial delay to avoid immediate rate limiting
    time.sleep(10)
    
    for i, player_name in enumerate(unique_players, 1):
        print(f"Processing player {i}/{total_players}: {player_name}")
        
        # Get first season with retries and backoff
        first_season = get_first_season(player_name)
        results.append({
            "Player Name": player_name,
            "First Season": first_season
        })
        
        # Add a delay between players to avoid rate limiting
        time.sleep(3)
    
    # Create DataFrame and save to CSV
    first_seasons_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    first_seasons_df.to_csv(output_file, index=False)
    print(f"\nSaved first seasons data to {output_file}")
    
    return first_seasons_df

def calculate_years_of_experience(df, first_seasons_file="data/final_data/player_first_seasons.csv"):
    """
    Calculate years of experience for each player in each season using the scraped first seasons.
    """
    # Load first seasons data
    first_seasons_df = pd.read_csv(first_seasons_file)
    
    # Create a dictionary for quick lookup
    first_seasons_dict = dict(zip(first_seasons_df["Player Name"], first_seasons_df["First Season"]))
    
    # Calculate years of experience
    df["Years_of_Experience"] = df.apply(
        lambda row: row["Season"] - first_seasons_dict.get(row["Player Name"], row["Season"])
        if first_seasons_dict.get(row["Player Name"]) is not None
        else None,
        axis=1
    )
    
    return df

def main():
    # Load the long format dataset
    df = pd.read_csv("data/final_data/nfl_stats_long_format_with_context_filtered.csv")
    
    # Scrape first seasons if not already done
    first_seasons_file = "data/final_data/player_first_seasons.csv"
    scrape_first_seasons(df, first_seasons_file)
    
    # Calculate years of experience
    df = calculate_years_of_experience(df, first_seasons_file)
    
    # Save updated dataset
    output_file = "data/final_data/nfl_stats_long_format_with_context_filtered_with_experience.csv"
    df.to_csv(output_file, index=False)
    print(f"\nSaved updated dataset with years of experience to {output_file}")
    

if __name__ == "__main__":
    main() 