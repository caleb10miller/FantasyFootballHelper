import requests
import json
import pandas as pd
from datetime import datetime
import time
import random
from collections import defaultdict
import os

def fetch_injury_data(season, season_type, week, api_key, max_retries=5, base_delay=5):
    """Fetch injury data for a specific season, type, and week with exponential backoff"""
    base_url = f"https://api.sportradar.com/nfl/official/trial/v5/en/seasons/{season}/{season_type}/{week}/injuries.json"
    url = f"{base_url}?api_key={api_key}"
    
    headers = {"accept": "application/json"}
    
    for attempt in range(max_retries):
        try:
            # Add jitter to avoid synchronized retries
            jitter = random.uniform(0, 1)
            time.sleep(base_delay + jitter)
            
            print(f"\nFetching URL: {url}")
            response = requests.get(url, headers=headers)
            print(f"Response status code: {response.status_code}")
            
            if response.status_code == 429:  # Too Many Requests
                wait_time = (2 ** attempt) * base_delay  # Exponential backoff
                print(f"Rate limited. Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
                continue
                
            response.raise_for_status()
            data = response.json()
            return data
            
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:  # Last attempt
                print(f"Error fetching data for season {season}, {season_type}, week {week}: {str(e)}")
                return None
            wait_time = (2 ** attempt) * base_delay
            print(f"Request failed. Waiting {wait_time} seconds before retry...")
            time.sleep(wait_time)

def process_injury_data(data, season):
    """Extract relevant fields from injury data and aggregate serious injuries by player"""
    if not data or 'teams' not in data:
        print("No teams found in data")
        return []
    
    # Use defaultdict to automatically initialize lists for new players
    player_injuries = defaultdict(set)  # Using set to automatically handle duplicates
    
    for team in data['teams']:
        for player in team.get('players', []):
            player_name = player.get('name', '')
            
            # Process each injury
            for injury in player.get('injuries', []):
                practice_status = injury.get('practice', {}).get('status', '')
                
                # Only include serious injuries (Did Not Participate In Practice)
                if practice_status == 'Did Not Participate In Practice':
                    injury_desc = injury.get('primary', '')
                    if injury_desc:
                        player_injuries[player_name].add(injury_desc)
    
    # Convert to list of records
    processed_data = []
    for player_name, injuries in player_injuries.items():
        if injuries:  # Only include players with serious injuries
            processed_data.append({
                'season': season,
                'player_name': player_name,
                'injuries': list(injuries)
            })
    
    return processed_data

def get_weeks_for_season_type(season_type):
    """Get the number of weeks for a given season type"""
    if season_type == 'PRE':
        return range(1, 5)  # Preseason typically 4 weeks
    elif season_type == 'REG':
        return range(1, 19)  # Regular season 18 weeks
    else:  # PST
        return range(1, 5)  # Postseason typically 4 weeks

def main():
    api_key = "iM1ISrY67kUlCr2pQE4UFwWAZqbd98FRrWBL24s1"
    seasons = range(2018, 2025)  # 2018-2024
    season_types = ['PRE', 'REG', 'PST']
    
    for season in seasons:
        print(f"\nProcessing season {season}...")
        
        # Create year-specific directory
        year_dir = f'data/{season}'
        os.makedirs(year_dir, exist_ok=True)
        
        season_injuries = []
        
        for season_type in season_types:
            print(f"\nProcessing {season_type} season...")
            weeks = get_weeks_for_season_type(season_type)
            
            for week in weeks:
                print(f"Fetching {season_type} week {week}...")
                data = fetch_injury_data(season, season_type, week, api_key)
                
                if data:
                    processed_data = process_injury_data(data, season)
                    season_injuries.extend(processed_data)
                    
                    # Save progress after each successful fetch
                    if processed_data:
                        df = pd.DataFrame(season_injuries)
                        output_file = f'{year_dir}/nfl_injuries_{season_type}_week{week}.csv'
                        df.to_csv(output_file, index=False)
                        print(f"Progress saved: {len(df)} records collected so far")
        
        # Save final season data
        if season_injuries:
            df = pd.DataFrame(season_injuries)
            output_file = f'{year_dir}/nfl_injuries_{season}_complete.csv'
            df.to_csv(output_file, index=False)
            print(f"\nSeason {season} data saved to {output_file}")
            print(f"Total records for season {season}: {len(df)}")

if __name__ == "__main__":
    main()