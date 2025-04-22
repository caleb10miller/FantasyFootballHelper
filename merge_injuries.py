import pandas as pd
import os
from ast import literal_eval
from collections import Counter

def normalize_injury_type(injury):
    """
    Normalize injury types to remove variations and standardize format
    """
    injury = injury.lower().strip()
    
    # Normalize rest-related injuries
    if any(term in injury for term in ['nir', 'rest', 'vet rest']):
        return 'rest'
    
    # Normalize quad-related injuries
    if any(term in injury for term in ['quad', 'quadricep']):
        return 'quad'
    
    # Normalize personal/illness
    if any(term in injury for term in ['personal', 'illness']):
        return 'illness'
    
    # Remove any extra spaces and standardize format
    return injury.strip()

def merge_injuries_by_player(input_file):
    """
    Read a season's injury file and merge injuries by player name, creating ML-friendly format
    """
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    # Convert string representation of lists to actual lists
    df['injuries'] = df['injuries'].apply(lambda x: literal_eval(x) if isinstance(x, str) else [x])
    
    # Group by season and player_name, combining injuries into a list and counting occurrences
    merged_df = df.groupby(['season', 'player_name'])['injuries'].agg(
        lambda x: Counter([normalize_injury_type(item) for sublist in x for item in sublist])
    ).reset_index()
    
    # Convert Counter objects to a list of tuples
    merged_df['injury_counts'] = merged_df['injuries'].apply(lambda x: sorted(x.items(), key=lambda x: (-x[1], x[0])))
    
    # Create separate columns for each injury type with counts
    all_injuries = set()
    for counts in merged_df['injury_counts']:
        all_injuries.update(count[0] for count in counts)
    
    # Add columns for each injury type
    for injury in sorted(all_injuries):
        merged_df[f'injury_{injury}'] = merged_df['injury_counts'].apply(
            lambda x: next((count for injury_type, count in x if injury_type == injury), 0)
        )
    
    # Drop the original injuries and injury_counts columns
    merged_df = merged_df.drop(['injuries', 'injury_counts'], axis=1)
    
    # Sort by player name
    merged_df = merged_df.sort_values('player_name')
    
    return merged_df

def process_all_seasons():
    """
    Process all season files in the data directory
    """
    # Get all season directories
    base_dir = 'data'
    seasons = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    for season in seasons:
        season_dir = os.path.join(base_dir, season)
        complete_file = os.path.join(season_dir, f'nfl_injuries_{season}_complete.csv')
        
        if os.path.exists(complete_file):
            print(f"\nProcessing season {season}...")
            
            # Merge injuries for this season
            merged_df = merge_injuries_by_player(complete_file)
            
            # Save merged data
            output_file = os.path.join(season_dir, f'nfl_injuries_{season}_merged.csv')
            merged_df.to_csv(output_file, index=False)
            
            print(f"Merged data saved to: {output_file}")
            print(f"Total unique players: {len(merged_df)}")
            print(f"Total injury types: {len(merged_df.columns) - 2}")  # -2 for season and player_name
            
            # Display first few entries as example
            print("\nSample of merged data:")
            print(merged_df.head().to_string())

if __name__ == "__main__":
    print("Starting injury data merge process...")
    process_all_seasons()
    print("\nMerge process complete!") 