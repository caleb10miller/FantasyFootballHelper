#!/usr/bin/env python3
"""
Merges injury data with finalized stats data for years 2018-2024.
Applies the same name cleaning logic from MainScraper.py to ensure consistent player names.
"""

import pandas as pd
import re
import os
from pathlib import Path

def format_initials(name):
    """
    Adds periods between uppercase letters if they are initials (e.g., 'JK Dobbins' -> 'J.K. Dobbins').
    """
    return re.sub(r'\b([A-Z])([A-Z])\b', r'\1.\2.', name)

def clean_names(names, reference_names):
    """
    Cleans and standardizes player names by removing suffixes and dynamically formatting initials.
    
    Parameters:
        names (list): List of player names to be cleaned
        reference_names (set): Set of reference names to check for matches after cleaning
    
    Returns:
        list: List of cleaned and standardized player names
    """
    # Convert all names to strings first
    names = [str(name) if pd.notnull(name) else '' for name in names]
    
    # Define a regex pattern to match unwanted suffixes
    pattern = r"( Jr\. O| Sr\. O| III O| Jr\.| Sr\.| II O| II| III| O)$"
    
    # Remove suffixes
    cleaned_names = [re.sub(pattern, '', name) for name in names]
    
    for i in range(len(cleaned_names)):
        # Apply custom replacements for known mismatches
        if cleaned_names[i] == "Gabe Davis":
            cleaned_names[i] = "Gabriel Davis"
        elif cleaned_names[i] == "Joshiua Palmer":
            cleaned_names[i] = "Josh Palmer"
        else:
            # Dynamically format initials and check against reference names
            formatted_name = format_initials(cleaned_names[i])
            if formatted_name in reference_names:
                cleaned_names[i] = formatted_name
    
    return cleaned_names

def process_year(year):
    """
    Process a single year's data by merging injuries with stats.
    
    Parameters:
        year (int): The year to process
    """
    print(f"\nProcessing year {year}...")
    
    # Define file paths
    data_dir = Path(f"data/{year}")
    finalized_data_path = data_dir / f"{year}_finalized_data.csv"
    injuries_data_path = data_dir / f"nfl_injuries_{year}_merged.csv"
    output_path = data_dir / f"{year}_final_data_with_injuries.csv"
    
    # Check if files exist
    if not finalized_data_path.exists():
        print(f"Warning: {finalized_data_path} not found")
        return
    if not injuries_data_path.exists():
        print(f"Warning: {injuries_data_path} not found")
        return
        
    # Read the data files
    print(f"Reading {finalized_data_path}")
    finalized_data = pd.read_csv(finalized_data_path)
    print(f"Reading {injuries_data_path}")
    injuries_data = pd.read_csv(injuries_data_path)
    
    # Get reference names from finalized data
    reference_names = set(finalized_data['Player Name'])
    
    # Clean names in injuries data
    print("Cleaning player names in injuries data...")
    injuries_data.rename(columns={'player_name': 'Player Name'}, inplace=True)
    injuries_data['Player Name'] = clean_names(injuries_data['Player Name'].tolist(), reference_names)
    
    # Merge the datasets
    print("Merging datasets...")
    merged_data = finalized_data.merge(injuries_data, on='Player Name', how='left')
    
    # Fill nulls with 0s for all injury-related columns
    print("Filling null injury values with 0s...")
    injury_columns = [col for col in merged_data.columns if col.startswith('injury_')]
    merged_data[injury_columns] = merged_data[injury_columns].fillna(0)
    
    # Save the merged data
    print(f"Saving merged data to {output_path}")
    merged_data.to_csv(output_path, index=False)
    print(f"Successfully processed year {year}")
    
    # Print some stats
    total_players = len(finalized_data)
    matched_players = len(merged_data[merged_data['season'].notna()])
    print(f"\nStats for {year}:")
    print(f"Total players in finalized data: {total_players}")
    print(f"Players with injury data: {matched_players}")
    print(f"Match rate: {(matched_players/total_players)*100:.1f}%")

def main():
    """Main function to process all years from 2018 to 2024."""
    print("Starting to merge injuries with stats data...")
    
    # Process each year from 2018 to 2024
    years = range(2018, 2025)
    for year in years:
        try:
            process_year(year)
        except Exception as e:
            print(f"Error processing {year}: {e}")
    
    print("\nProcessing complete!")

if __name__ == "__main__":
    main() 