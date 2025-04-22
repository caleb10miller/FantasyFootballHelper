#!/usr/bin/env python3
"""
Combine Severity Injuries

This script combines all processed data with severity injuries into a single file
for model training.
"""

import pandas as pd
import os
import glob
from pathlib import Path

def combine_all_seasons(base_dir='data', output_file='data/final_data/nfl_stats_with_severity_injuries.csv'):
    """
    Combine all processed data with severity injuries into a single file.
    
    Parameters:
        base_dir (str): Base directory for data files
        output_file (str): Path to save the combined data
    """
    print("Combining all seasons with severity injuries...")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Find all files with severity injuries
    all_files = []
    for year in range(2018, 2025):
        file_path = os.path.join(base_dir, str(year), f"{year}_final_data_with_severity_injuries.csv")
        if os.path.exists(file_path):
            all_files.append(file_path)
    
    if not all_files:
        print("No files found with severity injuries. Please run injury_severity_classifier.py first.")
        return
    
    # Read and combine all files
    dfs = []
    for file_path in all_files:
        print(f"Reading {file_path}")
        df = pd.read_csv(file_path)
        dfs.append(df)
    
    # Combine all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Save the combined data
    print(f"Saving combined data to {output_file}")
    combined_df.to_csv(output_file, index=False)
    
    # Print some stats
    print(f"\nCombined data stats:")
    print(f"Total records: {len(combined_df)}")
    print(f"Total players: {combined_df['Player Name'].nunique()}")
    print(f"Seasons: {combined_df['Season'].unique()}")
    
    # Print injury stats
    players_with_injuries = len(combined_df[combined_df['Total_Injuries'] > 0])
    players_with_critical = len(combined_df[combined_df['has_critical_injury'] > 0])
    players_with_serious = len(combined_df[combined_df['has_serious_injury'] > 0])
    
    print(f"\nInjury stats:")
    print(f"Players with injuries: {players_with_injuries} ({(players_with_injuries/len(combined_df))*100:.1f}%)")
    print(f"Players with critical injuries: {players_with_critical} ({(players_with_critical/len(combined_df))*100:.1f}%)")
    print(f"Players with serious injuries: {players_with_serious} ({(players_with_serious/len(combined_df))*100:.1f}%)")
    print(f"Average injury severity score: {combined_df['injury_severity_score'].mean():.2f}")
    
    # Print position-specific stats
    print("\nPosition-specific injury stats:")
    for position in combined_df['Position'].unique():
        pos_df = combined_df[combined_df['Position'] == position]
        pos_total = len(pos_df)
        pos_injured = len(pos_df[pos_df['Total_Injuries'] > 0])
        pos_critical = len(pos_df[pos_df['has_critical_injury'] > 0])
        pos_serious = len(pos_df[pos_df['has_serious_injury'] > 0])
        
        print(f"{position}:")
        print(f"  Total players: {pos_total}")
        print(f"  Players with injuries: {pos_injured} ({(pos_injured/pos_total)*100:.1f}%)")
        print(f"  Players with critical injuries: {pos_critical} ({(pos_critical/pos_total)*100:.1f}%)")
        print(f"  Players with serious injuries: {pos_serious} ({(pos_serious/pos_total)*100:.1f}%)")
        print(f"  Average injury severity score: {pos_df['injury_severity_score'].mean():.2f}")
    
    return combined_df

if __name__ == "__main__":
    combine_all_seasons() 