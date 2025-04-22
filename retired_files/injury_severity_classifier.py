#!/usr/bin/env python3
"""
Injury Severity Classifier

This module provides functions to classify NFL injuries by severity and generate
features that better capture the impact of injuries on player performance.
"""

import pandas as pd
import numpy as np
import re
from collections import defaultdict

# Define injury severity categories
SEVERITY_LEVELS = {
    'CRITICAL': 3,  # Season-ending or career-threatening injuries
    'SERIOUS': 2,   # Multi-week injuries that significantly impact performance
    'MINOR': 1,     # Short-term injuries with minimal impact
    'NONE': 0       # No injury or rest-related
}

# Define injury types and their severity levels
INJURY_SEVERITY_MAP = {
    # Critical injuries (severity 3)
    'acl': 'CRITICAL',
    'achilles': 'CRITICAL',
    'pectoral': 'CRITICAL',
    'rupture': 'CRITICAL',
    'herniated': 'CRITICAL',
    'fracture': 'CRITICAL',
    'torn': 'CRITICAL',
    'dislocation': 'CRITICAL',
    'spinal': 'CRITICAL',
    'neck': 'CRITICAL',
    'concussion': 'CRITICAL',
    
    # Serious injuries (severity 2)
    'mcl': 'SERIOUS',
    'ankle': 'SERIOUS',
    'hamstring': 'SERIOUS',
    'calf': 'SERIOUS',
    'groin': 'SERIOUS',
    'shoulder': 'SERIOUS',
    'knee': 'SERIOUS',
    'quad': 'SERIOUS',
    'biceps': 'SERIOUS',
    'triceps': 'SERIOUS',
    'rib': 'SERIOUS',
    'ribs': 'SERIOUS',
    'back': 'SERIOUS',
    'hip': 'SERIOUS',
    'thigh': 'SERIOUS',
    'tibia': 'SERIOUS',
    'forearm': 'SERIOUS',
    'wrist': 'SERIOUS',
    'hand': 'SERIOUS',
    'finger': 'SERIOUS',
    'thumb': 'SERIOUS',
    'toe': 'SERIOUS',
    'foot': 'SERIOUS',
    'heel': 'SERIOUS',
    'elbow': 'SERIOUS',
    'face': 'SERIOUS',
    'abdomen': 'SERIOUS',
    'oblique': 'SERIOUS',
    'kidney': 'SERIOUS',
    'stinger': 'SERIOUS',
    
    # Minor injuries (severity 1)
    'illness': 'MINOR',
    'tooth': 'MINOR',
    'shin': 'MINOR',
    'rest': 'NONE',
    'not injury related': 'NONE',
    'other': 'MINOR'
}

# Position-specific injury impact multipliers
POSITION_INJURY_IMPACT = {
    'QB': {
        'shoulder': 1.5,
        'hand': 1.5,
        'finger': 1.5,
        'thumb': 1.5,
        'wrist': 1.5,
        'elbow': 1.5,
        'rib': 1.5,
        'ribs': 1.5,
        'back': 1.5,
        'concussion': 1.5
    },
    'RB': {
        'knee': 1.5,
        'ankle': 1.5,
        'hamstring': 1.5,
        'foot': 1.5,
        'toe': 1.5,
        'concussion': 1.5
    },
    'WR': {
        'hamstring': 1.5,
        'ankle': 1.5,
        'knee': 1.5,
        'concussion': 1.5
    },
    'TE': {
        'hamstring': 1.5,
        'ankle': 1.5,
        'knee': 1.5,
        'concussion': 1.5
    }
}

def classify_injury_severity(injury_desc, position=None):
    """
    Classify an injury by severity level based on the injury description.
    
    Parameters:
        injury_desc (str): Description of the injury
        position (str, optional): Player position to apply position-specific multipliers
    
    Returns:
        int: Severity level (0-3)
    """
    if not injury_desc or injury_desc.lower() in ['rest', 'not injury related']:
        return SEVERITY_LEVELS['NONE']
    
    injury_desc = injury_desc.lower()
    
    # Check for critical injuries first
    for term, severity in INJURY_SEVERITY_MAP.items():
        if term in injury_desc:
            base_severity = SEVERITY_LEVELS[severity]
            
            # Apply position-specific multipliers if position is provided
            if position and position in POSITION_INJURY_IMPACT:
                for pos_injury, multiplier in POSITION_INJURY_IMPACT[position].items():
                    if pos_injury in injury_desc:
                        # Cap the severity at 3
                        return min(3, base_severity * multiplier)
            
            return base_severity
    
    # Default to minor severity if no match found
    return SEVERITY_LEVELS['MINOR']

def create_injury_features(injury_data, player_data=None):
    """
    Create advanced injury features from raw injury data.
    
    Parameters:
        injury_data (pd.DataFrame): DataFrame with injury data
        player_data (pd.DataFrame, optional): DataFrame with player data including position
    
    Returns:
        pd.DataFrame: DataFrame with new injury features
    """
    # Create a copy of the injury data
    features_df = injury_data.copy()
    
    # Initialize new columns with float dtype to avoid dtype incompatibility
    features_df['injury_severity_score'] = 0.0
    features_df['critical_injury_count'] = 0
    features_df['serious_injury_count'] = 0
    features_df['minor_injury_count'] = 0
    features_df['position_specific_injury_score'] = 0.0
    
    # Get all injury columns
    injury_columns = [col for col in injury_data.columns if col.startswith('injury_')]
    
    # Calculate total injuries if not already present
    if 'Total_Injuries' not in features_df.columns:
        features_df['Total_Injuries'] = features_df[injury_columns].sum(axis=1)
    
    # Process each player's injuries
    for idx, row in features_df.iterrows():
        severity_scores = []
        critical_count = 0
        serious_count = 0
        minor_count = 0
        position_specific_score = 0.0
        
        # Get player position if available
        position = None
        if player_data is not None and 'Position' in player_data.columns:
            player_name = row['player_name']
            player_row = player_data[player_data['Player Name'] == player_name]
            if not player_row.empty:
                position = player_row.iloc[0]['Position']
        
        # Process each injury type
        for col in injury_columns:
            injury_type = col.replace('injury_', '')
            count = row[col]
            
            if count > 0:
                # Classify the injury
                severity = classify_injury_severity(injury_type, position)
                severity_scores.append(severity * count)
                
                # Count by severity level
                if severity == SEVERITY_LEVELS['CRITICAL']:
                    critical_count += count
                elif severity == SEVERITY_LEVELS['SERIOUS']:
                    serious_count += count
                elif severity == SEVERITY_LEVELS['MINOR']:
                    minor_count += count
                
                # Calculate position-specific impact
                if position and position in POSITION_INJURY_IMPACT:
                    for pos_injury, multiplier in POSITION_INJURY_IMPACT[position].items():
                        if pos_injury in injury_type:
                            position_specific_score += severity * count * multiplier
        
        # Update the features
        features_df.at[idx, 'injury_severity_score'] = float(sum(severity_scores))
        features_df.at[idx, 'critical_injury_count'] = critical_count
        features_df.at[idx, 'serious_injury_count'] = serious_count
        features_df.at[idx, 'minor_injury_count'] = minor_count
        features_df.at[idx, 'position_specific_injury_score'] = float(position_specific_score)
    
    # Calculate additional features
    features_df['has_critical_injury'] = (features_df['critical_injury_count'] > 0).astype(int)
    features_df['has_serious_injury'] = (features_df['serious_injury_count'] > 0).astype(int)
    
    # Avoid division by zero
    features_df['injury_severity_ratio'] = features_df.apply(
        lambda row: row['injury_severity_score'] / (row['Total_Injuries'] + 1) if row['Total_Injuries'] > 0 else 0.0, 
        axis=1
    )
    
    return features_df

def merge_injuries_with_stats(injury_features, stats_data):
    """
    Merge injury features with player statistics data.
    
    Parameters:
        injury_features (pd.DataFrame): DataFrame with injury features
        stats_data (pd.DataFrame): DataFrame with player statistics
    
    Returns:
        pd.DataFrame: Merged DataFrame with injury features and statistics
    """
    # Clean player names in injury data to match stats data
    injury_features = injury_features.copy()
    injury_features.rename(columns={'player_name': 'Player Name'}, inplace=True)
    
    # Merge the datasets
    merged_data = stats_data.merge(injury_features, on='Player Name', how='left')
    
    # Fill null injury values with 0
    injury_columns = [
        'injury_severity_score', 'critical_injury_count', 'serious_injury_count',
        'minor_injury_count', 'position_specific_injury_score', 'has_critical_injury',
        'has_serious_injury', 'injury_severity_ratio', 'Total_Injuries'
    ]
    
    for col in injury_columns:
        if col in merged_data.columns:
            merged_data[col] = merged_data[col].fillna(0)
    
    return merged_data

def process_season(year, base_dir='data'):
    """
    Process a single season's injury data and merge with stats.
    
    Parameters:
        year (int): The year to process
        base_dir (str): Base directory for data files
    
    Returns:
        pd.DataFrame: Processed data with injury features
    """
    print(f"\nProcessing year {year}...")
    
    # Define file paths
    data_dir = f"{base_dir}/{year}"
    finalized_data_path = f"{data_dir}/{year}_finalized_data.csv"
    injuries_data_path = f"{data_dir}/nfl_injuries_{year}_merged.csv"
    output_path = f"{data_dir}/{year}_final_data_with_severity_injuries.csv"
    
    # Check if files exist
    import os
    if not os.path.exists(finalized_data_path):
        print(f"Warning: {finalized_data_path} not found")
        return None
    if not os.path.exists(injuries_data_path):
        print(f"Warning: {injuries_data_path} not found")
        return None
        
    # Read the data files
    print(f"Reading {finalized_data_path}")
    finalized_data = pd.read_csv(finalized_data_path)
    print(f"Reading {injuries_data_path}")
    injuries_data = pd.read_csv(injuries_data_path)
    
    # Create injury features
    print("Creating injury severity features...")
    injury_features = create_injury_features(injuries_data, finalized_data)
    
    # Merge with stats data
    print("Merging injury features with stats data...")
    merged_data = merge_injuries_with_stats(injury_features, finalized_data)
    
    # Save the merged data
    print(f"Saving merged data to {output_path}")
    merged_data.to_csv(output_path, index=False)
    print(f"Successfully processed year {year}")
    
    # Print some stats
    total_players = len(finalized_data)
    players_with_injuries = len(merged_data[merged_data['Total_Injuries'] > 0])
    players_with_critical = len(merged_data[merged_data['has_critical_injury'] > 0])
    players_with_serious = len(merged_data[merged_data['has_serious_injury'] > 0])
    
    print(f"\nStats for {year}:")
    print(f"Total players: {total_players}")
    print(f"Players with injuries: {players_with_injuries} ({(players_with_injuries/total_players)*100:.1f}%)")
    print(f"Players with critical injuries: {players_with_critical} ({(players_with_critical/total_players)*100:.1f}%)")
    print(f"Players with serious injuries: {players_with_serious} ({(players_with_serious/total_players)*100:.1f}%)")
    print(f"Average injury severity score: {merged_data['injury_severity_score'].mean():.2f}")
    
    return merged_data

def main():
    """Process all seasons from 2018 to 2024."""
    print("Starting injury severity classification process...")
    
    # Process each year from 2018 to 2024
    years = range(2018, 2025)
    for year in years:
        try:
            process_season(year)
        except Exception as e:
            print(f"Error processing {year}: {e}")
    
    print("\nProcessing complete!")

if __name__ == "__main__":
    main() 