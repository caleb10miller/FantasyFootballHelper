#!/usr/bin/env python3
"""
Run Injury Severity Pipeline

This script runs the entire injury severity classification pipeline from start to finish.
It processes each season's injury data, applies the severity classification, and combines
the results into a single file for model training.
"""

import os
import sys
import time
from datetime import datetime

def run_pipeline():
    """
    Run the entire injury severity classification pipeline.
    """
    start_time = time.time()
    print(f"Starting injury severity classification pipeline at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Step 1: Run the injury severity classifier
    print("\n" + "="*80)
    print("STEP 1: Running injury severity classifier")
    print("="*80)
    
    try:
        import injury_severity_classifier
        injury_severity_classifier.main()
    except Exception as e:
        print(f"Error running injury severity classifier: {e}")
        return False
    
    # Step 2: Combine all seasons
    print("\n" + "="*80)
    print("STEP 2: Combining all seasons")
    print("="*80)
    
    try:
        import combine_severity_injuries
        combine_severity_injuries.combine_all_seasons()
    except Exception as e:
        print(f"Error combining seasons: {e}")
        return False
    
    # Step 3: Verify the output
    print("\n" + "="*80)
    print("STEP 3: Verifying output")
    print("="*80)
    
    output_file = 'data/final_data/nfl_stats_with_severity_injuries.csv'
    if not os.path.exists(output_file):
        print(f"Error: Output file {output_file} not found")
        return False
    
    try:
        import pandas as pd
        df = pd.read_csv(output_file)
        print(f"Output file verified: {output_file}")
        print(f"Total records: {len(df)}")
        print(f"Total players: {df['Player Name'].nunique()}")
        print(f"Seasons: {df['Season'].unique()}")
        print(f"New injury features: {', '.join([col for col in df.columns if 'injury' in col.lower()])}")
    except Exception as e:
        print(f"Error verifying output: {e}")
        return False
    
    # Pipeline completed successfully
    end_time = time.time()
    duration = end_time - start_time
    print(f"\nPipeline completed successfully in {duration:.2f} seconds")
    print(f"Output file: {output_file}")
    
    return True

if __name__ == "__main__":
    success = run_pipeline()
    sys.exit(0 if success else 1) 