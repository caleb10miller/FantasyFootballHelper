import os
import pandas as pd

years = range(2018, 2025) 
combined_df = []

valid_positions = {"QB", "TE", "RB", "WR", "DEF", "K"}

for year in years:
    file_path = f"data/{year}/{year}_finalized_data.csv"
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        continue
        
    df = pd.read_csv(file_path)
    df['year'] = year  
    df = df[df['Position'].isin(valid_positions)]
    
    combined_df.append(df)

final_df = pd.concat(combined_df, ignore_index=True)

final_df.to_csv("data/final_data/nfl_stats_long_format.csv", index=False)
