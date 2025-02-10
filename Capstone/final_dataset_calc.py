from MainScraper import main
from fantasy_points_calc import calculate_fantasy_points
import pandas as pd
import os

def pull_all_years(years = [2022, 2023, 2024]):
    for year in years:
        main(year=year)
        calculate_fantasy_points(input_file = f"Capstone/data/{year}/nfl_{year}_final_data.csv", output_file = f"Capstone/data/{year}/nfl_{year}_fantasy_points.csv")
        main_df = pd.read_csv(f"Capstone/data/{year}/nfl_{year}_final_data.csv")
        fantasy_df = pd.read_csv(f"Capstone/data/{year}/nfl_{year}_fantasy_points.csv")
        fantasy_df = fantasy_df[['Player Name', 'PPR Fantasy Points Scored', 'Standard Fantasy Points Scored']]
        merged_df = pd.merge(main_df, fantasy_df, on='Player Name', how='outer')
        merged_df.to_csv(f"Capstone/data/{year}/{year}_finalized_data.csv", index=False)

def merge_datasets(years = [2022, 2023, 2024]):
    merged_df = None
    output_dir = "Capstone/data/final_data"  
    output_file = os.path.join(output_dir, "nfl_merged_wide_format.csv")

    os.makedirs(output_dir, exist_ok=True)

    for year in years:
        file_path = f"Capstone/data/{year}/{year}_finalized_data.csv"
        
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue
            
        df = pd.read_csv(file_path)

        rename_dict = {col: f"{year} {col}" for col in df.columns if col not in ["Player Name", "Season", "Player ID"]}
        df.rename(columns=rename_dict, inplace=True)
        
        df.drop(columns=["Season"], inplace=True)

        if merged_df is None:
            merged_df = df
        else:
            merged_df = pd.merge(merged_df, df, on=["Player Name"], how="outer")

    merged_df[f"{year} PPR Fantasy Points Scored"] = merged_df[f"{year} PPR Fantasy Points Scored"].round(2)
    merged_df[f"{year} Standard Fantasy Points Scored"] = merged_df[f"{year} Standard Fantasy Points Scored"].round(2)

    if merged_df is not None:
        merged_df.to_csv(output_file, index=False)
        print(f"Data merged and saved to {output_file}")
    else:
        print("No data to merge")

if __name__ == '__main__':
    pull_all_years([2022, 2023, 2024])
    merge_datasets([2022, 2023, 2024])