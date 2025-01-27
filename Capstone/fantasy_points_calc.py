import pandas as pd

def calculate_fantasy_points(input_file='nfl_2022_final_data.csv', output_file='nfl_2022_fantasy_points.csv'):
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    # Convert blank values to 0 for calculations
    df.fillna(0, inplace=True)
    
    # Fantasy points calculation for different positions
    df['Fantasy Points Scored'] = 0
    
    # Calculate fantasy points for QB, WR, RB, TE
    df.loc[df['Position'].isin(['QB', 'WR', 'RB', 'TE']), 'Fantasy Points Scored'] = (
        df['Passing Yards'] * 0.04 +
        df['Passing Touchdowns'] * 4 +
        df['Interceptions Thrown'] * -2 +
        df['Rushing Yards'] * 0.1 +
        df['Rushing Touchdowns'] * 6 +
        df['Receptions'] * 1 +
        df['Receiving Yards'] * 0.1 +
        df['Receiving Touchdowns'] * 6 +
        df['Fumbles Lost'] * -2 +
        df['XP2'] * 2
    )
    
    # Calculate fantasy points for Kickers (K)
    df.loc[df['Position'] == 'K', 'Fantasy Points Scored'] = (
        df['Field Goals Made 0-19'] * 3 +
        df['Field Goals Made 20-29'] * 3 +
        df['Field Goals Made 30-39'] * 3 +
        df['Field Goals Made 40-49'] * 4 +
        df['Field Goals Made 50+'] * 5 +
        df['Extra Points Made'] * 1 +
        (df['Field Goals Attempted'] - df['Field Goals Made']) * -1 +
        (df['Extra Points Attempted'] - df['Extra Points Made']) * -2
    )
    
    # Calculate fantasy points for Defense (DEF)
    df.loc[df['Position'] == 'DEF', 'Fantasy Points Scored'] = (
        df['ST_Sacks'] * 1 +
        df['ST_Interceptions'] * 2 +
        df['ST_Fumble Recoveries'] * 2 +
        df['ST_Safeties'] * 2 +
        df['ST_Special Teams Touchdowns'] * 6 +
        df['Fantasy Points From Points'] * 1
    )
    
    # Save the updated CSV file
    df.to_csv(output_file, index=False)
    print(f"Fantasy points calculated and saved to {output_file}")

if __name__ == '__main__':
    calculate_fantasy_points()
