import pandas as pd

def calculate_fantasy_points(input_file='nfl_2022_final_data.csv', output_file='nfl_2022_fantasy_points.csv'):
    df = pd.read_csv(input_file)
    
    df.fillna(0, inplace=True)
    
    df['PPR Fantasy Points Scored'] = 0
    df['Standard Fantasy Points Scored'] = 0
    
    df.loc[df['Position'].isin(['QB', 'WR', 'RB', 'TE']), 'PPR Fantasy Points Scored'] = (
        df['Passing Yards'] * 0.04 +
        df['Passing Touchdowns'] * 4 +
        df['Interceptions Thrown'] * -2 +
        df['Rushing Yards'] * 0.1 +
        df['Rushing Touchdowns'] * 6 +
        df['Receptions'] * 1 +  # PPR only
        df['Receiving Yards'] * 0.1 +
        df['Receiving Touchdowns'] * 6 +
        df['Fumbles Lost'] * -2 +
        df['XP2'] * 2
    )
    
    df.loc[df['Position'].isin(['QB', 'WR', 'RB', 'TE']), 'Standard Fantasy Points Scored'] = (
        df['Passing Yards'] * 0.04 +
        df['Passing Touchdowns'] * 4 +
        df['Interceptions Thrown'] * -2 +
        df['Rushing Yards'] * 0.1 +
        df['Rushing Touchdowns'] * 6 +
        df['Receiving Yards'] * 0.1 +
        df['Receiving Touchdowns'] * 6 +
        df['Fumbles Lost'] * -2 +
        df['XP2'] * 2
    )
    
    # Fantasy Points for Kickers (K) - Same for both formats
    df.loc[df['Position'] == 'K', ['PPR Fantasy Points Scored', 'Standard Fantasy Points Scored']] = (
        df['Field Goals Made 0-19'] * 3 +
        df['Field Goals Made 20-29'] * 3 +
        df['Field Goals Made 30-39'] * 3 +
        df['Field Goals Made 40-49'] * 4 +
        df['Field Goals Made 50+'] * 5 +
        df['Extra Points Made'] * 1 +
        (df['Field Goals Attempted'] - df['Field Goals Made']) * -1 +
        (df['Extra Points Attempted'] - df['Extra Points Made']) * -2
    )
    
    df.loc[df['Position'] == 'DEF', ['PPR Fantasy Points Scored', 'Standard Fantasy Points Scored']] = (
        df['ST_Sacks'] * 1 +
        df['ST_Interceptions'] * 2 +
        df['ST_Fumble Recoveries'] * 2 +
        df['ST_Safeties'] * 2 +
        df['ST_Special Teams Touchdowns'] * 6 +
        df['Fantasy Points From Points'] * 1
    )
    
    df.to_csv(output_file, index=False)
    print(f"Fantasy points calculated and saved to {output_file}")

if __name__ == '__main__':
    calculate_fantasy_points()
