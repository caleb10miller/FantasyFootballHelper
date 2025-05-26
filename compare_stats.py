import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi

def compare_stats(df, players, stats, chart_type='bar', seasons=None):
    if seasons is None:
        seasons = [df['Season'].max()]
    elif isinstance(seasons, int):
        seasons = [seasons]

    df = df[df['Season'].isin(seasons) & df['Player Name'].isin(players)]
    if df.empty:
        print(f"No data found for selected players in selected seasons: {seasons}")
        return

    if chart_type == 'radar':
        for season in seasons:
            season_data = df[(df['Season'] == season) & df['Player Name'].isin(players)]
            radar_data = season_data[['Player Name'] + stats].dropna()
            if radar_data.empty:
                print(f"Not enough data for radar chart in season {season}. Skipping.")
                continue

            radar_data = radar_data.set_index('Player Name')
            radar_normalized = (radar_data - radar_data.min()) / (radar_data.max() - radar_data.min())
            categories = stats
            num_vars = len(categories)
            angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
            angles += angles[:1]

            plt.figure(figsize=(7, 7))
            ax = plt.subplot(111, polar=True)

            for player in players:
                if player not in radar_normalized.index:
                    continue
                values = radar_normalized.loc[player].tolist()
                values += values[:1]
                ax.plot(angles, values, label=player)
                ax.fill(angles, values, alpha=0.1)

            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories)
            ax.set_title(f"Radar Chart - Season {season}", size=14)
            plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
            plt.tight_layout()
            plt.show()

    elif chart_type == 'bar':
        for season in seasons:
            season_data = df[df['Season'] == season]
            for stat in stats:
                if stat not in df.columns:
                    print(f"Stat '{stat}' not found. Skipping.")
                    continue

                stat_data = season_data[['Player Name', stat]].dropna()
                if stat_data.empty:
                    print(f"No valid data for '{stat}' in season {season}. Skipping.")
                    continue

                plt.figure(figsize=(8, 5))
                plt.title(f"{stat} - Season {season}")
                sns.barplot(data=stat_data, x='Player Name', y=stat, palette='Set2')
                plt.tight_layout()
                plt.show()

        for stat in stats:
            combined_data = df[['Player Name', stat]].dropna()
            if combined_data.empty:
                continue
            combined_avg = combined_data.groupby('Player Name')[stat].mean().reset_index()

            plt.figure(figsize=(8, 5))
            plt.title(f"{stat} - Combined Average ({min(seasons)}–{max(seasons)})")
            sns.barplot(data=combined_avg, x='Player Name', y=stat, palette='Set3')
            plt.tight_layout()
            plt.show()

    elif chart_type == 'line':
        for stat in stats:
            if stat not in df.columns:
                print(f"Stat '{stat}' not found. Skipping.")
                continue

            plt.figure(figsize=(8, 5))
            for player in players:
                player_df = df[df['Player Name'] == player][['Season', stat]].dropna().sort_values('Season')
                if player_df.empty:
                    continue
                plt.plot(player_df['Season'], player_df[stat], marker='o', label=player)

            plt.title(f"{stat} - Line Chart")
            plt.xlabel("Season")
            plt.xticks(seasons)
            plt.legend()
            plt.tight_layout()
            plt.show()

    elif chart_type == 'box':
        for stat in stats:
            if stat not in df.columns:
                print(f"Stat '{stat}' not found. Skipping.")
                continue

            stat_data = df[['Player Name', stat]].dropna()
            if stat_data.empty:
                continue

            plt.figure(figsize=(8, 5))
            plt.title(f"{stat} - Box Plot ({min(seasons)}–{max(seasons)})")
            sns.boxplot(data=stat_data, x='Player Name', y=stat)
            plt.tight_layout()
            plt.show()

    else:
        print(f"Unsupported chart type: {chart_type}")


def main():
    parser = argparse.ArgumentParser(description="Compare football player stats visually.")
    parser.add_argument('--csv_path', type=str, required=True, help='Path to the player stats CSV file')
    parser.add_argument('--players', type=str, required=True, help='Comma-separated player names')
    parser.add_argument('--stats', type=str, required=True, help='Comma-separated stat columns to compare')
    parser.add_argument('--chart_type', type=str, choices=['bar', 'radar', 'line', 'box'], default='bar', help='Type of chart to plot')
    parser.add_argument('--seasons', type=int, nargs='+', required=False, help='Seasons to include (e.g., 2021 2022 2023)')

    args = parser.parse_args()
    df = pd.read_csv(args.csv_path)
    players = [p.strip() for p in args.players.split(',')]
    stats = [s.strip() for s in args.stats.split(',')]

    compare_stats(df, players, stats, args.chart_type, args.seasons)


if __name__ == '__main__':
    main()

