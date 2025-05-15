import pandas as pd
import joblib

# -----------------------------
# Load Combined Pipeline & Model
# -----------------------------

def load_pipeline(pipeline_path):
    """Load the full MLP pipeline (preprocessing + model)."""
    return joblib.load(pipeline_path)

# -----------------------------
# Configurable Draft Rules
# -----------------------------

def is_elite_qb(player_name, round_num, predicted_points):
    """Determine if a QB is elite based on predicted points."""
    QB_ELITE_THRESHOLD = 200  # QBs projected for 200+ points
    return predicted_points >= QB_ELITE_THRESHOLD and round_num >= 3

def is_elite_te(player_name, round_num, predicted_points):
    """Determine if a TE is elite based on predicted points."""
    TE_ELITE_THRESHOLD = 170  # TEs projected for 170+ points
    return predicted_points >= TE_ELITE_THRESHOLD and round_num >= 3

def avoid_qb_early(row, round_num):
    """Avoid QB before round 4 unless elite."""
    if row["Position"] != "QB":
        return True
    return round_num >= 5 or is_elite_qb(row["Player Name"], round_num, row["Predicted_Points"])

def avoid_te_early(row, round_num):
    """Avoid TE before round 5 unless elite."""
    if row["Position"] != "TE":
        return True
    return round_num >= 6 or is_elite_te(row["Player Name"], round_num, row["Predicted_Points"])

def avoid_k_dst_early(row, round_num, num_rounds=15):
    """Delay Kicker/Defense picks until the last 2 rounds."""
    if row["Position"] in ["K", "DEF"]:
        return round_num >= (num_rounds - 1)
    return True

def respect_roster_limits(row, team_state, roster_config):
    """Avoid recommending players at positions already at or above max."""
    position = row["Position"]
    return team_state["position_counts"].get(position, 0) < roster_config.get(position, 99)

def prioritize_needs(row, team_state, roster_config):
    """Prefer positions not yet filled to required starting level."""
    position = row["Position"]
    filled = team_state["position_counts"].get(position, 0)
    max_allowed = roster_config.get(position, 99)
    return filled < max_allowed

def get_replacement_levels(df_players, league_size=12):
    """Calculate replacement level points for each position."""
    replacement_ranks = {
        'QB': league_size + 4,    # 16th best QB (deeper backup level)
        'RB': league_size * 2,    # 24th best RB (2 per team)
        'WR': league_size * 2,    # 24th best WR (2 per team)
        'TE': league_size + 4,    # 16th best TE (backup level)
        'K': league_size,         # 12th best K
        'DEF': league_size        # 12th best DEF
    }
    
    replacement_levels = {}
    
    for position in replacement_ranks:
        position_players = df_players[df_players['Position'] == position].copy()
        if not position_players.empty:
            # Sort by predicted points and get replacement level
            position_players = position_players.sort_values('Predicted_Points', ascending=False)
            rank = int(replacement_ranks[position])
            if len(position_players) >= rank:
                replacement_levels[position] = position_players.iloc[rank-1]['Predicted_Points']
            else:
                replacement_levels[position] = position_players['Predicted_Points'].min()
                
    return replacement_levels

def calculate_vor_and_score(df_players, replacement_levels, vor_weight=0.7):
    """Calculate Value Over Replacement and combined score for each player."""
    df = df_players.copy()
    
    # Position value multipliers to account for positional importance
    position_multipliers = {
        'QB': 1.2,   # Higher premium for QBs
        'RB': 1.15,  # Slightly reduced RB premium
        'WR': 1.15,   # Baseline value
        'TE': 1.0,   # Slightly reduced
        'K': 0.3,    # Significantly reduced
        'DEF': 0.3   # Significantly reduced
    }
    
    # Calculate VOR with position multipliers
    df['VOR'] = df.apply(
        lambda row: (row['Predicted_Points'] - replacement_levels.get(row['Position'], 0)) * 
                   position_multipliers.get(row['Position'], 1.0),
        axis=1
    )
    
    # Add a bonus for elite QBs (over 225 points)
    df['VOR'] = df.apply(
        lambda row: row['VOR'] * 1.1 if row['Position'] == 'QB' and row['Predicted_Points'] > 225 else row['VOR'],
        axis=1
    )
    
    # Normalize VOR and Predicted Points to 0-1 scale to make them comparable
    df['VOR_Normalized'] = (df['VOR'] - df['VOR'].min()) / (df['VOR'].max() - df['VOR'].min())
    df['Points_Normalized'] = (df['Predicted_Points'] - df['Predicted_Points'].min()) / (
        df['Predicted_Points'].max() - df['Predicted_Points'].min()
    )
    
    # Calculate combined score
    df['Overall_Score'] = vor_weight * df['VOR_Normalized'] + (1 - vor_weight) * df['Points_Normalized']
    
    return df

def apply_all_rules(row, round_num, team_state, roster_config):
    num_rounds = max(roster_config.values()) + sum(roster_config.values()) - 1  # Estimate total rounds from roster config
    return (
        avoid_qb_early(row, round_num)
        and avoid_te_early(row, round_num)
        and avoid_k_dst_early(row, round_num, num_rounds)
        and respect_roster_limits(row, team_state, roster_config)
        and prioritize_needs(row, team_state, roster_config)
    )

# -----------------------------
# Recommendation Engine
# -----------------------------

def recommend_players(
    df_players,
    round_num,
    team_state,
    pipeline,
    scoring_type="PPR",
    roster_config=None,
    top_n=5,
    league_size=12,
    vor_weight=0.7,
    num_rounds=None
):
    """
    Generate fantasy player recommendations using rules + ML pipeline.
    """
    if roster_config is None:
        roster_config = {"QB": 1, "RB": 4, "WR": 5, "TE": 2, "K": 1, "DEF": 1}
    
    if num_rounds is None:
        num_rounds = max(roster_config.values()) + sum(roster_config.values()) - 1

    df_players = df_players[df_players["Season"] == 2024].copy()

    # Ensure scoring_type has a valid value
    scoring_type = scoring_type or "PPR"
    if scoring_type not in ["PPR", "Standard"]:
        scoring_type = "PPR"

    exclude_cols = [
        "Player Name", "Season", "Target_PPR", "Target_Standard",
        "PPR Fantasy Points Scored", "Standard Fantasy Points Scored",
        "Delta_PPR_Fantasy_Points" if scoring_type == "Standard" else "Delta_Standard_Fantasy_Points"
    ]
    feature_cols = [col for col in df_players.columns if col not in exclude_cols]

    X = df_players[feature_cols].copy().fillna(0)
    for col in ["Team", "Position"]:
        X[col] = X[col].astype(str)

    try:
        df_players["Predicted_Points"] = pipeline.predict(X)
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        raise
    
    # Calculate VOR and Overall Score
    replacement_levels = get_replacement_levels(df_players, league_size)
    df_players = calculate_vor_and_score(df_players, replacement_levels, vor_weight)

    eligible_players = df_players[
        df_players.apply(
            lambda row: apply_all_rules(row, round_num, team_state, roster_config),
            axis=1
        )
    ]

    # Sort by Overall Score
    top_players = eligible_players.sort_values("Overall_Score", ascending=False).head(top_n)

    # Clean up the output by dropping normalized columns
    top_players = top_players.drop(['VOR_Normalized', 'Points_Normalized'], axis=1)

    return top_players.reset_index(drop=True)

# -----------------------------
# Example Usage
# -----------------------------

if __name__ == "__main__":
    # === CONFIGURATION ===
    scoring_type = "PPR"
    round_num = 1

    # Current state of user's team
    team_state = {
        "filled_positions": ["QB"],
        "drafted_players": ["Amon-Ra St. Brown"],
        "position_counts": {"QB": 1, "RB": 2, "WR": 1, "TE": 0, "K": 0, "DEF": 0}
    }

    # User-defined roster limits
    roster_config = {"QB": 1, "RB": 4, "WR": 5, "TE": 2, "K": 1, "DEF": 1}

    # === LOAD DATA & MODEL ===
    player_data_path = "data/final_data/nfl_stats_long_format_with_context_filtered.csv"
    pipeline_path = f"mlp_regression/joblib_files/mlp_pipeline_{scoring_type}.pkl"

    df_players = pd.read_csv(player_data_path)
    pipeline = load_pipeline(pipeline_path)

    # === GET RECOMMENDATIONS ===
    recommendations = recommend_players(
        df_players=df_players,
        round_num=round_num,
        team_state=team_state,
        pipeline=pipeline,
        scoring_type=scoring_type,
        roster_config=roster_config,
        top_n=10,
        vor_weight=0.7
    )

    print("\nTop Player Recommendations:")
    print(recommendations[["Player Name", "Position", "Team", "Predicted_Points", "VOR", "Overall_Score"]].round(2))
