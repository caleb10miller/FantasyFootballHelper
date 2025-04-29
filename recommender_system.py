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

ELITE_QBS = {"Josh Allen", "Jalen Hurts", "Patrick Mahomes", "Lamar Jackson"} #implement threshold
ELITE_TES = {"Travis Kelce", "Mark Andrews", "Sam LaPorta", "George Kittle"} #implement threshold

def is_elite_qb(player_name, round_num):
    return player_name in ELITE_QBS and round_num >= 3

def is_elite_te(player_name, round_num):
    return player_name in ELITE_TES and round_num >= 3

def avoid_qb_early(row, round_num):
    """Avoid QB before round 4 unless elite."""
    return row["Position"] != "QB" or round_num >= 5 or is_elite_qb(row["Player Name"], round_num)

def avoid_te_early(row, round_num):
    """Avoid TE before round 5 unless elite."""
    return row["Position"] != "TE" or round_num >= 6 or is_elite_te(row["Player Name"], round_num)

def avoid_k_dst_early(row, round_num, num_rounds=15):
    """Delay Kicker/Defense picks until the last 2 rounds."""
    return row["Position"] not in ["K", "DST"] or round_num >= (num_rounds - 1)

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

# add value over replacement type rule (look at average points per position and then compute standard deviations ?)

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
    num_rounds=None  # Add num_rounds parameter
):
    """
    Generate fantasy player recommendations using rules + ML pipeline.
    """
    if roster_config is None:
        roster_config = {"QB": 1, "RB": 4, "WR": 5, "TE": 2, "K": 1, "DST": 1}
    
    if num_rounds is None:
        num_rounds = max(roster_config.values()) + sum(roster_config.values()) - 1

    df_players = df_players[df_players["Season"] == 2024].copy()

    exclude_cols = [
        "Player Name", "Season", "Target_PPR", "Target_Standard",
        "PPR Fantasy Points Scored", "Standard Fantasy Points Scored",
        "Delta_PPR_Fantasy_Points" if scoring_type == "Standard" else "Delta_Standard_Fantasy_Points"
    ]
    feature_cols = [col for col in df_players.columns if col not in exclude_cols]

    X = df_players[feature_cols].copy().fillna(0)
    for col in ["Team", "Position"]:
        X[col] = X[col].astype(str)

    df_players["Predicted_Points"] = pipeline.predict(X)

    eligible_players = df_players[
        df_players.apply(
            lambda row: apply_all_rules(row, round_num, team_state, roster_config),
            axis=1
        )
    ]

    top_players = eligible_players.sort_values("Predicted_Points", ascending=False).head(top_n)

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
        "position_counts": {"QB": 1, "RB": 2, "WR": 1, "TE": 0, "K": 0, "DST": 0}
    }

    # User-defined roster limits
    roster_config = {"QB": 1, "RB": 4, "WR": 5, "TE": 2, "K": 1, "DST": 1}

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
        top_n=10
    )

    print("\nTop Player Recommendations:")
    print(recommendations[["Player Name", "Position", "Team", "Predicted_Points"]])
