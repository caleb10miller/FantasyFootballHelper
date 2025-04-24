import dash
from dash import dcc, html, Input, Output, State, ctx
import pandas as pd
import joblib

# Load model pipeline
pipeline_path = "mlp_regression/joblib_files/mlp_pipeline_PPR.pkl"
pipeline = joblib.load(pipeline_path)

# Load player data
df = pd.read_csv("data/final_data/nfl_stats_long_format_with_context_filtered.csv")
df = df[df["Season"] == 2024].copy()

# Prediction logic
def recommend_players(df_players, round_num, team_state, pipeline, scoring_type, roster_config, top_n=5):
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

    def is_elite_qb(name):
        return name in {"Josh Allen", "Jalen Hurts", "Patrick Mahomes", "Lamar Jackson"}

    def is_elite_te(name):
        return name in {"Travis Kelce", "Mark Andrews", "Sam LaPorta", "George Kittle"}

    def passes_rules(row):
        pos = row["Position"]
        name = row["Player Name"]
        count = team_state.get("position_counts", {}).get(pos, 0)
        max_count = roster_config.get(pos, 99)

        if pos == "QB" and round_num < 4 and not is_elite_qb(name):
            return False
        if pos == "TE" and round_num < 5 and not is_elite_te(name):
            return False
        if pos in ["K", "DST"] and round_num < 13:
            return False
        if count >= max_count:
            return False
        return True

    eligible = df_players[df_players.apply(passes_rules, axis=1)]
    return eligible.sort_values("Predicted_Points", ascending=False).head(top_n)

# Dash setup
app = dash.Dash(__name__)
app.title = "Fantasy Draft Assistant"

positions = ["QB", "RB", "WR", "TE", "K", "DST"]

app.layout = html.Div([
    html.H2("Fantasy Football Draft Assistant"),

    html.Div([
        html.Label("Round Number"),
        dcc.Input(id="round-num", type="number", value=1, min=1)
    ]),

    html.Div([
        html.Label("Scoring Type"),
        dcc.Dropdown(["PPR", "Standard"], value="PPR", id="scoring-type")
    ]),

    html.Div([
        html.Label("Position Counts"),
        html.Div([
            html.Div([
                html.Label(pos),
                dcc.Input(id=f"count-{pos}", type="number", value=0, min=0)
            ]) for pos in positions
        ], style={"display": "flex", "gap": "10px"})
    ]),

    html.Div([
        html.Label("Roster Max Config"),
        html.Div([
            html.Div([
                html.Label(pos),
                dcc.Input(id=f"max-{pos}", type="number", value=5 if pos in ["RB", "WR"] else 1, min=1)
            ]) for pos in positions
        ], style={"display": "flex", "gap": "10px"})
    ]),

    html.Button("Get Recommendations", id="submit-button"),

    html.Div(id="recommendation-output")
])

@app.callback(
    Output("recommendation-output", "children"),
    Input("submit-button", "n_clicks"),
    [State("round-num", "value"),
     State("scoring-type", "value")] +
    [State(f"count-{pos}", "value") for pos in positions] +
    [State(f"max-{pos}", "value") for pos in positions]
)
def update_recommendations(n_clicks, round_num, scoring_type, *args):
    if not n_clicks:
        return ""

    position_counts = dict(zip(positions, args[:len(positions)]))
    roster_config = dict(zip(positions, args[len(positions):]))

    team_state = {"position_counts": position_counts}
    recs = recommend_players(df.copy(), round_num, team_state, pipeline, scoring_type, roster_config)

    return html.Table([
        html.Thead(html.Tr([html.Th(c) for c in ["Player Name", "Position", "Team", "Predicted_Points"]])),
        html.Tbody([
            html.Tr([
                html.Td(row[col]) for col in ["Player Name", "Position", "Team", "Predicted_Points"]
            ]) for _, row in recs.iterrows()
        ])
    ])

if __name__ == '__main__':
    app.run_server(debug=True)
