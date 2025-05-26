import dash
from dash import html, dcc, Input, Output, State, callback
import dash_bootstrap_components as dbc
import pandas as pd
import io
import base64
from matplotlib import pyplot as plt
from recommender_system import recommend_players, load_pipeline
from lightgbm_regression.lightgbm_regressor import LightGBMRegressor
from compare_stats import compare_stats

# Initialize the Dash app with dark theme
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    suppress_callback_exceptions=True
)

# Custom dark theme colors
DARK_THEME = {
    'background': '#222',
    'text': '#fff',
    'primary': '#375a7f',
    'secondary': '#444',
    'success': '#00bc8c',
    'info': '#3498DB',
    'warning': '#F39C12',
    'danger': '#E74C3C'
}

# Custom styles for components
table_style = {
    'backgroundColor': DARK_THEME['background'],
    'color': DARK_THEME['text']
}

card_style = {
    'backgroundColor': DARK_THEME['background'],
    'color': DARK_THEME['text'],
    'borderColor': DARK_THEME['secondary']
}

def get_current_drafter(round_num, pick_num, num_teams):
    """
    Determine which team is currently drafting based on round and pick number.
    Handles serpentine draft order.
    """
    if round_num % 2 == 1:  # Odd rounds go 1 to num_teams
        return pick_num
    else:  # Even rounds go num_teams to 1
        return num_teams - pick_num + 1

# Default roster configuration
DEFAULT_ROSTER_CONFIG = {
    "QB": 2,
    "RB": 5,
    "WR": 5,
    "TE": 1,
    "K": 1,
    "DST": 1
}

def calculate_roster_limits(num_rounds):
    """
    Calculate roster limits based on number of rounds.
    Ensures the total matches the number of rounds while maintaining reasonable proportions.
    """
    if not num_rounds:
        return DEFAULT_ROSTER_CONFIG
    
    # Base proportions (out of total roster spots)
    proportions = {
        "QB": 0.07,  # ~1/15
        "RB": 0.27,  # ~4/15
        "WR": 0.33,  # ~5/15
        "TE": 0.13,  # ~2/15
        "K": 0.07,   # ~1/15
        "DST": 0.07  # ~1/15
    }
    
    # Calculate initial values
    limits = {pos: max(1, round(num_rounds * prop)) for pos, prop in proportions.items()}
    
    # Adjust to match total rounds
    total = sum(limits.values())
    while total != num_rounds:
        if total < num_rounds:
            # Add to RB or WR alternately if we need more spots
            if limits["RB"] <= limits["WR"]:
                limits["RB"] += 1
            else:
                limits["WR"] += 1
        else:
            # Remove from RB or WR if we have too many
            if limits["RB"] > limits["WR"]:
                limits["RB"] -= 1
            else:
                limits["WR"] -= 1
        total = sum(limits.values())
    
    return limits

# Load the pipelines and player data
pipelines = {
    "PPR": load_pipeline("lightgbm_regression/joblib_files/lightgbm_regression_pipeline_PPR_20250514_214644.pkl"),
    "Standard": load_pipeline("stacked_model/joblib_files/stacked_model_pipeline_Standard_20250514_214726.pkl")
}
df_players = pd.read_csv("data/final_data/nfl_stats_long_format_with_context_filtered_with_experience.csv")
# Filter to only 2024 players and create initial available players list
df_players = df_players[df_players['Season'] == 2024].copy()
AVAILABLE_PLAYERS = set(df_players['Player Name'].unique())

# Layout components
def create_roster_config_inputs():
    return dbc.Card([
        dbc.CardHeader("Roster Configuration"),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    dbc.Label("QB"),
                    dbc.Input(id="qb-limit", type="number", value=DEFAULT_ROSTER_CONFIG["QB"], min=1)
                ], width=2),
                dbc.Col([
                    dbc.Label("RB"),
                    dbc.Input(id="rb-limit", type="number", value=DEFAULT_ROSTER_CONFIG["RB"], min=1)
                ], width=2),
                dbc.Col([
                    dbc.Label("WR"),
                    dbc.Input(id="wr-limit", type="number", value=DEFAULT_ROSTER_CONFIG["WR"], min=1)
                ], width=2),
                dbc.Col([
                    dbc.Label("TE"),
                    dbc.Input(id="te-limit", type="number", value=DEFAULT_ROSTER_CONFIG["TE"], min=1)
                ], width=2),
                dbc.Col([
                    dbc.Label("K"),
                    dbc.Input(id="k-limit", type="number", value=DEFAULT_ROSTER_CONFIG["K"], min=1)
                ], width=2),
                dbc.Col([
                    dbc.Label("DST"),
                    dbc.Input(id="dst-limit", type="number", value=DEFAULT_ROSTER_CONFIG["DST"], min=1)
                ], width=2),
            ])
        ])
    ])

def create_draft_setup():
    return html.Div(
        dbc.Card([
            dbc.CardHeader("Draft Setup"),
            dbc.CardBody([
                # Draft Settings Section
                html.H5("Draft Settings", className="mb-3"),
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Scoring Type"),
                        dbc.Select(
                            id="scoring-type",
                            options=[
                                {"label": "PPR", "value": "PPR"},
                                {"label": "Standard", "value": "Standard"}
                            ],
                            value="PPR"
                        )
                    ], width=3),
                    dbc.Col([
                        dbc.Label("Number of Teams"),
                        dbc.Input(id="num-teams", type="number", value=12, min=2)
                    ], width=3),
                    dbc.Col([
                        dbc.Label("Number of Rounds"),
                        dbc.Input(id="num-rounds", type="number", value=15, min=1)
                    ], width=3),
                    dbc.Col([
                        dbc.Label("Your Draft Position"),
                        dbc.Input(id="draft-position", type="number", value=1, min=1)
                    ], width=3),
                ]),
                html.Hr(),
                # Roster Configuration Section
                html.H5("Roster Configuration", className="mb-3"),
                dbc.Row([
                    dbc.Col([
                        dbc.Label("QB"),
                        dbc.Input(id="qb-limit", type="number", value=DEFAULT_ROSTER_CONFIG["QB"], min=1)
                    ], width=2),
                    dbc.Col([
                        dbc.Label("RB"),
                        dbc.Input(id="rb-limit", type="number", value=DEFAULT_ROSTER_CONFIG["RB"], min=1)
                    ], width=2),
                    dbc.Col([
                        dbc.Label("WR"),
                        dbc.Input(id="wr-limit", type="number", value=DEFAULT_ROSTER_CONFIG["WR"], min=1)
                    ], width=2),
                    dbc.Col([
                        dbc.Label("TE"),
                        dbc.Input(id="te-limit", type="number", value=DEFAULT_ROSTER_CONFIG["TE"], min=1)
                    ], width=2),
                    dbc.Col([
                        dbc.Label("K"),
                        dbc.Input(id="k-limit", type="number", value=DEFAULT_ROSTER_CONFIG["K"], min=1)
                    ], width=2),
                    dbc.Col([
                        dbc.Label("DST"),
                        dbc.Input(id="dst-limit", type="number", value=DEFAULT_ROSTER_CONFIG["DST"], min=1)
                    ], width=2),
                ]),
                dbc.Button("Start Draft", id="start-draft", color="primary", className="mt-4"),
                html.Div(id="setup-error", className="text-danger mt-2")
            ])
        ]),
        id="draft-setup-container"
    )

def create_draft_board():
    return dbc.Card([
        dbc.CardHeader("Draft Board", style={'backgroundColor': DARK_THEME['secondary']}),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    dbc.Label("Current Round", className="text-light"),
                    html.Div(id="current-round", className="h4 text-info")
                ], width=3),
                dbc.Col([
                    dbc.Label("Current Pick", className="text-light"),
                    html.Div(id="current-pick", className="h4 text-info")
                ], width=3),
                dbc.Col([
                    dbc.Label("Current Drafter", className="text-light"),
                    html.Div(id="current-drafter", className="h4 text-info")
                ], width=3),
                dbc.Col([
                    dbc.Label("Picks Until Your Turn", className="text-light"),
                    html.Div(id="picks-until-turn", className="h4 text-warning")
                ], width=3),
            ]),
            dbc.Row([
                dbc.Col([
                    dbc.Label("Select Player", className="text-light"),
                    dcc.Dropdown(
                        id="player-dropdown",
                        multi=False,
                        style={
                            'backgroundColor': DARK_THEME['background'],
                            'color': 'white',
                        },
                        className="player-dropdown",
                        optionHeight=35,
                    ),
                    dbc.Button("Submit Pick", id="submit-pick", color="primary", className="mt-2"),
                    html.Div(id="pick-error-message", className="text-danger mt-2")
                ], width=12)
            ]),
            html.Hr(style={'borderColor': DARK_THEME['secondary']}),
            # Recommendations section
            dbc.Row([
                dbc.Col([
                    html.H5("Recommendations", className="mb-3 text-light"),
                    html.Div(id="recommendations-table")
                ], width=12)
            ]),
            html.Hr(style={'borderColor': DARK_THEME['secondary']}),
            # Add New Player section with collapse
            dbc.Row([
                dbc.Col([
                    dbc.Button(
                        "Toggle Add New Player",
                        id="toggle-add-player",
                        color="secondary",
                        className="mb-3"
                    ),
                    dbc.Collapse(
                        dbc.Card([
                            dbc.CardBody([
                                html.H5("Add New Player", className="text-light"),
                                dbc.Row([
                                    dbc.Col([
                                        dbc.Label("Player Name", className="text-light"),
                                        dbc.Input(
                                            id="new-player-name",
                                            type="text",
                                            placeholder="Enter player name",
                                            style={
                                                'backgroundColor': DARK_THEME['background'],
                                                'color': DARK_THEME['text'],
                                                'borderColor': 'white'
                                            }
                                        )
                                    ], width=4),
                                    dbc.Col([
                                        dbc.Label("Position", className="text-light"),
                                        dcc.Dropdown(
                                            id="new-player-position",
                                            options=[{"label": pos, "value": pos} for pos in ["QB", "RB", "WR", "TE", "K", "DST"]],
                                            placeholder="Select position",
                                            style={
                                                'backgroundColor': DARK_THEME['background'],
                                                'color': 'white',
                                            },
                                            className="dash-dropdown"
                                        )
                                    ], width=4),
                                    dbc.Col([
                                        dbc.Label("Team", className="text-light"),
                                        dbc.Input(
                                            id="new-player-team",
                                            type="text",
                                            placeholder="Enter team (e.g., SF)",
                                            style={
                                                'backgroundColor': DARK_THEME['background'],
                                                'color': DARK_THEME['text'],
                                                'borderColor': 'white'
                                            }
                                        )
                                    ], width=4)
                                ]),
                                dbc.Button("Add Player", id="add-player-button", color="secondary", className="mt-2")
                            ])
                        ], style=card_style),
                        id="add-player-collapse",
                        is_open=False
                    )
                ], width=12)
            ]),
            html.Hr(style={'borderColor': DARK_THEME['secondary']}),
            # Draft History section with collapse
            dbc.Row([
                dbc.Col([
                    dbc.Button(
                        "Toggle Draft History",
                        id="toggle-draft-history",
                        color="secondary",
                        className="mb-3"
                    ),
                    dbc.Collapse(
                        dbc.Card([
                            dbc.CardBody([
                                html.H5("Draft History", className="text-light"),
                                html.Div(id="draft-history-table")
                            ])
                        ], style=card_style),
                        id="draft-history-collapse",
                        is_open=False
                    )
                ], width=12)
            ])
        ], style={'backgroundColor': DARK_THEME['background']})
    ], style=card_style)

def create_recommendations():
    return dbc.Card([
        dbc.CardHeader("Recommendations"),
        dbc.CardBody([
            html.Div(id="recommendations-table")
        ])
    ])

# App layout
app.layout = dbc.Container([
    html.H1("Fantasy Football Draft Assistant", className="text-center my-4"),
    
    # Tabs for navigation
    dcc.Tabs(id="main-tabs", value="draft", children=[
        dcc.Tab(label="Draft Board", value="draft", style={'color': 'black'}, selected_style={'color': 'black'}),
        dcc.Tab(label="Team Overview", value="teams", style={'color': 'black'}, selected_style={'color': 'black'}),
        dcc.Tab(label="Visualizations", value="viz", style={'color': 'black'}, selected_style={'color': 'black'}), #NEW VIS FUNCTION
    ], style={'color': 'black'}),    
    # Tab content will be injected here
    html.Div(id="tab-content"),

    # Store for draft state
    dcc.Store(id='draft-state', data={
        'round': 1,
        'pick': 1,
        'draft_position': 1,
        'num_teams': 12,
        'num_rounds': 15,
        'team_state': {
            'filled_positions': [],
            'drafted_players': [],
            'position_counts': {pos: 0 for pos in DEFAULT_ROSTER_CONFIG.keys()}
        },
        'other_teams_picks': {},
        'pick_history': [],
        'available_players': list(AVAILABLE_PLAYERS),
        'error_message': None,
        'setup_collapsed': False  # Add this to track setup collapse state
    })
], fluid=True)

def create_visualizations_tab():
    # Build stat list dynamically
    numeric_cols = (
        df_players.select_dtypes(include="number")
                  .columns
                  .drop(["Season", "Age"], errors="ignore")
    )

    return dbc.Card([
        dbc.CardHeader("Player Stat Visualiser", className="text-center text-white"),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    dbc.Label("Players"),
                    dcc.Dropdown(
                        id="viz-player-dropdown",
                        options=[{"label": n, "value": n}
                                 for n in sorted(df_players["Player Name"].unique())],
                        multi=True
                    )
                ], width=6),
                dbc.Col([
                    dbc.Label("Statistics"),
                    dcc.Dropdown(
                        id="viz-stat-dropdown",
                        options=[{"label": c, "value": c} for c in numeric_cols],
                        multi=True,
                        style={"color": "black", "backgroundColor": "white"}
                    )
                ], width=6),
            ], className="mb-3"),
            dbc.Row([
                dbc.Col([
                    dbc.Label("Chart Type"),
                    dcc.Dropdown(
                        id="viz-chart-type",
                        options=[
                            {"label": "Bar", "value": "bar"},
                            {"label": "Radar", "value": "radar"},
                            {"label": "Line", "value": "line"},
                            {"label": "Box", "value": "box"}
                        ],
                        value="bar",
                        clearable=False
                    )
                ], width=4),
                dbc.Col([
                    dbc.Label("Season(s) (comma‑sep)"),
                    dbc.Input(id="viz-seasons", type="text", placeholder="2024, 2023")
                ], width=4),
                dbc.Col([
                    dbc.Button("Show Chart", id="viz-show-btn", color="info", className="mt-4")
                ], width=4)
            ]),
            html.Hr(),
            html.Div(id="viz-output")
        ])
    ], style={"backgroundColor": "#2a2a2a", "color": "white"})

# Callbacks
@app.callback(
    [Output('draft-state', 'data'),
     Output('draft-setup-container', 'style'),
     Output('setup-error', 'children')],
    Input('start-draft', 'n_clicks'),
    [State('num-teams', 'value'),
     State('num-rounds', 'value'),
     State('draft-position', 'value'),
     State('scoring-type', 'value'),
     State('draft-state', 'data')],
    prevent_initial_call=True
)
def start_draft(n_clicks, num_teams, num_rounds, draft_position, scoring_type, draft_state):
    if not n_clicks:
        return draft_state, {"display": "block" if not draft_state.get('setup_collapsed', False) else "none"}, ""
    # Validate inputs
    if not all([num_teams, num_rounds, draft_position]):
        return draft_state, {"display": "block"}, "Please fill in all required fields"
    if draft_position > num_teams:
        return draft_state, {"display": "block"}, f"Draft position cannot be greater than number of teams ({num_teams})"
    # Load player data and pipeline path based on scoring type
    player_data_path = "data/final_data/nfl_stats_long_format_with_context_filtered_with_experience.csv"
    if scoring_type == "PPR":
        pipeline_path = "lightgbm_regression/joblib_files/lightgbm_regression_pipeline_PPR_20250514_214644.pkl"
    else:
        pipeline_path = "stacked_model/joblib_files/stacked_model_pipeline_Standard_20250514_214726.pkl"
    # Load player data
    df_players = pd.read_csv(player_data_path)
    df_players = df_players[df_players['Season'] == 2024].copy()
    available_players = set(df_players['Player Name'].unique())
    # Initialize other_teams_picks with empty lists for all teams
    other_teams_picks = {i: [] for i in range(1, num_teams + 1) if i != draft_position}
    new_draft_state = {
        'round': 1,
        'pick': 1,
        'draft_position': draft_position,
        'num_teams': num_teams,
        'num_rounds': num_rounds,
        'team_state': {
            'filled_positions': [],
            'drafted_players': [],
            'position_counts': {pos: 0 for pos in DEFAULT_ROSTER_CONFIG.keys()}
        },
        'other_teams_picks': other_teams_picks,
        'pick_history': [],
        'available_players': list(available_players),
        'error_message': None,
        'setup_collapsed': True,
        'player_data_path': player_data_path,
        'pipeline_path': pipeline_path,
        'scoring_type': scoring_type
    }
    return new_draft_state, {"display": "none"}, ""

@app.callback(
    Output('player-dropdown', 'options', allow_duplicate=True),
    [Input('draft-state', 'data')],
    prevent_initial_call=True
)
def update_player_dropdown(draft_state):
    if not draft_state.get('available_players'):
        # Initialize available players if not in state
        return [{'label': name, 'value': name} for name in sorted(AVAILABLE_PLAYERS)]
    
    return [{'label': name, 'value': name} for name in sorted(draft_state['available_players'])]

@app.callback(
    [Output('recommendations-table', 'children'),
     Output('current-round', 'children'),
     Output('current-pick', 'children'),
     Output('current-drafter', 'children'),
     Output('picks-until-turn', 'children'),
     Output('draft-history-table', 'children'),
     Output('submit-pick', 'disabled'),
     Output('pick-error-message', 'children')],
    [Input('draft-state', 'data'),
     Input('scoring-type', 'value'),
     Input('qb-limit', 'value'),
     Input('rb-limit', 'value'),
     Input('wr-limit', 'value'),
     Input('te-limit', 'value'),
     Input('k-limit', 'value'),
     Input('dst-limit', 'value')]
)
def update_display(draft_state, scoring_type, qb_limit, rb_limit, wr_limit, te_limit, k_limit, dst_limit):
    try:
        roster_config = {
            "QB": qb_limit or 1,
            "RB": rb_limit or 2,
            "WR": wr_limit or 2,
            "TE": te_limit or 1,
            "K": k_limit or 1,
            "DST": dst_limit or 1
        }
        # Load player data and pipeline from draft_state
        player_data_path = draft_state.get('player_data_path', "data/final_data/nfl_stats_long_format_with_context_filtered_with_experience.csv")
        pipeline_path = draft_state.get('pipeline_path', "lightgbm_regression/joblib_files/lightgbm_regression_pipeline_PPR_20250514_214644.pkl")
        df_players = pd.read_csv(player_data_path)
        df_players = df_players[df_players['Season'] == 2024].copy()
        from recommender_system import load_pipeline
        pipeline = load_pipeline(pipeline_path)
        # Get current drafter
        current_drafter = get_current_drafter(
            draft_state['round'],
            draft_state['pick'],
            draft_state['num_teams']
        )
        # Calculate picks until user's turn
        user_position = draft_state['draft_position']
        picks_until_turn = 0
        if current_drafter != user_position:
            current_round = draft_state['round']
            current_pick = draft_state['pick']
            num_teams = draft_state['num_teams']
            if current_round % 2 == 0:
                current_pos = num_teams - current_drafter + 1
                user_pos = num_teams - user_position + 1
            else:
                current_pos = current_drafter
                user_pos = user_position
            if current_pick == num_teams:
                if current_round % 2 == 1:
                    picks_until_turn = num_teams - user_position + 1
                else:
                    picks_until_turn = user_position
            else:
                if current_pos < user_pos:
                    picks_until_turn = user_pos - current_pos
                else:
                    remaining_in_round = num_teams - current_pick
                    if current_round % 2 == 1:
                        picks_until_turn = remaining_in_round + (num_teams - user_position + 1)
                    else:
                        picks_until_turn = remaining_in_round + user_position
        # Get recommendations from available players
        available_players = df_players[df_players['Player Name'].isin(draft_state.get('available_players', set(df_players['Player Name'].unique())))]
        # Add engineered features ONLY for PPR (LightGBM model)
        if scoring_type == "PPR":
            available_players['QB_Features'] = (available_players['Position'] == 'QB').astype(int)
            available_players['RB_Features'] = (available_players['Position'] == 'RB').astype(int)
            available_players['WR_Features'] = (available_players['Position'] == 'WR').astype(int)
            available_players['TE_Features'] = (available_players['Position'] == 'TE').astype(int)
            available_players['QB_Passing_Yards'] = available_players['QB_Features'] * available_players['Yards per Completion']
            available_players['RB_Rushing_Yards'] = available_players['RB_Features'] * available_players['Yards per Carry']
            available_players['WR_Receiving_Yards'] = available_players['WR_Features'] * available_players['Yards per Reception']
            available_players['TE_Receiving_Yards'] = available_players['TE_Features'] * available_players['Yards per Reception']
        try:
            from recommender_system import recommend_players
            recommendations = recommend_players(
                df_players=available_players,
                round_num=draft_state['round'],
                team_state=draft_state.get('team_state', {'position_counts': {}}),
                pipeline=pipeline,
                scoring_type=scoring_type,
                roster_config=roster_config,
                top_n=100, 
                num_rounds=draft_state['num_rounds'],
                league_size=draft_state['num_teams'],
                vor_weight=0.7
            )
            recommendations['Predicted Points'] = recommendations['Predicted_Points'].map(lambda x: f"{x:.1f}")
            recommendations['Value Over Replacement'] = recommendations['VOR'].round(1)
            recommendations['Overall Score'] = recommendations['Overall_Score'].round(3)
            table = html.Div([
                dbc.Table.from_dataframe(
                    recommendations[['Player Name', 'Position', 'Team', 'Predicted Points', 'Value Over Replacement', 'Overall Score']],
                    striped=True,
                    bordered=True,
                    hover=True
                )
            ], style={
                'maxHeight': '300px',
                'overflowY': 'auto',
                'overflowX': 'hidden',
                'marginBottom': '20px'
            })
        except Exception as e:
            print(f"Error generating recommendations: {str(e)}")
            table = html.Div("Unable to generate recommendations at this time.")
        # Create draft history table from the pick_history list
        history_table = None
        if draft_state.get('pick_history'):
            history_data = []
            for pick in draft_state['pick_history']:
                history_data.append({
                    'Round': pick['round'],
                    'Pick': pick['pick'],
                    'Team': f"Team {pick['team']}" + (' (You)' if pick['is_user'] else ''),
                    'Player': pick['player'],
                    'Position': pick['position']
                })
            if history_data:
                history_df = pd.DataFrame(history_data)
                history_table = dbc.Table.from_dataframe(
                    history_df,
                    striped=True,
                    bordered=True,
                    hover=True
                )
        is_draft_complete = draft_state['round'] > draft_state['num_rounds']
        error_message = draft_state.get('error_message')
        if is_draft_complete and not error_message:
            error_message = "Draft is complete!"
        return (
            table,
            f"Round {draft_state['round']}",
            f"Pick {draft_state['pick']}",
            f"Team {current_drafter}'s turn to draft",
            f"{picks_until_turn} picks until your turn" if picks_until_turn > 0 else "Your turn to draft",
            history_table,
            is_draft_complete,
            error_message
        )
    except Exception as e:
        print(f"Error in update_display: {str(e)}")
        return (
            html.Div("Error loading recommendations"),
            "Round -",
            "Pick -",
            "Loading...",
            "Loading...",
            None,
            False,
            draft_state.get('error_message', f"An error occurred. Please try refreshing the page.")
        )

@app.callback(
    [Output('draft-state', 'data', allow_duplicate=True),
     Output('pick-error-message', 'children', allow_duplicate=True)],
    Input('submit-pick', 'n_clicks'),
    [State('player-dropdown', 'value'),
     State('draft-state', 'data')],
    prevent_initial_call=True
)
def submit_pick(n_clicks, selected_player, draft_state):
    if n_clicks and selected_player:
        # Don't process picks if draft is complete
        if draft_state['round'] > draft_state['num_rounds']:
            draft_state['error_message'] = "Draft is complete!"
            return draft_state, draft_state['error_message']
        
        # Check if player is available
        available_players = set(draft_state.get('available_players', AVAILABLE_PLAYERS))
        if selected_player not in available_players:
            draft_state['error_message'] = f"Error: {selected_player} is not available!"
            return draft_state, draft_state['error_message']
            
        current_drafter = get_current_drafter(
            draft_state['round'],
            draft_state['pick'],
            draft_state['num_teams']
        )
        
        new_state = {
            'round': draft_state['round'],
            'pick': draft_state['pick'],
            'draft_position': draft_state['draft_position'],
            'num_teams': draft_state['num_teams'],
            'num_rounds': draft_state['num_rounds'],
            'team_state': draft_state['team_state'].copy(),
            'other_teams_picks': {k: v.copy() for k, v in draft_state['other_teams_picks'].items()},
            'pick_history': draft_state.get('pick_history', []).copy(),
            'available_players': list(available_players - {selected_player}),  # Remove drafted player
            'error_message': None,  # Clear error message on successful pick
            'setup_collapsed': True  # Ensure setup stays collapsed after pick
        }
        
        # Add the pick to history first
        player_data = df_players[df_players['Player Name'] == selected_player].iloc[0]
        new_state['pick_history'].append({
            'round': new_state['round'],
            'pick': new_state['pick'],
            'team': current_drafter,
            'player': selected_player,
            'position': player_data['Position'],
            'is_user': current_drafter == draft_state['draft_position']
        })
        
        # Update based on whether it's the user's turn or not
        if current_drafter == draft_state['draft_position']:
            # User's pick
            new_state['team_state']['drafted_players'].append(selected_player)
            new_state['team_state']['position_counts'][player_data['Position']] = new_state['team_state']['position_counts'].get(player_data['Position'], 0) + 1
        else:
            # Other team's pick
            if current_drafter not in new_state['other_teams_picks']:
                new_state['other_teams_picks'][current_drafter] = []
            new_state['other_teams_picks'][current_drafter].append(selected_player)
        
        # Update pick number
        new_state['pick'] += 1
        if new_state['pick'] > new_state['num_teams']:
            new_state['round'] += 1
            new_state['pick'] = 1
        
        return new_state, None
    
    return draft_state, None

# Add new callback for adding players
@app.callback(
    [Output('draft-state', 'data', allow_duplicate=True),
     Output('pick-error-message', 'children', allow_duplicate=True)],
    Input('add-player-button', 'n_clicks'),
    [State('new-player-name', 'value'),
     State('new-player-position', 'value'),
     State('new-player-team', 'value'),
     State('draft-state', 'data')],
    prevent_initial_call=True
)
def add_new_player(n_clicks, player_name, position, team, draft_state):
    if not n_clicks or not player_name or not position or not team:
        return draft_state, None
    
    global df_players
    
    # Create new state
    new_state = draft_state.copy()
    
    # Add the new player to available players
    if 'available_players' not in new_state:
        new_state['available_players'] = list(AVAILABLE_PLAYERS)
    
    # Check if player already exists in available players
    if player_name in new_state['available_players']:
        new_state['error_message'] = f"Error: {player_name} is already in the available players list!"
        return new_state, new_state['error_message']
    
    # Check if player has been drafted by the user
    if player_name in draft_state['team_state']['drafted_players']:
        new_state['error_message'] = f"Error: {player_name} has already been drafted by you!"
        return new_state, new_state['error_message']
    
    # Check if player has been drafted by other teams
    for team_picks in draft_state['other_teams_picks'].values():
        if player_name in team_picks:
            new_state['error_message'] = f"Error: {player_name} has already been drafted by another team!"
            return new_state, new_state['error_message']
    
    # Check if player exists in the global dataset
    if player_name in df_players['Player Name'].values:
        new_state['error_message'] = f"Error: {player_name} already exists in the player database!"
        return new_state, new_state['error_message']
    
    new_state['available_players'].append(player_name)
    new_state['error_message'] = None  # Clear error message on successful add
    
    # Add the player to the dataframe for recommendations
    new_player = pd.DataFrame({
        'Player Name': [player_name],
        'Position': [position],
        'Team': [team],
        'Season': [2024],
        # Fill other required columns with default values
        'Age': [22],  # Default rookie age
        'Games Played': [0],
        'Games Started': [0],
        # Add other required columns with default values
        'Target_PPR': [0],
        'Target_Standard': [0],
        'PPR Fantasy Points Scored': [0],
        'Standard Fantasy Points Scored': [0],
        'Delta_PPR_Fantasy_Points': [0],
        'Delta_Standard_Fantasy_Points': [0]
    })
    
    df_players = pd.concat([df_players, new_player], ignore_index=True)
    
    return new_state, f"Added {player_name} ({position}-{team}) to available players"

### CREATES NEW CREATE TEAM OVERVIEW FUNCTION
### Generate team roster tables for all teams based on draft pick history
    


def create_team_overview(draft_state):
    num_teams = draft_state.get("num_teams", 12)
    num_rounds = draft_state.get("num_rounds", 15)
    pick_history = draft_state.get("pick_history", [])

    # Roster template
    starting_slots = ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "DEF", "K"]
    bench_count = num_rounds - len(starting_slots)
    full_roster = starting_slots + ["BENCH"] * bench_count

    # Build per-team draft records
    team_rosters = {i: [] for i in range(1, num_teams + 1)}
    for pick in pick_history:
        team = pick["team"]
        team_rosters[team].append({
            "name": pick["player"],
            "position": pick["position"]
        })

    team_cards = []

    for team_num in range(1, num_teams + 1):
        players = team_rosters.get(team_num, [])

        slot_limits = {
            "QB": 1,
            "RB": 2,
            "WR": 2,
            "TE": 1,
            "K": 1,
            "DEF": 1,
            "FLEX": 1  # Only for RB/WR/TE
        }

        assigned = []
        bench = []

        for p in players:
            pos = p["position"]
            name = p["name"]

            # Try primary slot
            if slot_limits.get(pos, 0) > 0:
                assigned.append({"slot": pos, "name": name})
                slot_limits[pos] -= 1
            # Try FLEX if eligible
            elif pos in ["RB", "WR", "TE"] and slot_limits["FLEX"] > 0:
                assigned.append({"slot": "FLEX", "name": name})
                slot_limits["FLEX"] -= 1
            else:
                bench.append({"slot": "BENCH", "name": name})

        # Fill BENCH
        assigned += bench[:bench_count]

        # Fill remaining with placeholders
        while len(assigned) < num_rounds:
            assigned.append({"slot": "BENCH", "name": "---"})

        # Render rows in correct order
        rows = []
        for slot in full_roster:
            match = next((a for a in assigned if a["slot"] == slot), None)
            if match:
                rows.append(html.Tr([
                    html.Td(slot, style={"textAlign": "center"}),
                    html.Td(match["name"], style={"textAlign": "center"})
                ]))
                assigned.remove(match)
            else:
                rows.append(html.Tr([
                    html.Td(slot, style={"textAlign": "center"}),
                    html.Td("---", style={"textAlign": "center"})
                ]))

        card = dbc.Card([
            dbc.CardHeader(f"Team {team_num}", className="text-center text-white"),
            dbc.CardBody([
                dbc.Table([
                    html.Thead(html.Tr([
                        html.Th("Position", style={"textAlign": "center"}),
                        html.Th("Player", style={"textAlign": "center"})
                    ])),
                    html.Tbody(rows)
                ], bordered=True, hover=False, style={"color": "white", "width": "100%"})
            ])
        ], style={
            "minWidth": "180px",
            "margin": "10px",
            "backgroundColor": "#2a2a2a",
            "border": "1px solid #444"
        })

        team_cards.append(card)

    # Create rows of team cards, spaced evenly
    rows = []
    if num_teams <= 4:
        col_width = 12 // num_teams
        row = dbc.Row([
            dbc.Col(card, width=col_width) for card in team_cards
        ], className="mb-4", justify="center")
        rows.append(row)
    else:
        for i in range(0, len(team_cards), 4):
            row_cards = team_cards[i:i+4]
            row = dbc.Row([
                dbc.Col(card, width=3) for card in row_cards
            ], className="mb-4", justify="center")
            rows.append(row)

    return html.Div(
        dbc.Container(rows, fluid=True),
        style={"padding": "20px"}
    )

@app.callback(
    Output("add-player-collapse", "is_open"),
    [Input("toggle-add-player", "n_clicks")],
    [State("add-player-collapse", "is_open")],
)
def toggle_add_player(n_clicks, is_open):
    if n_clicks:
        return not is_open
    return is_open

@app.callback(
    Output("draft-history-collapse", "is_open"),
    [Input("toggle-draft-history", "n_clicks")],
    [State("draft-history-collapse", "is_open")],
)
def toggle_draft_history(n_clicks, is_open):
    if n_clicks:
        return not is_open
    return is_open

@app.callback(
    Output("viz-output", "children"),
    Input("viz-show-btn", "n_clicks"),
    State("viz-player-dropdown", "value"),
    State("viz-stat-dropdown", "value"),
    State("viz-chart-type", "value"),
    State("viz-seasons", "value"),
    prevent_initial_call=True
)

def mpl_fig_to_base64_img(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode()
    plt.close(fig)  # Avoid memory leaks
    return f"data:image/png;base64,{encoded}"
    
@app.callback(
    Output("viz-output", "children"),
    Input("viz-show-btn", "n_clicks"),
    State("viz-player-dropdown", "value"),
    State("viz-stat-dropdown", "value"),
    State("viz-chart-type", "value"),
    State("viz-seasons", "value"),
    prevent_initial_call=True
)

def update_visual(n_clicks, players, stats, chart_type, seasons_text):
    if not players or not stats:
        return "Select at least one player and one stat."

    # Parse season input
    if seasons_text:
        try:
            seasons = [int(s.strip()) for s in seasons_text.split(",") if s.strip()]
        except ValueError:
            return "Season field must be comma-separated years (e.g. 2023, 2024)."
    else:
        seasons = [df_players["Season"].max()]

    # Filter the data just like compare_stats would
    df_filtered = df_players[df_players["Player Name"].isin(players) & df_players["Season"].isin(seasons)]

    if df_filtered.empty:
        return "No data found for selected players/seasons."

    # Run compare_stats but capture figures instead of showing them
    figures = []

    # Call your function and capture figures
    fig_list = compare_stats(df_filtered, players, stats, chart_type, seasons, return_figs=True)

    for fig in fig_list:
        img_uri = mpl_fig_to_base64_img(fig)
        figures.append(html.Img(src=img_uri, style={"width": "90%", "margin": "20px auto", "display": "block"}))

    return figures
# Add new callback for updating roster limits
@app.callback(
    [Output(f"{pos.lower()}-limit", "value") for pos in ["QB", "RB", "WR", "TE", "K", "DST"]],
    Input("num-rounds", "value"),
    prevent_initial_call=True
)
def update_roster_limits(num_rounds):
    if not num_rounds:
        return [DEFAULT_ROSTER_CONFIG[pos] for pos in ["QB", "RB", "WR", "TE", "K", "DST"]]
    
    limits = calculate_roster_limits(num_rounds)
    return [limits[pos] for pos in ["QB", "RB", "WR", "TE", "K", "DST"]]
@app.callback(
    Output("tab-content", "children"),
    Input("main-tabs", "value"),
    State("draft-state", "data")
)

### CREATES NEW TAB FOR TEAM VIEWs

def render_tab_content(tab, draft_state):
    if tab == "draft":
        return dbc.Row([
            dbc.Col(create_draft_setup(), width=12, className="mb-4"),
            dbc.Col(create_draft_board(), width=12, className="mb-4")
        ])
    
    elif tab == "teams":
        # Only render if draft has started
        if draft_state["pick_history"] == [] and draft_state["round"] == 1 and draft_state["pick"] == 1:
            return html.Div(
                "Teams will be displayed here after the draft starts.",
                style={"padding": "20px", "color": "white"}
            )
    elif tab == "viz":                              # <‑‑ NEW
        return dbc.Container(create_visualizations_tab(), fluid=True)
    
    ### CENTERS NEW TAB CONTENT
    
    return dbc.Container(
    create_team_overview(draft_state),
    fluid=True,
    style={"display": "flex", "justifyContent": "center", "padding": "20px"}
)
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            .dash-dropdown .Select-menu-outer {
                background-color: #222 !important;
            }
            .dash-dropdown .Select-option {
                color: white !important;
                background-color: #222 !important;
            }
            .dash-dropdown .Select-value-label {
                color: white !important;
            }
            .dash-dropdown .Select-option:hover {
                background-color: #375a7f !important;
            }
            .dash-dropdown .Select-option.is-selected {
                background-color: #375a7f !important;
            }
            /* Make dropdown input text white */
            .dash-dropdown .Select-input {
                color: white !important;
            }
            .dash-dropdown .Select-input input {
                color: white !important;
            }
            .dash-dropdown .Select-control {
                color: white !important;
            }
            /* Remove all collapse animations */
            .collapse {
                transition: none !important;
            }
            .collapsing {
                transition: none !important;
            }
            .collapse.show {
                transition: none !important;
            }
            /* Override Bootstrap's transition */
            .collapse, .collapsing {
                transition-property: none !important;
                transition-duration: 0s !important;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

if __name__ == '__main__':
    app.run(debug=True) 
