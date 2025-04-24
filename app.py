import dash
from dash import html, dcc, Input, Output, State, callback
import dash_bootstrap_components as dbc
import pandas as pd
from recommender_system import recommend_players, load_pipeline

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

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
    "QB": 1,
    "RB": 4,
    "WR": 5,
    "TE": 2,
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

# Load the pipeline and player data
pipeline = load_pipeline("mlp_regression/joblib_files/mlp_pipeline_PPR.pkl")
df_players = pd.read_csv("data/final_data/nfl_stats_long_format_with_context_filtered.csv")
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
    return dbc.Collapse(
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
        id="draft-setup-collapse",
        is_open=True
    )

def create_draft_board():
    return dbc.Card([
        dbc.CardHeader("Draft Board"),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    dbc.Label("Current Round"),
                    html.Div(id="current-round", className="h4")
                ], width=3),
                dbc.Col([
                    dbc.Label("Current Pick"),
                    html.Div(id="current-pick", className="h4")
                ], width=3),
                dbc.Col([
                    dbc.Label("Current Drafter"),
                    html.Div(id="current-drafter", className="h4 text-primary")
                ], width=3),
                dbc.Col([
                    dbc.Label("Picks Until Your Turn"),
                    html.Div(id="picks-until-turn", className="h4")
                ], width=3),
            ]),
            dbc.Row([
                dbc.Col([
                    dbc.Label("Select Player"),
                    dcc.Dropdown(id="player-dropdown", multi=False),
                    dbc.Button("Submit Pick", id="submit-pick", color="primary", className="mt-2"),
                    html.Div(id="pick-error-message", className="text-danger mt-2")
                ], width=12)
            ]),
            html.Hr(),
            # Recommendations section
            dbc.Row([
                dbc.Col([
                    html.H5("Recommendations", className="mb-3"),
                    html.Div(id="recommendations-table")
                ], width=12)
            ]),
            html.Hr(),
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
                                html.H5("Add New Player"),
                                dbc.Row([
                                    dbc.Col([
                                        dbc.Label("Player Name"),
                                        dbc.Input(id="new-player-name", type="text", placeholder="Enter player name")
                                    ], width=4),
                                    dbc.Col([
                                        dbc.Label("Position"),
                                        dcc.Dropdown(
                                            id="new-player-position",
                                            options=[{"label": pos, "value": pos} for pos in ["QB", "RB", "WR", "TE", "K", "DST"]],
                                            placeholder="Select position"
                                        )
                                    ], width=4),
                                    dbc.Col([
                                        dbc.Label("Team"),
                                        dbc.Input(id="new-player-team", type="text", placeholder="Enter team (e.g., SF)")
                                    ], width=4)
                                ]),
                                dbc.Button("Add Player", id="add-player-button", color="secondary", className="mt-2")
                            ])
                        ]),
                        id="add-player-collapse",
                        is_open=False
                    )
                ], width=12)
            ]),
            html.Hr(),
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
                                html.H5("Draft History"),
                                html.Div(id="draft-history-table")
                            ])
                        ]),
                        id="draft-history-collapse",
                        is_open=False
                    )
                ], width=12)
            ])
        ])
    ])

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
    dbc.Row([
        dbc.Col(create_draft_setup(), width=12, className="mb-4"),
        dbc.Col(create_draft_board(), width=12, className="mb-4")
    ]),
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
        'error_message': None
    })
], fluid=True)

# Callbacks
@app.callback(
    [Output('draft-state', 'data'),
     Output('draft-setup-collapse', 'is_open'),
     Output('setup-error', 'children')],
    Input('start-draft', 'n_clicks'),
    [State('num-teams', 'value'),
     State('num-rounds', 'value'),
     State('draft-position', 'value'),
     State('draft-state', 'data')],
    prevent_initial_call=True
)
def start_draft(n_clicks, num_teams, num_rounds, draft_position, draft_state):
    if not n_clicks:
        return draft_state, True, ""
    
    # Validate inputs
    if not all([num_teams, num_rounds, draft_position]):
        return draft_state, True, "Please fill in all required fields"
    
    if draft_position > num_teams:
        return draft_state, True, f"Draft position cannot be greater than number of teams ({num_teams})"
    
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
        'available_players': list(AVAILABLE_PLAYERS),
        'error_message': None
    }
    
    # Return new state and collapse the setup section
    return new_draft_state, False, ""

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
        
        # Get current drafter
        current_drafter = get_current_drafter(
            draft_state['round'],
            draft_state['pick'],
            draft_state['num_teams']
        )
        
        # Calculate picks until user's turn
        user_position = draft_state['draft_position']
        picks_until_turn = 0
        
        if current_drafter != user_position:  # Only calculate if it's not user's turn
            if draft_state['round'] % 2 == 1:  # Odd rounds
                if current_drafter < user_position:
                    picks_until_turn = user_position - current_drafter
                else:
                    picks_until_turn = (draft_state['num_teams'] - current_drafter) + user_position
            else:  # Even rounds
                reversed_current = draft_state['num_teams'] - current_drafter + 1
                reversed_user = draft_state['num_teams'] - user_position + 1
                if reversed_current < reversed_user:
                    picks_until_turn = reversed_user - reversed_current
                else:
                    picks_until_turn = (draft_state['num_teams'] - reversed_current) + reversed_user
        
        # Get recommendations from available players
        available_players = df_players[df_players['Player Name'].isin(draft_state.get('available_players', AVAILABLE_PLAYERS))]
        
        try:
            recommendations = recommend_players(
                df_players=available_players,
                round_num=draft_state['round'],
                team_state=draft_state.get('team_state', {'position_counts': {}}),
                pipeline=pipeline,
                scoring_type=scoring_type or "PPR",
                roster_config=roster_config,
                top_n=5,
                num_rounds=draft_state['num_rounds']
            )
            
            table = dbc.Table.from_dataframe(
                recommendations[['Player Name', 'Position', 'Team', 'Predicted_Points']],
                striped=True,
                bordered=True,
                hover=True
            )
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
        
        # Determine if draft is complete
        is_draft_complete = draft_state['round'] > draft_state['num_rounds']
        error_message = draft_state.get('error_message')  # Get error message from state
        if is_draft_complete and not error_message:
            error_message = "Draft is complete!"
        
        return (
            table,
            f"Round {draft_state['round']}",
            f"Pick {draft_state['pick']}",
            f"Team {current_drafter}'s turn to draft",
            f"{picks_until_turn} picks until your turn",
            history_table,
            is_draft_complete,  # Disable submit button if draft is complete
            error_message
        )
    except Exception as e:
        print(f"Error in update_display: {str(e)}")
        # Return safe default values if an error occurs
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
            'error_message': None  # Clear error message on successful pick
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

# Add new callbacks for toggling sections
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

if __name__ == '__main__':
    app.run(debug=True) 