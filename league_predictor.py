import torch
from torch_geometric.data import Data
import json
import torch.nn.functional as F
from LeagueModel import LeagueGNN
import math


def load_season_data(file_path, numeric_features):
    """Load player season data from JSON and create a single league graph."""
    with open(file_path, 'r') as f:
        season_data = json.load(f)

    team_to_players = {}
    node_features = []
    batch_list = []
    edge_index = []

    for player_data in season_data:
        team = player_data['team']
        if team not in team_to_players:
            team_to_players[team] = []

        features = [player_data.get(f, 0.0) for f in numeric_features]
        features_tensor = torch.tensor(features, dtype=torch.float)
        team_to_players[team].append(features_tensor)

    team_start_idx = 0
    labels = []
    idx_to_team = {}

    for team_idx, (team, players) in enumerate(team_to_players.items()):
        idx_to_team[team_idx] = team
        num_players = len(players)
        team_node_features = torch.stack(players)
        node_features.append(team_node_features)

        # Fully connected team graph edges
        team_edges = torch.combinations(
            torch.arange(team_start_idx, team_start_idx + num_players),
            r=2,
            with_replacement=False
        ).t()
        edge_index.append(team_edges)

        batch_list.extend([team_idx] * num_players)
        team_start_idx += num_players

    # Combine into a single graph
    league_node_features = torch.cat(node_features, dim=0)
    league_edge_index = torch.cat(edge_index, dim=1)
    batch_tensor = torch.tensor(batch_list, dtype=torch.long)

    return Data(x=league_node_features, edge_index=league_edge_index, batch=batch_tensor, teams=list(team_to_players.keys())), idx_to_team


def predict_team_wins(file_path, model_path, numeric_features):
    """Predict win/loss records for a single season."""
    # Load the league graph for the season
    league_graph, idx_to_team = load_season_data(file_path, numeric_features)

    # Load the trained model
    in_channels = league_graph.x.size(1)
    model = LeagueGNN(in_channels=in_channels, hidden_channels=64)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Predict win percentages
    with torch.no_grad():
        predictions = model(league_graph).squeeze().tolist()
        
    S = sum(predictions)
    max_fraction = 73.0 / 82.0  # Upper bound fraction for realism

    if S == 0:
        # If all predictions are zero, just evenly distribute to match sum=15
        predictions = [15.0/30] * 30
    else:
        p_min = min(predictions)
        p_max = max(predictions)

        # If all predictions are the same, only one solution: all at 0.5
        if math.isclose(p_min, p_max, rel_tol=1e-9):
            predictions = [0.5] * 30
        else:
            # We want to solve for a and b in:
            # a*S + 30*b = 15  => b = (15 - a*S)/30
            #
            # Range constraints:
            # 0 <= a*p_min + b and a*p_max + b <= max_fraction
            #
            # For min bound: a*p_min + (15 - a*S)/30 >= 0
            # => a*(p_min - S/30) >= -0.5
            #
            # For max bound: a*p_max + (15 - a*S)/30 <= max_fraction
            # => a*p_max - a*S/30 <= max_fraction - 0.5
            # => a*(p_max - S/30) <= max_fraction - 0.5

            denom_min = p_min - (S/30.0)
            denom_max = p_max - (S/30.0)

            feasible = True
            a_lower_bound = -math.inf
            a_upper_bound = math.inf

            # From min bound: a*(p_min - S/30) >= -0.5
            if not math.isclose(denom_min, 0, rel_tol=1e-12):
                required_a = -0.5 / denom_min
                if denom_min > 0:
                    # a >= required_a
                    a_lower_bound = max(a_lower_bound, required_a)
                else:
                    # a <= required_a
                    a_upper_bound = min(a_upper_bound, required_a)

            # From max bound: a*(p_max - S/30) <= (max_fraction - 0.5)
            max_rhs = max_fraction - 0.5
            if not math.isclose(denom_max, 0, rel_tol=1e-12):
                required_a = max_rhs / denom_max
                if denom_max > 0:
                    # a <= required_a
                    a_upper_bound = min(a_upper_bound, required_a)
                else:
                    # a >= required_a
                    a_lower_bound = max(a_lower_bound, required_a)

            if a_lower_bound > a_upper_bound:
                # No feasible solution
                feasible = False

            if feasible:
                # Choose an a in the feasible range, try midpoint or fallback to 15/S if no bounds
                if math.isinf(a_lower_bound) and math.isinf(a_upper_bound):
                    a = 15.0 / S
                elif math.isinf(a_lower_bound):
                    # No lower bound
                    a = min(a_upper_bound, 15.0 / S)
                elif math.isinf(a_upper_bound):
                    # No upper bound
                    a = max(a_lower_bound, 15.0 / S)
                else:
                    # Both bounds finite
                    a = (a_lower_bound + a_upper_bound) / 2.0

                b = (15 - a*S)/30.0
                adjusted = [a*p + b for p in predictions]

                # Check feasibility again
                if any(x < 0 for x in adjusted) or any(x > max_fraction for x in adjusted):
                    feasible = False

            if not feasible:
                # Fallback: just scale to sum=15 and then clip to [0, max_fraction]
                a = 15.0 / S
                b = 0.0
                adjusted = [a*p + b for p in predictions]
                adjusted = [min(max(x, 0.0), max_fraction) for x in adjusted]

            predictions = adjusted

    # Convert fractions to win/loss records
    # Ensure no team exceeds 73 wins
    # After ensuring fraction <= max_fraction, fraction*82 <= 73
    wins = [int(round(p * 82)) for p in predictions]
    wins = [min(w, 73) for w in wins]  # Double-check capping
    win_loss_records = [(w, 82 - w) for w in wins]
    print(predictions)

    # # Print team win/loss records
    # for team, record in zip(team_names, win_loss_records):
    #     print(f"{team}: {record[0]}-{record[1]}")
    
    eastern_conference = [
        "MILWAUKEE BUCKS", "BOSTON CELTICS", "ORLANDO MAGIC", "NEW YORK KNICKS", 
        "CLEVELAND CAVALIERS", "CHARLOTTE HORNETS", "MIAMI HEAT", "CHICAGO BULLS", 
        "TORONTO RAPTORS", "DETROIT PISTONS", "ATLANTA HAWKS", "INDIANA PACERS", 
        "BROOKLYN NETS", "WASHINGTON WIZARDS", "PHILADELPHIA 76ERS"
    ]
    western_conference = [
        "OKLAHOMA CITY THUNDER", "SACRAMENTO KINGS", "LOS ANGELES LAKERS", 
        "DENVER NUGGETS", "MINNESOTA TIMBERWOLVES", "PHOENIX SUNS", 
        "DALLAS MAVERICKS", "LOS ANGELES CLIPPERS", "MEMPHIS GRIZZLIES", 
        "HOUSTON ROCKETS", "SAN ANTONIO SPURS", "GOLDEN STATE WARRIORS", 
        "NEW ORLEANS PELICANS", "UTAH JAZZ", "PORTLAND TRAIL BLAZERS"
    ]

    # Pair teams with their win/loss records
    team_records = {team: record for team, record in zip([idx_to_team[idx] for idx in range(len(win_loss_records))], win_loss_records)}

    # Sort teams in each conference by wins (descending order)
    eastern_standings = sorted(
        [(team, team_records[team]) for team in eastern_conference if team in team_records],
        key=lambda x: x[1][0],
        reverse=True
    )
    western_standings = sorted(
        [(team, team_records[team]) for team in western_conference if team in team_records],
        key=lambda x: x[1][0],
        reverse=True
    )

    # Print standings
    print("EASTERN CONFERENCE STANDINGS")
    print("-" * 30)
    for rank, (team, record) in enumerate(eastern_standings, start=1):
        print(f"{rank:2}. {team:25} {record[0]:2}-{record[1]:2}")

    print("\nWESTERN CONFERENCE STANDINGS")
    print("-" * 30)
    for rank, (team, record) in enumerate(western_standings, start=1):
        print(f"{rank:2}. {team:25} {record[0]:2}-{record[1]:2}")


if __name__ == "__main__":
    numeric_features = [
        "age", "assist_percentage", "block_percentage", "box_plus_minus",
        "defensive_box_plus_minus", "defensive_rebound_percentage", "defensive_win_shares",
        "free_throw_attempt_rate", "games_played", "minutes_played", "offensive_box_plus_minus",
        "offensive_rebound_percentage", "offensive_win_shares", "player_efficiency_rating",
        "steal_percentage", "three_point_attempt_rate", "total_rebound_percentage",
        "true_shooting_percentage", "turnover_percentage", "usage_percentage",
        "value_over_replacement_player", "win_shares", "win_shares_per_48_minutes",
        "is_combined_totals"
    ]

    file_path = "data/player_season_jsons/2024_2025_advanced_player_season_totals.json"  # Replace with your JSON file path
    model_path = "league_predictor_best_weights.pth"
    predict_team_wins(file_path, model_path, numeric_features)
