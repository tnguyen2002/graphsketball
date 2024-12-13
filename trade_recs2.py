import torch
from torch_geometric.data import Data
import json
import os
from torch.distributions import Normal
from copy import deepcopy
import math

from LeagueModel import LeagueGNN

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


def build_graph_from_season_data(season_data, standings_data):
    """Build a graph Data object from raw season_data (player-level) and standings_data."""
    team_to_players = {}
    team_win_percentages = {}

    for player_data in season_data:
        team = player_data['team']
        if team not in team_to_players:
            team_to_players[team] = []
        features = [player_data.get(f, 0.0) for f in numeric_features]
        features_tensor = torch.tensor(features, dtype=torch.float)
        team_to_players[team].append(features_tensor)

    for team_data in standings_data:
        team = team_data['team']
        wins = team_data['wins']
        losses = team_data['losses']
        team_win_percentages[team] = wins / (wins + losses)

    node_features = []
    edge_index = []
    labels = []
    batch_list = []
    teams_in_order = []
    team_start_idx = 0

    for team_idx, (team, players) in enumerate(team_to_players.items()):
        if team not in team_win_percentages:
            continue
        num_players = len(players)
        team_node_features = torch.stack(players)
        node_features.append(team_node_features)

        team_edges = torch.combinations(
            torch.arange(team_start_idx, team_start_idx + num_players),
            r=2,
            with_replacement=False
        ).t()

        if team_edges.numel() > 0:
            edge_index.append(team_edges)

        batch_list.extend([team_idx] * num_players)
        labels.append(team_win_percentages[team])
        teams_in_order.append(team)
        team_start_idx += num_players

    if len(node_features) > 0:
        league_node_features = torch.cat(node_features, dim=0)
    else:
        league_node_features = torch.empty((0, len(numeric_features)))

    if len(edge_index) > 0:
        league_edge_index = torch.cat(edge_index, dim=1)
    else:
        league_edge_index = torch.empty((2,0), dtype=torch.long)

    league_labels = torch.tensor(labels, dtype=torch.float)
    batch_tensor = torch.tensor(batch_list, dtype=torch.long)

    data = Data(x=league_node_features, edge_index=league_edge_index, y=league_labels, batch=batch_tensor)
    return data, teams_in_order, team_win_percentages


def load_season_data(player_file, standings_file):
    with open(player_file, 'r') as f:
        season_data = json.load(f)
    with open(standings_file, 'r') as f:
        standings_data = json.load(f)
    return season_data, standings_data


def predict(model, data):
    model.eval()
    with torch.no_grad():
        predictions = model(data).squeeze().tolist()
    if isinstance(predictions, float):
        predictions = [predictions]
    return predictions


def print_standings(predictions, teams_in_order):
    """Print nicely formatted standings from predictions."""
    # This code is adapted from your snippet. We remove hard-coded 30 teams logic and adapt it.
    num_teams = len(teams_in_order)
    target_sum = num_teams * 0.5  # We want average ~0.5 win fraction

    S = sum(predictions)
    max_fraction = 73.0 / 82.0  # Upper bound fraction for realism

    if math.isclose(S, 0.0, rel_tol=1e-9):
        # If all predictions are zero, just evenly distribute
        predictions = [0.5]*num_teams
    else:
        p_min = min(predictions)
        p_max = max(predictions)

        if math.isclose(p_min, p_max, rel_tol=1e-9):
            predictions = [0.5]*num_teams
        else:
            # Attempt similar linear transform: a*p + b
            # We want sum(a*p_i + b) = target_sum
            # a*S + num_teams*b = target_sum
            # b = (target_sum - a*S)/num_teams
            #
            # Constraints: 0 <= a*p_min + b <= max_fraction and same for p_max
            # Solve for feasible a similarly to original code:

            denom_min = p_min - (S/num_teams)
            denom_max = p_max - (S/num_teams)

            a_lower_bound = -math.inf
            a_upper_bound = math.inf

            # Min bound: a*p_min + b >= 0
            # a*(p_min - S/num_teams) >= -target_sum/num_teams
            # => a*denom_min >= -(target_sum/num_teams)
            min_rhs = -(target_sum/num_teams)
            if not math.isclose(denom_min, 0, rel_tol=1e-12):
                required_a = min_rhs/denom_min
                if denom_min > 0:
                    a_lower_bound = max(a_lower_bound, required_a)
                else:
                    a_upper_bound = min(a_upper_bound, required_a)

            # Max bound: a*p_max + b <= max_fraction
            # a*(p_max - S/num_teams) <= max_fraction - target_sum/num_teams
            max_rhs = max_fraction - (target_sum/num_teams)
            if not math.isclose(denom_max, 0, rel_tol=1e-12):
                required_a = max_rhs/denom_max
                if denom_max > 0:
                    a_upper_bound = min(a_upper_bound, required_a)
                else:
                    a_lower_bound = max(a_lower_bound, required_a)

            # Check feasibility
            if a_lower_bound > a_upper_bound:
                # No feasible solution
                # Fallback: just scale to sum=target_sum and then clip
                a = target_sum/S
                b = 0.0
                adjusted = [min(max(a*p+b, 0.0), max_fraction) for p in predictions]
            else:
                # Choose a in feasible range. Try a = target_sum/S if possible.
                a_candidate = target_sum/S
                if a_candidate < a_lower_bound:
                    a = a_lower_bound
                elif a_candidate > a_upper_bound:
                    a = a_upper_bound
                else:
                    a = a_candidate

                b = (target_sum - a*S)/num_teams
                adjusted = [a*p+b for p in predictions]
                # Clip to ensure 0..max_fraction
                adjusted = [min(max(x, 0.0), max_fraction) for x in adjusted]
            predictions = adjusted

    # Convert fractions to win/loss records
    wins = [int(round(p * 82)) for p in predictions]
    wins = [min(w, 73) for w in wins]  # Just ensure no team exceeds 73 wins
    team_records = {team: (w, 82-w) for team, w in zip(teams_in_order, wins)}

    # Sort each conference
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

    print("EASTERN CONFERENCE STANDINGS")
    print("-" * 30)
    for rank, (team, record) in enumerate(eastern_standings, start=1):
        print(f"{rank:2}. {team:25} {record[0]:2}-{record[1]:2}")

    print("\nWESTERN CONFERENCE STANDINGS")
    print("-" * 30)
    for rank, (team, record) in enumerate(western_standings, start=1):
        print(f"{rank:2}. {team:25} {record[0]:2}-{record[1]:2}")


def rebuild_graph_with_player_moved(original_season_data, standings_data, player_name, new_team):
    """Rebuild the graph after moving a given player to a new team (and removing from old team)."""
    # We will copy the original season_data to avoid permanent modifications
    modified_data = deepcopy(original_season_data)

    # Find the player and change their team
    player_found = False
    for p in modified_data:
        if p['name'] == player_name:
            p['team'] = new_team
            player_found = True
            break

    if not player_found:
        raise ValueError(f"Player {player_name} not found in season data.")

    # Now build a new graph from modified_data
    data, teams_in_order, _ = build_graph_from_season_data(modified_data, standings_data)
    return data, teams_in_order


def main():
    # Specify which season to load
    season = "2024_2025"  # Example
    player_file = f"data/player_season_jsons/{season}_advanced_player_season_totals.json"
    standings_file = f"data/standings_jsons/{season}_standings.json"

    original_season_data, standings_data = load_season_data(player_file, standings_file)
    original_data, teams_in_order, team_win_percentages = build_graph_from_season_data(original_season_data, standings_data)

    # Load trained model
    model = LeagueGNN(in_channels=original_data.x.size(1), hidden_channels=64)
    model.load_state_dict(torch.load('league_predictor_best_weights.pth', map_location=torch.device('cpu')))
    model.eval()

    # Baseline predictions
    baseline_predictions = predict(model, original_data)
    print("Baseline Standings:")
    print_standings(baseline_predictions, teams_in_order)
    print()

    # Target scenario: Adding players to the Warriors
    target_team = "GOLDEN STATE WARRIORS"
    baseline_warriors_prediction = baseline_predictions[teams_in_order.index(target_team)]

    # Gather candidates (player name, features, and original team)
    candidates = []
    for player_data in original_season_data:
        if player_data['team'] == target_team:
            continue
        features = [player_data.get(f, 0.0) for f in numeric_features]
        features_tensor = torch.tensor(features, dtype=torch.float)
        player_name = player_data['name']
        original_team = player_data['team']
        candidates.append((player_name, features_tensor, original_team))

    improvements = []
    # Evaluate each candidate
    for player_name, feat, old_team in candidates:
        # Move player from old_team to Warriors
        modified_data, modified_teams_in_order = rebuild_graph_with_player_moved(original_season_data, standings_data, player_name, target_team)

        # Predictions for modified scenario
        new_predictions = predict(model, modified_data)
        if target_team in modified_teams_in_order:
            new_warriors_prediction = new_predictions[modified_teams_in_order.index(target_team)]
            improvement = new_warriors_prediction - baseline_warriors_prediction
            improvements.append((player_name, old_team, improvement, new_predictions, modified_teams_in_order))
        else:
            # If target team somehow is missing, skip
            continue

    # Sort by improvement descending
    improvements.sort(key=lambda x: x[2], reverse=True)

    # Get top k players
    k = 10
    top_k = improvements[:k]

    print("Top k players who improve the Warriors the most:")
    for player_name, old_team, imp, preds, t_in_order in top_k:
        print(f"{player_name} (from {old_team}): +{imp:.4f} improvement in predicted win%")
        # Print standings for this scenario
        print(f"\nStandings after adding {player_name} to the Warriors and removing from {old_team}:")
        print_standings(preds, t_in_order)
        print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    main()
