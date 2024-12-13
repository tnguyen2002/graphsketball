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
    num_teams = len(teams_in_order)
    target_sum = num_teams * 0.5  # We want average ~0.5 win fraction

    S = sum(predictions)
    max_fraction = 73.0 / 82.0  # Upper bound fraction for realism

    if math.isclose(S, 0.0, rel_tol=1e-9):
        predictions = [0.5]*num_teams
    else:
        p_min = min(predictions)
        p_max = max(predictions)

        if math.isclose(p_min, p_max, rel_tol=1e-9):
            predictions = [0.5]*num_teams
        else:
            denom_min = p_min - (S/num_teams)
            denom_max = p_max - (S/num_teams)

            a_lower_bound = -math.inf
            a_upper_bound = math.inf

            min_rhs = -(target_sum/num_teams)
            if not math.isclose(denom_min, 0, rel_tol=1e-12):
                required_a = min_rhs/denom_min
                if denom_min > 0:
                    a_lower_bound = max(a_lower_bound, required_a)
                else:
                    a_upper_bound = min(a_upper_bound, required_a)

            max_rhs = max_fraction - (target_sum/num_teams)
            if not math.isclose(denom_max, 0, rel_tol=1e-12):
                required_a = max_rhs/denom_max
                if denom_max > 0:
                    a_upper_bound = min(a_upper_bound, required_a)
                else:
                    a_lower_bound = max(a_lower_bound, required_a)

            if a_lower_bound > a_upper_bound:
                # No feasible solution
                a = target_sum/S
                b = 0.0
                adjusted = [min(max(a*p+b, 0.0), max_fraction) for p in predictions]
            else:
                a_candidate = target_sum/S
                if a_candidate < a_lower_bound:
                    a = a_lower_bound
                elif a_candidate > a_upper_bound:
                    a = a_upper_bound
                else:
                    a = a_candidate

                b = (target_sum - a*S)/num_teams
                adjusted = [a*p+b for p in predictions]
                adjusted = [min(max(x, 0.0), max_fraction) for x in adjusted]
            predictions = adjusted

    wins = [int(round(p * 82)) for p in predictions]
    wins = [min(w, 73) for w in wins]
    team_records = {team: (w, 82-w) for team, w in zip(teams_in_order, wins)}

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


def rebuild_graph_with_swap(original_season_data, standings_data, player_out_name, player_in_name):
    """
    Rebuild the graph after swapping two players:
    player_out_name: The player currently on target_team who will be swapped out.
    player_in_name: The player from another team who will be swapped in.

    Steps:
    - Find both players in the season_data.
    - Swap their teams: player_out goes to player_in's old team, player_in goes to player_out's old team.
    """
    modified_data = deepcopy(original_season_data)

    player_out_team = None
    player_in_team = None
    for p in modified_data:
        if p['name'] == player_out_name:
            player_out_team = p['team']
        if p['name'] == player_in_name:
            player_in_team = p['team']

    if player_out_team is None:
        raise ValueError(f"Player {player_out_name} not found.")
    if player_in_team is None:
        raise ValueError(f"Player {player_in_name} not found.")

    # Perform the swap
    player_out_found = False
    player_in_found = False
    for p in modified_data:
        if p['name'] == player_out_name:
            p['team'] = player_in_team
            player_out_found = True
        elif p['name'] == player_in_name:
            p['team'] = player_out_team
            player_in_found = True

    if not player_out_found or not player_in_found:
        raise ValueError("Could not perform swap, player not found in data.")

    data, teams_in_order, _ = build_graph_from_season_data(modified_data, standings_data)
    return data, teams_in_order


def main():
    season = "2024_2025"  # Example
    player_file = f"data/player_season_jsons/{season}_advanced_player_season_totals.json"
    standings_file = f"data/standings_jsons/{season}_standings.json"

    original_season_data, standings_data = load_season_data(player_file, standings_file)
    original_data, teams_in_order, team_win_percentages = build_graph_from_season_data(original_season_data, standings_data)

    # Load trained model
    model = LeagueGNN(in_channels=original_data.x.size(1), hidden_channels=64)
    model.load_state_dict(torch.load('league_predictor_best_weights.pth', map_location=torch.device('cpu')))
    model.eval()

    # Baseline predictions for all teams
    baseline_predictions = predict(model, original_data)
    baseline_team_preds = {team: baseline_predictions[i] for i, team in enumerate(teams_in_order)}

    print("Baseline Standings:")
    print_standings(baseline_predictions, teams_in_order)
    print()

    target_team = "GOLDEN STATE WARRIORS"
    baseline_warriors_prediction = baseline_team_preds[target_team]

    # Identify Warriors players and non-Warriors players
    # We'll use the original_season_data to know which players are on the Warriors
    warriors_players = []
    other_players = []
    for p in original_season_data:
        if p['team'] == target_team:
            warriors_players.append((p['name'], p['team']))
        else:
            other_players.append((p['name'], p['team']))

    # We want trades that are win-win:
    # Conditions:
    #   new_warriors_pred - baseline_warriors_pred > 0
    #   new_other_team_pred - baseline_other_team_pred > 0
    #   (new_other_team_pred - baseline_other_team_pred) < (new_warriors_pred - baseline_warriors_pred)

    results = []
    for w_player, w_team in warriors_players:
        for o_player, o_team in other_players:
            # Skip if either player is missing baseline predictions for their teams
            if w_team not in baseline_team_preds or o_team not in baseline_team_preds:
                continue

            baseline_other_team_pred = baseline_team_preds[o_team]

            # Perform the swap
            # w_player (from Warriors) <-> o_player (from o_team)
            # After swap: w_player -> o_team, o_player -> Warriors
            swapped_data, swapped_teams_in_order = rebuild_graph_with_swap(original_season_data, standings_data, w_player, o_player)
            swapped_preds = predict(model, swapped_data)
            swapped_team_preds = {team: swapped_preds[i] for i, team in enumerate(swapped_teams_in_order)}

            # Check if target team and other team are in swapped scenario
            if target_team not in swapped_team_preds or o_team not in swapped_team_preds:
                # One of the teams disappeared somehow
                continue

            new_warriors_pred = swapped_team_preds[target_team]
            new_other_team_pred = swapped_team_preds[o_team]

            w_imp = new_warriors_pred - baseline_warriors_prediction
            o_imp = new_other_team_pred - baseline_other_team_pred

            if w_imp > 0 and o_imp > 0 and o_imp < w_imp:
                # A valid win-win trade favoring Warriors
                results.append((w_player, o_player, w_imp, o_imp, swapped_preds, swapped_teams_in_order, o_team))

    # Sort by Warriors improvement descending
    results.sort(key=lambda x: x[2], reverse=True)

    k = 10
    top_k = results[:k]

    print(f"Top {k} Win-Win Trades (Favoring the Warriors):")
    for i, (w_player, o_player, w_imp, o_imp, preds, t_in_order, o_team) in enumerate(top_k, start=1):
        print(f"{i}. Swap {w_player} (GSW) with {o_player} ({o_team})")
        print(f"   Warriors improvement: {w_imp:.4f}")
        print(f"   {o_team} improvement: {o_imp:.4f}")
        print("\nStandings after this trade:")
        print_standings(preds, t_in_order)
        print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    main()
