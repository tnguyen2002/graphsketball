import torch
from torch_geometric.data import Data
import json
import os
from torch.distributions import Normal
from copy import deepcopy
import pickle
import itertools
from LeagueModel import LeagueGNN

# Assuming we have the classes and methods defined previously, especially:
# - LeagueGNN
# - TeamGNN
# - The numeric_features list
# - The code structure from prepare_league_graphs (slightly modified to load only one season)

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

def load_season_graph(player_file, standings_file):
    with open(player_file, 'r') as f:
        season_data = json.load(f)

    with open(standings_file, 'r') as f:
        standings_data = json.load(f)

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

    # Build graph
    node_features = []
    edge_index = []
    labels = []
    batch_list = []

    team_start_idx = 0
    teams_in_order = []
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
        if team_edges.size(1) > 0:
            edge_index.append(team_edges)

        batch_list.extend([team_idx] * num_players)

        labels.append(team_win_percentages[team])
        teams_in_order.append(team)

        team_start_idx += num_players

    league_node_features = torch.cat(node_features, dim=0) if len(node_features) > 0 else torch.empty((0,len(numeric_features)))
    if len(edge_index) > 0:
        league_edge_index = torch.cat(edge_index, dim=1)
    else:
        league_edge_index = torch.empty((2,0), dtype=torch.long)
    league_labels = torch.tensor(labels, dtype=torch.float)
    batch_tensor = torch.tensor(batch_list, dtype=torch.long)

    data = Data(x=league_node_features, edge_index=league_edge_index, y=league_labels, batch=batch_tensor)
    return data, teams_in_order, team_win_percentages

def rebuild_graph_with_extra_player(original_data: Data, teams_in_order, team_name, extra_player_features):
    # This function rebuilds the league graph by adding an extra player to a specified team.
    # We'll need to identify which team index the given team_name corresponds to.
    
    # We know batch marks teams for each player node. So we'll reconstruct by splitting according to batch.
    x = original_data.x
    y = original_data.y
    batch = original_data.batch
    edge_index = original_data.edge_index

    # Identify which team index corresponds to the requested team
    if team_name not in teams_in_order:
        raise ValueError(f"{team_name} not found in the team list.")
    team_idx = teams_in_order.index(team_name)

    # Separate players by team
    num_teams = len(teams_in_order)
    team_to_nodes = [[] for _ in range(num_teams)]
    for node_idx, t_idx in enumerate(batch):
        team_to_nodes[t_idx].append(node_idx)

    # Add the new player
    new_player_index = x.size(0)
    new_x = torch.cat([x, extra_player_features.unsqueeze(0)], dim=0)
    # The batch for the new player will be the same as team_idx
    new_batch = torch.cat([batch, torch.tensor([team_idx], dtype=torch.long)], dim=0)

    # Now we have to update edges. We'll create new edges fully connecting the new player to the existing players on that team.
    # The existing team node indices:
    old_team_nodes = team_to_nodes[team_idx]
    new_team_nodes = old_team_nodes + [new_player_index]

    old_team_nodes_tensor = torch.tensor(old_team_nodes, dtype=torch.long)
    new_connections_i = torch.cat([old_team_nodes_tensor, torch.full_like(old_team_nodes_tensor, new_player_index)])
    new_connections_j = torch.cat([torch.full_like(old_team_nodes_tensor, new_player_index), old_team_nodes_tensor])
    new_edges = torch.stack([torch.cat([new_connections_i, new_connections_j]),
                             torch.cat([new_connections_j, new_connections_i])], dim=0)

    # Combine with existing edges
    combined_edge_index = torch.cat([edge_index, new_edges], dim=1)

    # Note: After adding edges, we might want to remove duplicates if any. We'll use unique:
    combined_edge_index = torch.unique(combined_edge_index, dim=1)

    # The label vector (y) and teams_in_order don't change because number of teams is the same, we just added a player.
    # Return the new Data object
    new_data = Data(x=new_x, edge_index=combined_edge_index, y=y, batch=new_batch)
    return new_data

def predict_team_win_percentage(model, data, teams_in_order, target_team):
    model.eval()
    with torch.no_grad():
        predictions = model(data)
    # predictions[i] corresponds to teams_in_order[i]
    # Find target_team index
    if target_team not in teams_in_order:
        raise ValueError(f"Target team {target_team} not in teams_in_order.")
    team_idx = teams_in_order.index(target_team)
    return float(predictions[team_idx].item())

def main():
    # Specify which season to load
    season = "2021_2022"  # Example
    player_file = f"data/player_season_jsons/{season}_advanced_player_season_totals.json"
    standings_file = f"data/standings_jsons/{season}_standings.json"

    # Load original season graph
    original_data, teams_in_order, team_win_percentages = load_season_graph(player_file, standings_file)

    # Load trained model
    model = LeagueGNN(in_channels=original_data.x.size(1), hidden_channels=64)
    model.load_state_dict(torch.load('league_predictor_best_weights.pth', map_location=torch.device('cpu')))
    model.eval()

    # Baseline prediction for the Warriors
    target_team = "GOLDEN STATE WARRIORS"
    baseline_warriors_prediction = predict_team_win_percentage(model, original_data, teams_in_order, target_team)
    print(f"Baseline {target_team} predicted win percentage: {baseline_warriors_prediction}")

    # We want to consider adding players from other teams
    # Let's load the raw player data to iterate through players
    with open(player_file, 'r') as f:
        season_data = json.load(f)

    # Collect all candidates (player name and features) who are not on the target team
    candidates = []
    for player_data in season_data:
        if player_data['team'] == target_team:
            continue
        features = [player_data.get(f, 0.0) for f in numeric_features]
        features_tensor = torch.tensor(features, dtype=torch.float)
        player_name = player_data['name']
        candidates.append((player_name, features_tensor))

    improvements = []
    # Iterate through candidates and measure improvement
    for player_name, feat in candidates:
        # Rebuild graph with this player added
        modified_data = rebuild_graph_with_extra_player(original_data, teams_in_order, target_team, feat)
        new_pred = predict_team_win_percentage(model, modified_data, teams_in_order, target_team)
        improvement = new_pred - baseline_warriors_prediction
        improvements.append((player_name, improvement))

    # Sort by improvement descending
    improvements.sort(key=lambda x: x[1], reverse=True)

    # Get top k players
    k = 10
    top_k = improvements[:k]

    print("Top k players who improve the Warriors the most:")
    for player_name, imp in top_k:
        print(f"{player_name}: +{imp:.4f} improvement in predicted win%")

if __name__ == "__main__":
    main()
