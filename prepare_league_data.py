import json
import os
import torch
from torch_geometric.data import Data
import pickle

# Directories containing JSON files
player_data_dir = 'data/player_season_jsons/'
standings_data_dir = 'data/standings_jsons/'

def prepare_league_graphs(player_data_dir=player_data_dir, standings_data_dir=standings_data_dir, save_data=False):
    seasons = [f"{year - 1}_{year}" for year in range(2000, 2025)]

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

    league_graphs = []
    for season in seasons:
        player_file = os.path.join(player_data_dir, f'{season}_advanced_player_season_totals.json')
        standings_file = os.path.join(standings_data_dir, f'{season}_standings.json')

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

        node_features = []
        edge_index = []
        labels = []
        batch_list = []

        team_start_idx = 0
        for team_idx, (team, players) in enumerate(team_to_players.items()):
            if team not in team_win_percentages:
                continue

            num_players = len(players)
            team_node_features = torch.stack(players)
            # print(team, team_node_features)
            # print("=====================================")
            node_features.append(team_node_features)

            team_edges = torch.combinations(
                torch.arange(team_start_idx, team_start_idx + num_players),
                r=2,
                with_replacement=False
            ).t()
            edge_index.append(team_edges)
            
            batch_list.extend([team_idx] * num_players)

            labels.append(team_win_percentages[team])
            team_start_idx += num_players

        # print(len(node_features), node_features[0].size())
        league_node_features = torch.cat(node_features, dim=0)
        # print(league_node_features.size())
        league_edge_index = torch.cat(edge_index, dim=1)
        league_labels = torch.tensor(labels, dtype=torch.float)
        
        batch_tensor = torch.tensor(batch_list, dtype=torch.long)
        
        league_graphs.append(Data(x=league_node_features, edge_index=league_edge_index, y=league_labels, batch=batch_tensor))

    train_size = int(0.9 * len(league_graphs))
    val_size = int(0.05 * len(league_graphs))
    train_data = league_graphs[:train_size]
    val_data = league_graphs[train_size:train_size + val_size]
    test_data = league_graphs[train_size + val_size:]

    if save_data:
        with open("data/league_graphs.pkl", 'wb') as f:
            pickle.dump({'train': train_data, 'val': val_data, 'test': test_data}, f)

    return train_data, val_data, test_data

prepare_league_graphs(save_data=True)