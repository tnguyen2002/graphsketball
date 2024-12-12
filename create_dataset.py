import json
import os
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.nn import GCNConv, BatchNorm, LayerNorm
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score
from utils import visualize_graph

# Directory containing the JSON files     
data_dir = 'data/player_season_jsons/'  # Adjust the path accordingly

# Function to prepare data
def prepare_data(data_dir = data_dir, save_data=False):
    # List of seasons
    seasons = [
        '2012_2013', '2013_2014', '2014_2015', '2015_2016',
        '2016_2017', '2017_2018', '2018_2019', '2019_2020',
        '2020_2021', '2021_2022'
    ]

    # Create mappings and data structures
    player_to_node = {}
    node_features = []
    edges = {}
    edge_weights = {}

    node_idx = 0
    positions_set = set()

    # First pass: Collect all possible positions
    for season in seasons:
        filename = os.path.join(data_dir, f'{season}_advanced_player_season_totals.json')
        with open(filename, 'r') as f:
            season_data = json.load(f)

        for player_data in season_data:
            positions = player_data.get('positions', [])
            for pos in positions:
                positions_set.add(pos)

    # Create a mapping from position to index
    position_to_idx = {pos: idx for idx, pos in enumerate(sorted(positions_set))}
    num_positions = len(position_to_idx)
    
    '''
    Example player data:
    {
        "age": 36,
        "assist_percentage": 41.8,
        "block_percentage": 1.5,
        "box_plus_minus": 8.1,
        "defensive_box_plus_minus": 2.3,
        "defensive_rebound_percentage": 23.6,
        "defensive_win_shares": 2.6,
        "free_throw_attempt_rate": 0.31,
        "games_played": 45,
        "is_combined_totals": false,
        "minutes_played": 1504,
        "name": "LeBron James",
        "offensive_box_plus_minus": 5.9,
        "offensive_rebound_percentage": 2.2,
        "offensive_win_shares": 3.0,
        "player_efficiency_rating": 24.2,
        "positions": [
            "POINT GUARD"
        ],
        "slug": "jamesle01",
        "steal_percentage": 1.6,
        "team": "LOS ANGELES LAKERS",
        "three_point_attempt_rate": 0.346,
        "total_rebound_percentage": 12.9,
        "true_shooting_percentage": 0.602,
        "turnover_percentage": 15.2,
        "usage_percentage": 31.9,
        "value_over_replacement_player": 3.8,
        "win_shares": 5.6,
        "win_shares_per_48_minutes": 0.179
    }
    '''

    # Define the list of numeric features to use
    numeric_features = [
        "age",
        "assist_percentage",
        "block_percentage",
        "box_plus_minus",
        "defensive_box_plus_minus",
        "defensive_rebound_percentage",
        "defensive_win_shares",
        "free_throw_attempt_rate",
        "games_played",
        "minutes_played",
        "offensive_box_plus_minus",
        "offensive_rebound_percentage",
        "offensive_win_shares",
        "player_efficiency_rating",
        "steal_percentage",
        "three_point_attempt_rate",
        "total_rebound_percentage",
        "true_shooting_percentage",
        "turnover_percentage",
        "usage_percentage",
        "value_over_replacement_player",
        "win_shares",
        "win_shares_per_48_minutes",
        "is_combined_totals"
    ]

    # Initialize player features and edges across seasons
    slug_to_name = {}
    for season in seasons:
        filename = os.path.join(data_dir, f'{season}_advanced_player_season_totals.json')
        with open(filename, 'r') as f:
            season_data = json.load(f)

        team_to_players = {}  # Temporary dictionary to store team composition per season
        for player_data in season_data:
            player_slug = player_data['slug']
            team = player_data['team']
            player_name = player_data['name']
            slug_to_name[player_slug] = player_name

            # Assign a node index for each unique player
            if player_slug not in player_to_node:
                player_to_node[player_slug] = node_idx
                node_idx += 1

                # Collect node features for each player (initial setup with one feature set per player)
                features = []
                for key in numeric_features:
                    value = player_data.get(key, 0.0)
                    if isinstance(value, bool):
                        value = float(value)
                    features.append(value)
                # Encode positions as one-hot vector
                positions = player_data.get('positions', [])
                position_one_hot = [0.0] * num_positions
                for pos in positions:
                    idx = position_to_idx[pos]
                    position_one_hot[idx] = 1.0
                features.extend(position_one_hot)
                # Convert features to a tensor
                features_tensor = torch.tensor(features, dtype=torch.float)
                node_features.append(features_tensor)

            # Map team to list of player indices and data
            node_index = player_to_node[player_slug]
            if team not in team_to_players:
                team_to_players[team] = []
            team_to_players[team].append((node_index, player_data))

        # Add edges between players on the same team with cumulative box_plus_minus
        for team_players in team_to_players.values():
            num_players = len(team_players)
            for i in range(num_players):
                for j in range(i + 1, num_players):
                    player_i, data_i = team_players[i]
                    player_j, data_j = team_players[j]

                    # Calculate weight as sum of box_plus_minus for this season
                    season_weight = data_i["box_plus_minus"] + data_j["box_plus_minus"]

                    # Update cumulative edge weights and counts
                    if (player_i, player_j) in edges:
                        edges[(player_i, player_j)] += season_weight
                        edge_weights[(player_i, player_j)] += 1  # Track seasons played together
                    else:
                        edges[(player_i, player_j)] = season_weight
                        edge_weights[(player_i, player_j)] = 1

    # Convert cumulative edge weights to averages
    edge_index_list = []
    edge_weight_list = []
    for (player_i, player_j), total_weight in edges.items():
        # Average the weight over the seasons the players played together
        avg_weight = total_weight / edge_weights[(player_i, player_j)]
        edge_index_list.append((player_i, player_j))
        edge_index_list.append((player_j, player_i))  # Duplicate for undirected graph
        edge_weight_list.append(avg_weight)
        edge_weight_list.append(avg_weight)

    # Create edge index tensor and edge weight tensor
    edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()  # Shape [2, num_edges]
    edge_weight = torch.tensor(edge_weight_list, dtype=torch.float)  # Shape [num_edges]

    # Stack node features into a single tensor
    x = torch.stack(node_features)  # Shape [num_nodes, num_features]

    # Create Data object with edge weights
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_weight)

    # Use RandomLinkSplit to split edges into train/val/test sets
    transform = RandomLinkSplit(
        num_val=0.05,        # 5% for validation
        num_test=0.1,        # 10% for testing
        is_undirected=True,
        add_negative_train_samples=True
    )
    train_data, val_data, test_data = transform(data)

    if save_data:
        import pickle
        
        with open("data/graph_data.pkl", 'wb') as f:
            pickle.dump({
                'graph_data': {'train': train_data, 'val': val_data, 'test': test_data},
                'mappings': {'player_to_node': player_to_node, 'slug_to_name': slug_to_name}
            }, f)

    return train_data, val_data, test_data

prepare_data(save_data=True)