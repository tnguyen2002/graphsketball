import json
import os
import torch
from torch_geometric.data import Data

# Directory containing the JSON files     
player_data_dir = 'data/player_season_jsons/'  # Adjust the path accordingly
standings_data_dir = 'data/standings_jsons/'  # Adjust the path for standings

def prepare_team_subgraphs(player_data_dir=player_data_dir, standings_data_dir=standings_data_dir, save_data=False):
    # List of seasons
    seasons = [
        '2012_2013', '2013_2014', '2014_2015', '2015_2016',
        '2016_2017', '2017_2018', '2018_2019', '2019_2020',
        '2020_2021', '2021_2022'
    ]

    # Create mappings and data structures
    position_to_idx = {}
    slug_to_name = {}
    team_graphs = []
    team_targets = []

    # Define the list of numeric features to use
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

    for season in seasons:
        # Load player data for the season
        player_file = os.path.join(player_data_dir, f'{season}_advanced_player_season_totals.json')
        standings_file = os.path.join(standings_data_dir, f'{season}_standings.json')

        with open(player_file, 'r') as f:
            season_data = json.load(f)

        with open(standings_file, 'r') as f:
            standings_data = json.load(f)

        # Create team-level graphs
        team_to_players = {}
        for player_data in season_data:
            team = player_data['team']
            player_slug = player_data['slug']
            slug_to_name[player_slug] = player_data['name']

            if team not in team_to_players:
                team_to_players[team] = []

            # Prepare node features for the player
            features = [player_data.get(f, 0.0) for f in numeric_features]
            features_tensor = torch.tensor(features, dtype=torch.float)
            team_to_players[team].append(features_tensor)

        # Prepare subgraphs and targets
        for team_data in standings_data:
            team_name = team_data['team']
            wins = team_data['wins']
            losses = team_data['losses']
            win_percentage = wins / (wins + losses)

            if team_name in team_to_players:
                # Build the graph
                node_features = torch.stack(team_to_players[team_name])
                num_players = node_features.size(0)

                # Fully connected graph within the team
                edge_index = torch.combinations(
                    torch.arange(num_players), r=2, with_replacement=False
                ).t()
                print(team_name)
                print(win_percentage)
                # Create a Data object for the team
                team_graph = Data(x=node_features, edge_index=edge_index, y=torch.tensor([win_percentage], dtype=torch.float))
                print(team_graph)
                # print(team_graph.edge_index)
                team_graphs.append(team_graph)
                print(team_graphs[-1])
                # team_targets.append(win_percentage)

    train_size = int(0.8 * len(team_graphs))
    val_size = int(0.1 * len(team_graphs))
    train_data = team_graphs[:train_size]
    val_data = team_graphs[train_size:train_size + val_size]
    test_data = team_graphs[train_size + val_size:]

    if save_data:
        import pickle
        with open("data/team_graphs.pkl", 'wb') as f:
            pickle.dump({'train': train_data, 'val': val_data, 'test': test_data}, f)

    return train_data, val_data, test_data

prepare_team_subgraphs(player_data_dir, standings_data_dir, save_data=True)