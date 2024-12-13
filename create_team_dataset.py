import json
import torch
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
import numpy as np
from pathlib import Path
from utils import visualize_graph

def load_json_data(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def create_player_graph(json_data):
    teams = list(set(player['team'] for player in json_data))
    team_to_idx = {team: idx for idx, team in enumerate(teams)}
    
    players = [(player['name'], player['team']) for player in json_data]
    player_to_idx = {player[0]: idx for idx, player in enumerate(players)}
    
    num_players = len(players)
    num_teams = len(teams)
    node_features = torch.zeros((num_players, num_teams))
    for player, team in players:
        node_features[player_to_idx[player]][team_to_idx[team]] = 1
    
    edge_list = []
    for i, (player1, team1) in enumerate(players):
        for j, (player2, team2) in enumerate(players):
            if i < j and team1 == team2:  
                edge_list.append([i, j])
                edge_list.append([j, i])  
    
    edge_index = torch.tensor(edge_list, dtype=torch.long).t()
    
    data = Data(x=node_features, edge_index=edge_index)
    
    return data, players, teams

def visualize_graph_interactive(data, players, teams):
    """
    Create an interactive visualization of the graph.
    
    Args:
        data: PyTorch Geometric Data object
        players: List of player tuples (name, team)
        teams: List of team names
    """

    G = to_networkx(data, to_undirected=True)
    

    team_colors = plt.cm.rainbow(np.linspace(0, 1, len(teams)))
    team_to_color = dict(zip(teams, team_colors))
    
    node_colors = [team_to_color[players[node][1]] for node in G.nodes()]
    plt.figure(figsize=(15, 15))
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    def update_graph(event=None):
        plt.clf()  
        

        nx.draw(G, pos, 
               node_color=node_colors,
               node_size=1000,
               with_labels=True,
               labels={i: players[i][0] for i in G.nodes()},
               font_size=8)
        

        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                    label=team,
                                    markerfacecolor=team_to_color[team], 
                                    markersize=10)
                          for team in teams]
        plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
        plt.draw()
    

    update_graph()

    
    plt.show()

def save_graph(data, save_path):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, save_path)
    print(f"Graph data saved to: {save_path}")

def load_graph(load_path):
    return torch.load(load_path)

def process_player_graph(json_path, save_path=None):
    """
    Process player data from JSON file and return graph data.
    
    Args:
        json_path (str or Path): Path to the JSON file
        save_path (str or Path, optional): Path to save the graph data
    Returns:
        tuple: (data, players, teams)
    """
   
    json_path = Path(json_path)
    json_data = load_json_data(json_path)
    data, players, teams = create_player_graph(json_data)

    if save_path:
        save_graph(data, save_path)
    
    return data, players, teams

if __name__ == "__main__":

    save_path_train = Path("output/player_graphs/train.pt")
    save_path_val = Path("output/player_graphs/val.pt")
    save_path_test = Path("output/player_graphs/test.pt")

    train_path = Path("data/player_season_jsons/2019_2020_advanced_player_season_totals.json")
    val_path = Path("data/player_season_jsons/2020_2021_advanced_player_season_totals.json")
    test_path = Path("data/player_season_jsons/2021_2022_advanced_player_season_totals.json")

    train_data, train_players, train_teams = process_player_graph(train_path, save_path_train)
    val_data, val_players, val_teams = process_player_graph(val_path, save_path_val)
    test_data, test_players, test_teams = process_player_graph(test_path, save_path_test)

    train_player_to_node = {player[0]: idx for idx, player in enumerate(train_players)}
    train_slug_to_name = {player[0]: player[0] for player in train_players}  # Assuming slug == name

    visualize_graph(
        train_data, 
        player_to_node=train_player_to_node, 
        slug_to_name=train_slug_to_name, 
        title="Training Graph Visualization"
    )



