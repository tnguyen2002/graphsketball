import json
import torch
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
import numpy as np
from pathlib import Path

def load_json_data(file_path):
    """Load JSON data from a file path."""
    with open(file_path, 'r') as f:
        return json.load(f)

def create_player_graph(json_data):
    # Extract unique teams and create team-to-index mapping
    teams = list(set(player['team'] for player in json_data))
    team_to_idx = {team: idx for idx, team in enumerate(teams)}
    
    # Create player-to-index mapping and feature matrix
    players = [(player['name'], player['team']) for player in json_data]
    player_to_idx = {player[0]: idx for idx, player in enumerate(players)}
    
    # Create node features (one-hot encoding of teams)
    num_players = len(players)
    num_teams = len(teams)
    node_features = torch.zeros((num_players, num_teams))
    for player, team in players:
        node_features[player_to_idx[player]][team_to_idx[team]] = 1
    
    # Create edges between players on the same team
    edge_list = []
    for i, (player1, team1) in enumerate(players):
        for j, (player2, team2) in enumerate(players):
            if i < j and team1 == team2:  # Only create one edge between each pair
                edge_list.append([i, j])
                edge_list.append([j, i])  # Add both directions for undirected graph
    
    edge_index = torch.tensor(edge_list, dtype=torch.long).t()
    
    # Create PyG Data object
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
    # Convert to networkx
    G = to_networkx(data, to_undirected=True)
    
    # Create a color map based on teams
    team_colors = plt.cm.rainbow(np.linspace(0, 1, len(teams)))
    team_to_color = dict(zip(teams, team_colors))
    
    node_colors = [team_to_color[players[node][1]] for node in G.nodes()]
    
    # Set up the plot with a larger figure size
    plt.figure(figsize=(15, 15))
    
    # Create initial spring layout
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    def update_graph(event=None):
        plt.clf()  # Clear the current figure
        
        # Draw the network
        nx.draw(G, pos, 
               node_color=node_colors,
               node_size=1000,
               with_labels=True,
               labels={i: players[i][0] for i in G.nodes()},
               font_size=8)
        
        # Add legend
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                    label=team,
                                    markerfacecolor=team_to_color[team], 
                                    markersize=10)
                          for team in teams]
        plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
        plt.draw()
    
    def on_click(event):
        if event.inaxes:
            # Get the closest node to the click
            click_pos = np.array([event.xdata, event.ydata])
            distances = {node: np.linalg.norm(np.array(pos[node]) - click_pos) 
                       for node in G.nodes()}
            closest_node = min(distances.keys(), key=lambda n: distances[n])
            
            # Print information about the clicked node
            player_name, team = players[closest_node]
            print(f"\nClicked node info:")
            print(f"Player: {player_name}")
            print(f"Team: {team}")
            
            # Highlight connections
            neighbors = list(G.neighbors(closest_node))
            print("Connected to:")
            for neighbor in neighbors:
                print(f"- {players[neighbor][0]} ({players[neighbor][1]})")

    def on_key(event):
        nonlocal pos
        if event.key == 'r':  # Press 'r' to regenerate layout
            pos = nx.spring_layout(G, k=1, iterations=50)
            update_graph()
        elif event.key == 'h':  # Press 'h' for help
            print("\nInteractive controls:")
            print("- Click on a node to see player and connection information")
            print("- Press 'r' to regenerate the layout")
            print("- Press 'h' to see this help message")
            print("- Close the window to exit")

    # Connect event handlers
    plt.connect('button_press_event', on_click)
    plt.connect('key_press_event', on_key)
    
    # Initial draw
    update_graph()
    
    # Print initial instructions
    print("\nInteractive Graph Controls:")
    print("- Click on a node to see player and connection information")
    print("- Press 'r' to regenerate the layout")
    print("- Press 'h' to see this help message")
    print("- Close the window to exit")
    
    plt.show()

def save_graph(data, save_path):
    """Save PyTorch Geometric Data object to specified path."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, save_path)
    print(f"Graph data saved to: {save_path}")

def load_graph(load_path):
    """Load PyTorch Geometric Data object from specified path."""
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
    # Convert paths to Path objects
    json_path = Path(json_path)
    
    # Load and process data
    json_data = load_json_data(json_path)
    data, players, teams = create_player_graph(json_data)
    
    # Save graph if save_path is provided
    if save_path:
        save_graph(data, save_path)
    
    return data, players, teams

if __name__ == "__main__":
    # Define input and output paths


    save_path_train = Path("output/player_graphs/train.pt")
    save_path_val = Path("output/player_graphs/val.pt")
    save_path_test= Path("output/player_graphs/test.pt")
    train_path = Path("data/player_season_jsons/2019_2020_advanced_player_season_totals.json")
    val_path = Path("data/player_season_jsons/2020_2021_advanced_player_season_totals.json")
    test_path = Path("data/player_season_jsons/2021_2022_advanced_player_season_totals.json")


    # input_path = Path("data/player_season_jsons/2021_2022_advanced_player_season_totals.json")

    
    # Process player data and save the graph
    train_data, train_players, train_teams = process_player_graph(train_path, save_path_train)
    val_data, val_players, val_teams = process_player_graph(val_path, save_path_val)
    test_data, test_players, test_teams = process_player_graph(test_path, save_path_test)




