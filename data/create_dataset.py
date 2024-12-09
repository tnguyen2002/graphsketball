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
data_dir = './player_season_jsons/'  # Adjust the path accordingly

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
    "is_combined_totals"  # Include as numeric feature (0 or 1)
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
# visualize_graph(data, player_to_node, slug_to_name, title="Full Graph")

# Use RandomLinkSplit to split edges into train/val/test sets
transform = RandomLinkSplit(
    num_val=0.05,        # 5% for validation
    num_test=0.1,        # 10% for testing
    is_undirected=True,
    add_negative_train_samples=True
)
train_data, val_data, test_data = transform(data)

# Move data to the appropriate device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_data = train_data.to(device)
val_data = val_data.to(device)
test_data = test_data.to(device)

# Define the GCN model for link prediction
class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, dropout = 0.5):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.norm1 = BatchNorm(2 * out_channels)
        self.conv2 = GCNConv(2 * out_channels, out_channels)
        self.norm2 = BatchNorm(out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.norm1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = self.norm2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x

class Net(torch.nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.5):
        super(Net, self).__init__()
        self.encoder = GCNEncoder(in_channels, out_channels, dropout)

    def forward(self, x, edge_index):
        z = self.encoder(x, edge_index)
        return z

    def decode(self, z, edge_index):
        # Compute dot product between node embeddings
        return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)

# Training and evaluation functions
def train(model, optimizer, train_data):
    model.train()
    optimizer.zero_grad()
    z = model(train_data.x, train_data.edge_index)

    edge_index = train_data.edge_label_index
    labels = train_data.edge_label.float()

    logits = model.decode(z, edge_index)
    loss = F.mse_loss(logits, labels)  # Use MSE for edge weight prediction
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return loss.item()

def test(model, data):
    model.eval()
    with torch.no_grad():
        z = model(data.x, data.edge_index)

    edge_index = data.edge_label_index
    labels = data.edge_label.float()

    logits = model.decode(z, edge_index).cpu()
    labels = labels.cpu()

    # Compute mean squared error as metric for edge weight prediction
    mse = F.mse_loss(logits, labels)

    return mse.item()

# Initialize the model, optimizer
model = Net(in_channels=data.num_features, out_channels=128, dropout=0.5).to(device)
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)

# # Training loop
# num_epochs = 3000
# best_val_mse = float('inf')

# for epoch in range(1, num_epochs + 1):
#     loss = train(model, optimizer, train_data)s
    
#     if epoch % 10 == 0:
#         val_mse = test(model, val_data)
#         test_mse = test(model, test_data)
#         print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, '
#               f'Val MSE: {val_mse:.4f}, Test MSE: {test_mse:.4f}')
        
#         # Optionally, save the best model based on validation MSE
#         if val_mse < best_val_mse:
#             best_val_mse = val_mse
#             torch.save(model.state_dict(), 'best_model.pth')

# print(f'Best Val MSE: {best_val_mse:.4f}')

import matplotlib.pyplot as plt

# Initialize lists to store training loss, validation MSE, and test MSE over epochs
train_loss_history = []
val_mse_history = []
test_mse_history = []

# Example training loop to populate these lists
num_epochs = 500
for epoch in range(1, num_epochs + 1):
    loss = train(model, optimizer, train_data)
    train_loss_history.append(loss)
    
    if epoch % 10 == 0:  # Assuming you want to evaluate every 10 epochs
        val_mse = test(model, val_data)
        test_mse = test(model, test_data)
        val_mse_history.append(val_mse)
        test_mse_history.append(test_mse)

        print(f'Epoch {epoch}: Training Loss: {loss:.4f}, Val MSE: {val_mse:.4f}, Test MSE: {test_mse:.4f}')

# Plot training loss over epochs
plt.figure(figsize=(12, 6))
plt.plot(train_loss_history, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Training Loss')
plt.yscale('log')  # Set y-axis to log scale
plt.title('Training Loss over Epochs')
plt.legend()
plt.show()

# Plot validation and test MSE over epochs
plt.figure(figsize=(12, 6))
epochs = range(10, num_epochs + 1, 10)  # Every 10th epoch for val/test MSE
plt.plot(epochs, val_mse_history, label='Validation MSE')
plt.plot(epochs, test_mse_history, label='Test MSE')
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error (MSE)')
plt.yscale('log')
plt.title('Validation and Test MSE over Epochs')
plt.legend()
plt.show()

import pickle

# Save the trained model
torch.save(model.state_dict(), 'best_model.pth')

# Save the mappings and node features
with open('data_mappings.pkl', 'wb') as f:
    pickle.dump({
        'player_to_node': player_to_node,
        'slug_to_name': slug_to_name,
        'node_features': node_features,
    }, f)