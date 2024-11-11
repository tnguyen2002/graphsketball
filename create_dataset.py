import json
import os
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.nn import GCNConv
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

player_season_to_node = {}
node_features = []
edges = []
edge_weights = []

node_idx = 0

# Mapping from node index to (player_slug, season)
node_idx_to_player_season = {}

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

# Second pass: Process data and build the graph
slug_to_name = {}
for season in seasons:
    filename = os.path.join(data_dir, f'{season}_advanced_player_season_totals.json')
    with open(filename, 'r') as f:
        season_data = json.load(f)
    
    # Build mappings for this season
    team_to_players = {}
    for player_data in season_data:
        player_slug = player_data['slug']
        team = player_data['team']
        player_name = player_data['name']
        slug_to_name[player_slug] = player_name
        
        # Create a unique key for each player-season
        player_season_key = (player_slug, season)
        if player_season_key not in player_season_to_node:
            # Assign a new node index
            player_season_to_node[player_season_key] = node_idx
            node_idx_to_player_season[node_idx] = player_season_key
            node_idx += 1
            
            # Collect node features
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
        
        node_index = player_season_to_node[player_season_key]
        
        if team not in team_to_players:
            team_to_players[team] = []
        team_to_players[team].append((node_index, player_data))
    
    # Create edges between players on the same team
    for team_players in team_to_players.values():
        num_players = len(team_players)
        for i in range(num_players):
            for j in range(i + 1, num_players):
                player_i, data_i = team_players[i]
                player_j, data_j = team_players[j]
                
                # Define the edge weight as sum of box_plus_minus for chemistry
                weight = data_i["box_plus_minus"] + data_j["box_plus_minus"]
                
                # Add edges and weights
                edges.append((player_i, player_j))
                edges.append((player_j, player_i))  # Undirected graph
                edge_weights.append(weight)
                edge_weights.append(weight)  # Duplicate weight for both directions

# Create edge index tensor
edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()  # Shape [2, num_edges]
edge_weight = torch.tensor(edge_weights, dtype=torch.float)  # Shape [num_edges]

# Stack node features into a single tensor
x = torch.stack(node_features)  # Shape [num_nodes, num_features]

# Create Data object
data = Data(x=x, edge_index=edge_index, edge_attr=edge_weight)
visualize_graph(data, node_idx_to_player_season, slug_to_name, title="Full Graph")

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
    def __init__(self, in_channels, out_channels):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv2 = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

class Net(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Net, self).__init__()
        self.encoder = GCNEncoder(in_channels, out_channels)

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
    loss = F.binary_cross_entropy_with_logits(logits, labels)
    loss.backward()
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

    # Compute metrics
    preds = torch.sigmoid(logits)
    auc = roc_auc_score(labels.numpy(), preds.numpy())
    ap = average_precision_score(labels.numpy(), preds.numpy())

    return auc, ap

# Initialize the model, optimizer
# model = Net(in_channels=data.num_features, out_channels=128).to(device)
# optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)

# # Training loop
# num_epochs = 200
# best_val_auc = 0
# best_test_ap = 0

# for epoch in range(1, num_epochs + 1):
#     loss = train(model, optimizer, train_data)
    
#     if epoch % 10 == 0:
#         val_auc, val_ap = test(model, val_data)
#         test_auc, test_ap = test(model, test_data)
#         print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, '
#               f'Val AUC: {val_auc:.4f}, Val AP: {val_ap:.4f}, '
#               f'Test AUC: {test_auc:.4f}, Test AP: {test_ap:.4f}')
        
#         # Optionally, you can save the best model based on validation AUC
#         if val_auc > best_val_auc:
#             best_val_auc = val_auc
#             best_test_ap = test_ap
#             torch.save(model.state_dict(), 'best_model.pth')

# print(f'Best Val AUC: {best_val_auc:.4f}, Corresponding Test AP: {best_test_ap:.4f}')
