import torch
import pickle
from torch_geometric.nn import GCNConv, BatchNorm
import torch.nn.functional as F

# Define the GCN model structure (must match the training structure)
class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.5):
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
        return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)

# Load mappings and node features
with open('data_mappings.pkl', 'rb') as f:
    mappings = pickle.load(f)

player_to_node = mappings['player_to_node']
slug_to_name = mappings['slug_to_name']
node_features = mappings['node_features']

# Determine the input feature size dynamically
in_channels = len(node_features[0])  # Number of features per node
out_channels = 128  # Set this to the same value used during training

# Initialize model with correct input feature size
model = Net(in_channels=in_channels, out_channels=out_channels, dropout=0.5)
model.load_state_dict(torch.load('best_model.pth', weights_only=True))
model.eval()

# Convert node features to tensor
x = torch.stack(node_features)

# Generate embeddings for all nodes
with torch.no_grad():
    z = model.encoder(x, torch.tensor([[], []], dtype=torch.long))

# Function to predict edge weight between two players
def predict_edge_weight(player1, player2):
    node1 = player_to_node.get(player1)
    node2 = player_to_node.get(player2)
    
    if node1 is None or node2 is None:
        print("One or both players not found.")
        return None
    
    with torch.no_grad():
        edge_index = torch.tensor([[node1], [node2]], dtype=torch.long)
        score = model.decode(z, edge_index)
    return score.item()

# Example usage: predict compatibility score between two players
player1 = "jamesle01"  # Replace with actual player slug
player2 = "curryst01"  # Replace with actual player slug
score = predict_edge_weight(player1, player2)
if score is not None:
    print(f"Predicted compatibility score between {slug_to_name[player1]} and {slug_to_name[player2]}: {score}")

# Function to find top N unplayed pairs with highest predicted compatibility scores
import itertools

def find_top_unplayed_pairs(top_n=100):
    existing_edges = set((i.item(), j.item()) for i, j in torch.tensor([[], []], dtype=torch.long).t())
    
    potential_pairs = []
    for player1, player2 in itertools.combinations(player_to_node.keys(), 2):
        node1 = player_to_node[player1]
        node2 = player_to_node[player2]
        
        if (node1, node2) not in existing_edges and (node2, node1) not in existing_edges:
            score = predict_edge_weight(player1, player2)
            if score is not None:
                potential_pairs.append((score, player1, player2))
    
    # Sort pairs by score and return the top results
    potential_pairs.sort(reverse=True, key=lambda x: x[0])
    return potential_pairs[:top_n]

# Display the top 10 predicted best unplayed pairs
top_pairs = find_top_unplayed_pairs(100)
for score, player1, player2 in top_pairs:
    print(f"{slug_to_name[player1]} and {slug_to_name[player2]}: Predicted score = {score}")
