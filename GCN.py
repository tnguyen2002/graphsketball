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
from create_dataset import prepare_data

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

data_dir = 'data/player_season_jsons/'  # Adjust the path accordingly

train_data, val_data, test_data = prepare_data(data_dir)

# Initialize the model, optimizer
model = Net(in_channels=train_data.num_features, out_channels=128, dropout=0.5).to(train_data.x.device)
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