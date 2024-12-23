import json
import os
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.nn import GCNConv, BatchNorm, LayerNorm
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score
from utils import visualize_graph, get_data
import matplotlib.pyplot as plt

class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout = 0.5, num_layers=1):
        super(GCNEncoder, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.layers.append(GCNConv(hidden_channels, hidden_channels))
        self.layers.append(GCNConv(hidden_channels, out_channels))
        self.norms = torch.nn.ModuleList([BatchNorm(hidden_channels) for _ in range(num_layers)])
        self.dropout = dropout

    def forward(self, x, edge_index):
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x, edge_index)
            x = self.norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layers[-1](x, edge_index)
        return x

class Net(torch.nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.5):
        super(Net, self).__init__()
        self.encoder = GCNEncoder(in_channels, out_channels, out_channels, dropout)

    def forward(self, x, edge_index):
        z = self.encoder(x, edge_index)
        return z

    def decode(self, z, edge_index):
        return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)

def train(model, optimizer, train_data, scheduler=None):
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
    if scheduler is not None:
        scheduler.step(loss)
    return loss.item()

def test(model, data):
    model.eval()
    with torch.no_grad():
        z = model(data.x, data.edge_index)

    edge_index = data.edge_label_index
    labels = data.edge_label.float()

    logits = model.decode(z, edge_index).cpu()
    labels = labels.cpu()

    mse = F.mse_loss(logits, labels)

    return mse.item()

data_dir = 'data/player_season_jsons/'  # Adjust the path accordingly

train_data, val_data, test_data = get_data()

model = Net(in_channels=train_data.num_features, out_channels=128, dropout=0.5).to(train_data.x.device)
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.75, patience=5)


# Initialize lists to store training loss, validation MSE, and test MSE over epochs
train_loss_history = []
val_mse_history = []
test_mse_history = []

# Example training loop to populate these lists
num_epochs = 300
for epoch in range(1, num_epochs + 1):
    loss = train(model, optimizer, train_data, scheduler)
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
