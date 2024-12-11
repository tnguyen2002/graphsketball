import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.data import DataLoader
from create_dataset import prepare_data


class GATEdgeWeightPredictor(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_heads, num_layers, dropout=0.2):
        super(GATEdgeWeightPredictor, self).__init__()

        self.gat_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()

        # Initial GAT layer
        self.gat_layers.append(GATConv(in_channels, hidden_channels, heads=num_heads, concat=True, dropout=dropout))
        self.norm_layers.append(nn.LayerNorm(hidden_channels * num_heads))

        # Hidden GAT layers
        for _ in range(num_layers - 2):
            self.gat_layers.append(GATConv(hidden_channels * num_heads, hidden_channels, heads=num_heads, concat=True, dropout=dropout))
            self.norm_layers.append(nn.LayerNorm(hidden_channels * num_heads))

        # Output GAT layer
        self.gat_layers.append(GATConv(hidden_channels * num_heads, hidden_channels, heads=1, concat=False, dropout=dropout))
        self.norm_layers.append(nn.LayerNorm(hidden_channels))

        # Edge prediction MLP
        self.edge_predictor = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1)
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # Pass through GAT layers with normalization and skip connections
        for i, gat_layer in enumerate(self.gat_layers):
            residual = x
            x = gat_layer(x, edge_index)
            x = self.norm_layers[i](x)
            if x.shape == residual.shape:  # Apply residual if dimensions match
                x = x + residual
            x = F.elu(x)

        # Compute edge features by concatenating node embeddings
        row, col = edge_index
        edge_features = torch.cat([x[row], x[col]], dim=1)

        # Predict edge weights
        edge_weights = self.edge_predictor(edge_features).squeeze()

        return edge_weights


# Load data
train_data, val_data, test_data = prepare_data('data/player_season_jsons')

# Example usage
in_channels = train_data.x.size(1)  # Number of input features per node
hidden_channels = 128
num_heads = 4
num_layers = 5
model = GATEdgeWeightPredictor(in_channels, hidden_channels, num_heads, num_layers)

# Define optimizer, scheduler, and loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
criterion = nn.MSELoss()

# Training loop
for epoch in range(500):  # Reduce epochs for testing adjustments
    model.train()
    optimizer.zero_grad()

    # Forward pass
    pred_edge_weights = model(train_data)

    # Compute loss
    loss = criterion(pred_edge_weights, train_data.edge_attr)

    # Backward pass
    loss.backward()
    optimizer.step()
    scheduler.step()

    # Print loss for the current epoch
    
    print(f'Epoch {epoch + 1}, MSE Loss: {loss.item()}')

# Evaluation (validation/test)
model.eval()
with torch.no_grad():
    val_pred = model(val_data)
    val_loss = criterion(val_pred, val_data.edge_attr)
    print(f'Validation Loss (MSE): {val_loss.item()}')

    test_pred = model(test_data)
    test_loss = criterion(test_pred, test_data.edge_attr)
    print(f'Test Loss (MSE): {test_loss.item()}')
