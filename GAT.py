import json
import os
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.nn import GATConv
from torch_geometric.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
from create_dataset import prepare_data
from tqdm import tqdm

# Define the GAT model class
class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=1, dropout=0.3):
        super(GAT, self).__init__()
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.gat2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout)

    def forward(self, x, edge_index, edge_weight=None):
        x = self.gat1(x, edge_index, edge_weight)
        x = F.elu(x)
        x = self.gat2(x, edge_index, edge_weight)
        return x

# Initialize the model
in_channels = 64  # Placeholder, adjust according to actual data
hidden_channels = 64
out_channels = 1
heads = 8
dropout = 0.3

def train_gat():
    data_dir = '.data/player_season_jsons/'  # Adjust the path accordingly

    train_data, val_data, test_data = prepare_data(data_dir)
    model = GAT(train_data.num_features, hidden_channels, out_channels, heads, dropout).to(train_data.x.device)

    # Define optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    # Training loop
    def train():
        model.train()
        optimizer.zero_grad()
        out = model(train_data.x, train_data.edge_index, train_data.edge_attr)

        # Get predictions for positive and negative edges
        pos_edge_index = train_data.pos_edge_label_index
        neg_edge_index = train_data.neg_edge_label_index

        pos_preds = out[pos_edge_index[0]] * out[pos_edge_index[1]]
        neg_preds = out[neg_edge_index[0]] * out[neg_edge_index[1]]

        pos_labels = torch.ones(pos_preds.shape[0], device=train_data.x.device)
        neg_labels = torch.zeros(neg_preds.shape[0], device=train_data.x.device)

        preds = torch.cat([pos_preds, neg_preds], dim=0)
        labels = torch.cat([pos_labels, neg_labels], dim=0)

        loss = loss_fn(preds, labels)
        loss.backward()
        optimizer.step()
        return loss.item()

    # Validation and testing loop
    def evaluate(data):
        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index, data.edge_attr)

            pos_edge_index = data.pos_edge_label_index
            neg_edge_index = data.neg_edge_label_index

            pos_preds = (out[pos_edge_index[0]] * out[pos_edge_index[1]]).sum(dim=-1)
            neg_preds = (out[neg_edge_index[0]] * out[neg_edge_index[1]]).sum(dim=-1)

            preds = torch.cat([pos_preds, neg_preds], dim=0).sigmoid()
            labels = torch.cat([
                torch.ones(pos_preds.shape[0], device=data.x.device),
                torch.zeros(neg_preds.shape[0], device=data.x.device)
            ])

            auc = roc_auc_score(labels.cpu(), preds.cpu())
            ap = average_precision_score(labels.cpu(), preds.cpu())
            return auc, ap

    # Training and evaluation loop
    num_epochs = 50
    for epoch in tqdm(range(1, num_epochs + 1), desc="Training Progress"):
        loss = train()
        train_auc, train_ap = evaluate(train_data)
        val_auc, val_ap = evaluate(val_data)

        print(f"Epoch {epoch:02d}, Loss: {loss:.4f}, Train AUC: {train_auc:.4f}, Train AP: {train_ap:.4f}, Val AUC: {val_auc:.4f}, Val AP: {val_ap:.4f}")

    # Evaluate on test data
    test_auc, test_ap = evaluate(test_data)
    print(f"Test AUC: {test_auc:.4f}, Test AP: {test_ap:.4f}")

    # Save the model
    model_save_path = "GAT_Model.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    return model


model = train_gat()