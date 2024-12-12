import torch
from torch_geometric.nn import GCNConv, global_mean_pool, LayerNorm, BatchNorm
from torch_geometric.loader import DataLoader
import pickle
import copy
import matplotlib.pyplot as plt

class TeamGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(TeamGNN, self).__init__()
        self.norm1 = LayerNorm(hidden_channels)
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.norm2 = LayerNorm(hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.pool = global_mean_pool
        self.relu1 = torch.nn.LeakyReLU()
        self.dropout = torch.nn.Dropout(0.2)
        # self.fc = torch.nn.Linear(in_channels, hidden_channels)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.conv2(x, edge_index)
        x = self.norm2(x)
        x = self.relu1(x)
        
        #x = self.dropout(x)
        
        x = self.pool(x, batch)

        return x


class LeagueGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_teams=30):
        super(LeagueGNN, self).__init__()
        self.team_gnn = TeamGNN(in_channels, hidden_channels)
        self.norm1 = LayerNorm(hidden_channels)
        self.league_conv1 = GCNConv(hidden_channels, hidden_channels)
        self.norm2 = LayerNorm(hidden_channels)
        self.league_conv2 = GCNConv(hidden_channels, hidden_channels)
        self.fc = torch.nn.Linear(hidden_channels, 1)
        # self.skip_fc = torch.nn.Linear(hidden_channels, hidden_channels)
        torch.nn.init.xavier_uniform_(self.fc.weight)
        self.relu1 = torch.nn.LeakyReLU()
        
        self.dropout = torch.nn.Dropout(0.2)
        # torch.nn.init.xavier_uniform_(self.skip_fc.weight)
        

    def forward(self, data):
        team_embeddings = self.team_gnn(data)
        # print(team_embeddings.size())
        league_edge_index = torch.combinations(torch.arange(team_embeddings.size(0)), r=2, with_replacement=False).t()
        x_initial = team_embeddings
        x = self.league_conv1(team_embeddings, league_edge_index)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.league_conv2(x, league_edge_index)
        x = self.norm2(x)
        x = self.relu1(x)
        x = x + x_initial
        #x = self.dropout(x)
        predictions = self.fc(x).squeeze(-1)
        return predictions # torch.softmax(x, dim=1)


def train_model():
    with open("data/league_graphs.pkl", "rb") as f:
        data = pickle.load(f)
        train_data, val_data, test_data = data['train'], data['val'], data['test']
        
    # Load data
    train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False)

    # Initialize model, optimizer, scheduler, and criterion
    model = LeagueGNN(in_channels=train_data[0].x.size(1), hidden_channels=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50, min_lr=0.0001)
    criterion = torch.nn.MSELoss()

    best_model_weights = copy.deepcopy(model.state_dict())
    best_val_loss = float('inf')

    train_losses = []
    val_losses = []
    max_epochs = 300

    for epoch in range(max_epochs):
        # Training phase
        model.train()
        total_loss = 0
        for data in train_data:
            optimizer.zero_grad()
            predictions = model(data)

            loss = criterion(predictions, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_data)
        train_losses.append(avg_train_loss)

        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data in val_loader:
                predictions = model(data)
                loss = criterion(predictions, data.y)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        # Scheduler step
        scheduler.step(avg_val_loss)

        # Save the best model weights
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_weights = copy.deepcopy(model.state_dict())

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, LR: {optimizer.param_groups[0]['lr']:.6f}")

    # Save the best weights
    torch.save(best_model_weights, 'league_predictor_best_weights.pth')

    # Plot training and validation losses
    plt.figure(figsize=(10, 6))
    plt.plot(range(max_epochs), train_losses, label='Training Loss')
    plt.plot(range(max_epochs), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    train_model()
