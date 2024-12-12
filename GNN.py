import torch
from torch_geometric.nn import GCNConv, global_mean_pool, LayerNorm
from torch_geometric.loader import DataLoader
# from new_prepare_data import prepare_team_subgraphs
import pickle
   

class TeamWinPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(TeamWinPredictor, self).__init__()
        self.norm1 = LayerNorm(hidden_channels)
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.norm2 = LayerNorm(hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.fc = torch.nn.Linear(hidden_channels, 1)  # Output is a single scalar (win percentage)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # GCN layers
        x = self.conv1(x, edge_index)
        x = self.norm1(x)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = self.norm2(x)
        x = torch.relu(x)

        # Global pooling
        x = global_mean_pool(x, batch)

        # Final linear layer
        x = self.fc(x)
        return torch.sigmoid(x).squeeze()  # Predict values between 0 and 1

# Load data
def train():
    with open("data/team_graphs.pkl", 'rb') as f:
        saved_data = pickle.load(f)
        train_data, val_data, test_data = saved_data['train'], saved_data['val'], saved_data['test']
        
    print(train_data)

    # Prepare data loaders
    train_loader = DataLoader(train_data, batch_size=10, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False)

    # Model, optimizer, and loss
    model = TeamWinPredictor(in_channels=train_data[0].x.size(1), hidden_channels=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.75, patience=10)
    criterion = torch.nn.MSELoss()
    def initialize_weights(m):
        if isinstance(m, (torch.nn.Linear)):
            torch.nn.init.xavier_uniform_(m.weight)

    model.apply(initialize_weights)


    # Training loop
    for epoch in range(300):
        model.train()
        total_loss = 0
        for data in train_loader:
            optimizer.zero_grad()
            output = model(data)
            # print(output)
            loss = criterion(output, data.y)
            # scheduler.step(loss)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data in val_loader:
                output = model(data)
                loss = criterion(output, data.y)
                val_loss += loss.item()
                
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Training Loss: {total_loss / len(train_loader)}, Validation Loss: {val_loss / len(val_loader)}')
        #print(f'Validation Loss: {val_loss / len(val_loader)}')
        
    torch.save(model.state_dict(), 'team_win_predictor_weights.pth')
    
if __name__ == "__main__":
    train()