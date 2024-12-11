from create_dataset import prepare_data
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

# Prepare features for Linear Regression
# Aggregate node features for each edge (e.g., element-wise mean)
def prepare_edge_features(data):
    edge_index = data.edge_index
    node_features = data.x.cpu().numpy()
    edge_features = []
    edge_weights = data.edge_attr.cpu().numpy()

    for i in range(edge_index.shape[1]):
        node_i = edge_index[0, i]
        node_j = edge_index[1, i]
        combined_features = np.mean([node_features[node_i], node_features[node_j]], axis=0)  # Element-wise mean
        edge_features.append(combined_features)
    
    return np.array(edge_features), edge_weights

data_dir = 'data/player_season_jsons/'
train_data, val_data, test_data = prepare_data(data_dir)


def construct_player_matrix(data):
    edge_index = data.edge_index
    edge_weights = data.edge_attr.cpu().numpy()
    num_players = data.num_nodes

    # Initialize the matrix with NaNs or zeros (use zeros if you don't use matrix factorization)
    player_matrix = np.full((num_players, num_players), np.nan)

    # Fill the matrix with edge weights
    for idx in range(edge_index.shape[1]):
        i, j = edge_index[:, idx]
        player_matrix[i, j] = edge_weights[idx]
        player_matrix[j, i] = edge_weights[idx]  # Symmetric for undirected graph

    return player_matrix

from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd

def train_collaborative_filtering(player_matrix):
    # Convert player matrix to edge list format
    rows, cols = np.where(~np.isnan(player_matrix))  # Find non-NaN indices
    edge_weights = player_matrix[rows, cols]

    # Prepare edge list for surprise
    edge_list = [(int(row), int(col), float(weight)) for row, col, weight in zip(rows, cols, edge_weights)]

    # Load data into surprise
    reader = Reader(rating_scale=(np.nanmin(player_matrix), np.nanmax(player_matrix)))
    data = Dataset.load_from_df(pd.DataFrame(edge_list, columns=["player1", "player2", "weight"]), reader)

    # Train-test split
    trainset, testset = train_test_split(data, test_size=0.2)

    # Train SVD model
    model = SVD()
    model.fit(trainset)

    # Predict on testset and compute MSE
    predictions = model.test(testset)
    true_ratings = np.array([pred.r_ui for pred in predictions])
    pred_ratings = np.array([pred.est for pred in predictions])
    mse = mean_squared_error(true_ratings, pred_ratings)
    print(f"Collaborative Filtering MSE: {mse:.4f}")

    return model

# Create player-player matrices for train, validation, and test sets
train_matrix = construct_player_matrix(train_data)
val_matrix = construct_player_matrix(val_data)
test_matrix = construct_player_matrix(test_data)

# Train CF model on the train matrix
cf_model = train_collaborative_filtering(train_matrix)

# Evaluate the CF model on the test matrix
rows, cols = np.where(~np.isnan(test_matrix))  # Test indices
test_edge_weights = test_matrix[rows, cols]

cf_predictions = [cf_model.predict(int(row), int(col)).est for row, col in zip(rows, cols)]
cf_mse = mean_squared_error(test_edge_weights, cf_predictions)
print(f"Test MSE for CF Model: {cf_mse:.4f}")
