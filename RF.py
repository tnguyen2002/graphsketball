from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from create_dataset import prepare_data
import numpy as np

def evaluate_model(model, edge_features, edge_weights, dataset_name=""):
    predictions = model.predict(edge_features)
    mse = mean_squared_error(edge_weights, predictions)
    print(f"{dataset_name} MSE: {mse:.4f}")
    return mse

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

# Prepare train, validation, and test features
train_edge_features, train_edge_weights = prepare_edge_features(train_data)
val_edge_features, val_edge_weights = prepare_edge_features(val_data)
test_edge_features, test_edge_weights = prepare_edge_features(test_data)

# Train Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(train_edge_features, train_edge_weights)

# Evaluate the model
train_mse_rf = evaluate_model(rf_model, train_edge_features, train_edge_weights, "Train")
val_mse_rf = evaluate_model(rf_model, val_edge_features, val_edge_weights, "Validation")
test_mse_rf = evaluate_model(rf_model, test_edge_features, test_edge_weights, "Test")
