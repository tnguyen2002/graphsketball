import torch
import numpy as np
from torch_geometric.data import Data

# Load the processed data
data_file = './processed_data.pt'
data = torch.load(data_file)

# Analysis functions
def analyze_graph(data):
    # Calculate degree for each node
    edge_index = data.edge_index
    num_nodes = data.num_nodes
    degrees = torch.zeros(num_nodes, dtype=torch.long)

    # Increment degree counts for each node in edge_index
    for node in edge_index[0]:
        degrees[node] += 1

    # Average degree
    avg_degree = degrees.float().mean().item()

    # Node with max degree
    max_degree = degrees.max().item()
    max_degree_node = degrees.argmax().item()

    # Number of edges
    num_edges = data.edge_index.size(1) // 2  # Divide by 2 for undirected graph

    # Number of nodes
    num_nodes = data.num_nodes

    # Edge weight statistics if available
    if 'edge_attr' in data:
        edge_weights = data.edge_attr
        avg_edge_weight = edge_weights.mean().item()
        max_edge_weight = edge_weights.max().item()
        min_edge_weight = edge_weights.min().item()
    else:
        avg_edge_weight = max_edge_weight = min_edge_weight = None

    # Print analysis results
    print("Graph Analysis")
    print("--------------")
    print(f"Number of nodes: {num_nodes}")
    print(f"Number of edges: {num_edges}")
    print(f"Average degree: {avg_degree:.2f}")
    print(f"Max degree: {max_degree} (Node {max_degree_node})")
    
    if avg_edge_weight is not None:
        print(f"Average edge weight: {avg_edge_weight:.2f}")
        print(f"Max edge weight: {max_edge_weight:.2f}")
        print(f"Min edge weight: {min_edge_weight:.2f}")

if __name__ == "__main__":
    analyze_graph(data)
