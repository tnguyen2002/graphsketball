import networkx as nx
from torch_geometric.utils import to_networkx
from create_dataset import prepare_data

def compute_diameter(data):
    graph = to_networkx(data, to_undirected=True, edge_attrs=['edge_attr'])
    
    if not nx.is_connected(graph):
        print("Graph is not connected; computing diameter of the largest connected component.")
        largest_cc = max(nx.connected_components(graph), key=len)
        subgraph = graph.subgraph(largest_cc).copy()
    else:
        subgraph = graph

    diameter = nx.diameter(subgraph)
    return diameter

train_data, val_data, test_data = prepare_data()
graph_diameter = compute_diameter(train_data)
print(f"Graph Diameter: {graph_diameter}")
