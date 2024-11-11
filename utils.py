import networkx as nx
import plotly.graph_objects as go
import numpy as np

def visualize_graph(data, node_idx_to_player_season, slug_to_name, title="Graph Visualization", max_nodes=1000, color_by=None):
    """
    Visualize the graph using Plotly for interactivity.

    Args:
        data (torch_geometric.data.Data): The graph data object.
        node_idx_to_player_season (dict): Mapping from node index to (player_slug, season).
        title (str): Title of the graph.
        max_nodes (int): Maximum number of nodes to display (to avoid clutter).
        color_by (str): Optional attribute to color nodes by.
    """
    # Convert PyTorch Geometric data to NetworkX graph
    G = nx.Graph()
    edges = data.edge_index.cpu().numpy()
    for i in range(edges.shape[1]):
        G.add_edge(edges[0, i], edges[1, i])
    
    # Limit the number of nodes visualized for readability
    if len(G.nodes) > max_nodes:
        print(f"Graph has {len(G.nodes)} nodes, visualizing only the first {max_nodes} nodes.")
        G = G.subgraph(list(G.nodes)[:max_nodes])

    # Position the nodes with spring layout
    pos = nx.spring_layout(G, seed=42)

    # Extract edge coordinates
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    # Plot edges
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    # Plot nodes
    node_x = []
    node_y = []
    node_color = []
    node_labels = []

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        # Color mapping (change this to suit your data)
        if color_by and color_by in data:
            node_color.append(data[color_by][node].item())  # Use attribute value if provided
        else:
            node_color.append(G.degree[node])  # Default color by degree

        # Set player name as label
        slug = node_idx_to_player_season[node][0]  # Get player_slug
        player_name = slug_to_name[slug]
        node_labels.append(player_name)

    # Normalize node colors to the color scale
    node_color = np.array(node_color)
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            color=node_color,
            size=10,
            colorbar=dict(
                thickness=15,
                title='Node Color Scale',
                xanchor='left',
                titleside='right'
            ),
        ),
        text=node_labels,  # Use player names as labels
        hoverinfo='text'
    )

    # Create the Plotly figure
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title=title,
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=0, l=0, r=0, t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )
    
    fig.show()

# Visualize the graph with player names
#visualize_graph(data=data, node_idx_to_player_season=node_idx_to_player_season, title="Full Graph")
