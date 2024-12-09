import networkx as nx
import plotly.graph_objects as go
import numpy as np

def visualize_graph(data, player_to_node, slug_to_name, title="Graph Visualization", max_nodes=25, color_by=None):
    """
    Visualize the graph using Plotly for interactivity, with edge weights displayed as labels.

    Args:
        data (torch_geometric.data.Data): The graph data object.
        player_to_node (dict): Mapping from player slug to node index.
        slug_to_name (dict): Mapping from player slug to player name.
        title (str): Title of the graph.
        max_nodes (int): Maximum number of nodes to display (to avoid clutter).
        color_by (str): Optional attribute to color nodes by.
    """
    # Convert PyTorch Geometric data to NetworkX graph
    G = nx.Graph()
    edges = data.edge_index.cpu().numpy()
    edge_weights = data.edge_attr.cpu().numpy() if data.edge_attr is not None else None

    for i in range(edges.shape[1]):
        node1, node2 = edges[0, i], edges[1, i]
        weight = edge_weights[i] if edge_weights is not None else 1.0
        G.add_edge(node1, node2, weight=weight)

    # Limit the number of nodes visualized for readability
    if len(G.nodes) > max_nodes:
        print(f"Graph has {len(G.nodes)} nodes, visualizing only the first {max_nodes} nodes.")
        G = G.subgraph(list(G.nodes)[:max_nodes])

    # Position the nodes with spring layout
    pos = nx.spring_layout(G, seed=42)

    # Extract edge coordinates
    edge_x = []
    edge_y = []
    edge_labels_x = []
    edge_labels_y = []
    edge_text = []
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

        # Position the weight label at the center of the edge
        edge_labels_x.append((x0 + x1) / 2)
        edge_labels_y.append((y0 + y1) / 2)
        weight = edge[2]['weight']
        edge_text.append(f"{weight:.2f}")

    # Plot edges without hover text for cleaner display
    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )

    # Plot edge labels as separate trace
    edge_labels_trace = go.Scatter(
        x=edge_labels_x,
        y=edge_labels_y,
        mode='text',
        text=edge_text,
        textposition='middle center',
        hoverinfo='none',
        textfont=dict(color='blue')
    )

    # Plot nodes with player names
    node_x = []
    node_y = []
    node_color = []
    node_labels = []

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

        # Determine node color
        if color_by and hasattr(data, color_by):
            node_color.append(getattr(data, color_by)[node].item())  # Use attribute if provided
        else:
            node_color.append(G.degree[node])  # Default color by degree

        # Set player name as label using slug_to_name
        player_slug = next((slug for slug, idx in player_to_node.items() if idx == node), None)
        player_name = slug_to_name.get(player_slug, f"Node {node}")  # Fallback to node index if no name
        node_labels.append(player_name)

    # Normalize node colors for visualization
    node_color = np.array(node_color)
    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
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
        text=node_labels,  # Player names as hover text
        hoverinfo='text'
    )

    # Create the Plotly figure with edge labels trace
    fig = go.Figure(
        data=[edge_trace, edge_labels_trace, node_trace],
        layout=go.Layout(
            title=title,
            titlefont_size=16,
            showlegend=False,
            hovermode='closest',
            margin=dict(b=0, l=0, r=0, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
    )

    fig.show()
