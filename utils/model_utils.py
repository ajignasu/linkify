import os
import json
import random
import torch
import numpy as np
from PIL import Image
import networkx as nx
from matplotlib import cm
from datetime import datetime
import torch.nn.functional as F
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from torch_geometric.data import Data
from scipy.spatial.distance import cdist
from torch_geometric.utils import to_networkx
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split

random.seed(42)

def find_matching_embedding(uuid_candidate, uuid_to_embedding):
    """Find a matching embedding for a UUID that might be a string or a list."""
    if isinstance(uuid_candidate, list):
        # Try each UUID in the list
        for uuid in uuid_candidate:
            if uuid in uuid_to_embedding:
                return uuid_to_embedding[uuid], uuid
        # If none match directly, try splitting on underscores
        for uuid in uuid_candidate:
            if '_' in uuid:
                first_part = uuid.split('_')[0]
                if first_part in uuid_to_embedding:
                    return uuid_to_embedding[first_part], uuid
        return None, None
    else:
        # Single UUID case
        if uuid_candidate in uuid_to_embedding:
            return uuid_to_embedding[uuid_candidate], uuid_candidate
        elif '_' in uuid_candidate:
            first_part = uuid_candidate.split('_')[0]
            if first_part in uuid_to_embedding:
                return uuid_to_embedding[first_part], uuid_candidate
        return None, None

def visualize_topk_parts(topk_pairs, data_root, images_subdir="images", cols=5, save_path=None, reference_part=None, cluster_info=None):
    """
    Visualizes top-k parts, with the option to include a reference part.
    topk_pairs: list of (assembly_id, part_uuid) tuples
    cluster_info: optional dict mapping part_uuid to cluster_id for display
    """
    n = len(topk_pairs) + (1 if reference_part else 0)
    rows = (n + cols - 1) // cols
    
    # Adjust figure size based on whether we're showing cluster info
    fig_height = 3.5 * rows if cluster_info else 3 * rows
    plt.figure(figsize=(3 * cols, fig_height))
    plot_idx = 1

    # Plot reference part first, if provided
    if reference_part:
        ref_assembly_id, ref_uuid = reference_part
        img_path = os.path.join(data_root, ref_assembly_id, images_subdir, f"{ref_uuid}.png") if images_subdir else os.path.join(data_root, ref_assembly_id, f"{ref_uuid}.png")
        plt.subplot(rows, cols, plot_idx)
        if os.path.exists(img_path):
            img = Image.open(img_path)
            plt.imshow(img)
            title = f"Reference\n{ref_assembly_id}\n{ref_uuid}"
            if cluster_info and ref_uuid in cluster_info:
                title += f"\nTrue Cluster: {cluster_info[ref_uuid]}"
            plt.title(title, fontsize=8)
        else:
            print(f"Reference image not found: {img_path}")
            plt.text(0.5, 0.5, "Reference\nImage\nNot Found", ha='center', va='center', fontsize=10)
            title = f"Reference\n{ref_assembly_id}\n{ref_uuid}"
            if cluster_info and ref_uuid in cluster_info:
                title += f"\nTrue Cluster: {cluster_info[ref_uuid]}"
            plt.title(title, fontsize=8)
        plt.axis('off')
        plot_idx += 1

    # Plot retrieved parts
    for assembly_id, uuid in topk_pairs:
        img_path = os.path.join(data_root, assembly_id, images_subdir, f"{uuid}.png") if images_subdir else os.path.join(data_root, assembly_id, f"{uuid}.png")
        plt.subplot(rows, cols, plot_idx)
        if os.path.exists(img_path):
            img = Image.open(img_path)
            plt.imshow(img)
            title = f"{assembly_id}\n{uuid}"
            if cluster_info and uuid in cluster_info:
                title += f"\nCluster: {cluster_info[uuid]}"
            plt.title(title, fontsize=8)
        else:
            print(f"Image not found: {img_path}")
            plt.text(0.5, 0.5, "Image\nNot Found", ha='center', va='center', fontsize=10)
            title = f"{assembly_id}\n{uuid}"
            if cluster_info and uuid in cluster_info:
                title += f"\nCluster: {cluster_info[uuid]}"
            plt.title(title, fontsize=8)
        plt.axis('off')
        plot_idx += 1

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def visualize_graph_custom(data, node_colors=None, edge_colors=None, node_labels=None, 
                          edge_labels=None, save_path=None, title=None, figsize=(24, 8),
                          node_size=400, show_edge_labels=True, show_node_labels=True):
    """
    Visualize an assembly graph with explicit control over node and edge colors.
    
    Args:
        data: PyTorch Geometric Data object containing:
            - x: node features [N, D]
            - edge_index: edge connectivity [2, E]  
            - edge_attr: edge attributes [E, D] (optional)
            - node_uuids: list of node identifiers
            - assembly_id: assembly identifier
        node_colors: List/array of colors for nodes. Can be:
            - List of color names ['red', 'blue', 'green', ...]
            - List of RGB tuples [(1,0,0), (0,1,0), (0,0,1), ...]
            - List of hex colors ['#FF0000', '#00FF00', '#0000FF', ...]
            - If None, uses default 'skyblue'
        edge_colors: List/array of colors for edges. Same format as node_colors.
            - If None, uses automatic coloring based on edge attributes
        node_labels: List of custom labels for nodes. If None, uses node indices.
        edge_labels: Dict of custom edge labels {(u,v): 'label'}. If None, shows edge attributes.
        save_path: Path to save the plot. If None, saves to default logs directory.
        title: Custom title for the plot. If None, uses assembly_id.
        figsize: Figure size tuple (width, height).
        node_size: Size of nodes in the graph.
        show_edge_labels: Whether to show edge labels.
        show_node_labels: Whether to show node labels.
    
    Returns:
        str: Path where the plot was saved.
    """
    data = data.to('cpu')
    graph = to_networkx(data, to_undirected=True)
    
    num_nodes = data.x.shape[0]
    num_edges = data.edge_index.shape[1]
    
    # Set up the figure
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
    
    # Set title
    if title is None:
        title = f"Graph Visualization - Assembly ID: {data.assembly_id.item() if isinstance(data.assembly_id, torch.Tensor) else data.assembly_id}"
    fig.suptitle(title)
    
    # --- Graph Visualization (Left Subplot) ---
    pos = nx.spring_layout(graph, seed=42)
    
    # Handle node colors
    if node_colors is None:
        final_node_colors = 'skyblue'
    else:
        if len(node_colors) != num_nodes:
            raise ValueError(f"node_colors length ({len(node_colors)}) must match number of nodes ({num_nodes})")
        final_node_colors = node_colors
    
    # Handle edge colors
    if edge_colors is None:
        # Use automatic coloring based on edge attributes (original behavior)
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            edge_attrs = data.edge_attr.numpy()
            edge_attrs = np.nan_to_num(edge_attrs, nan=0.0)
            
            # Handle multi-dimensional edge attributes
            if len(edge_attrs.shape) > 1 and edge_attrs.shape[1] > 1:
                edge_attr_dim = edge_attrs.shape[1]
                
                # Check if first dimension is scalar contact (0 or 1) and rest are embeddings
                first_col = edge_attrs[:, 0]
                is_binary_first_col = np.all(np.isin(first_col, [0.0, 1.0]))
                
                if edge_attr_dim in [256, 384] and not is_binary_first_col:
                    # Pure contact embeddings
                    edge_values = np.linalg.norm(edge_attrs, axis=1)
                elif is_binary_first_col and edge_attr_dim > 1 and edge_attr_dim not in [256, 384]:
                    # First column is binary contact labels
                    edge_values = first_col
                else:
                    # Use L2 norm of the embedding vector
                    edge_values = np.linalg.norm(edge_attrs, axis=1)
                
                # Create edge colors
                if np.max(edge_values) > np.min(edge_values):
                    normalized_values = (edge_values - np.min(edge_values)) / (np.max(edge_values) - np.min(edge_values))
                    final_edge_colors = cm.viridis(normalized_values)
                else:
                    final_edge_colors = 'gray'
            else:
                # Single dimensional edge attributes
                edge_values = edge_attrs.squeeze()
                max_edge_attr = np.max(edge_values)
                final_edge_colors = cm.viridis(edge_values / max_edge_attr) if max_edge_attr > 0 else 'gray'
        else:
            final_edge_colors = 'gray'
    else:
        if len(edge_colors) != num_edges:
            raise ValueError(f"edge_colors length ({len(edge_colors)}) must match number of edges ({num_edges})")
        final_edge_colors = edge_colors
    
    # Handle node labels
    if node_labels is None:
        if show_node_labels:
            final_node_labels = True  # Use default node indices
        else:
            final_node_labels = False
    else:
        if len(node_labels) != num_nodes:
            raise ValueError(f"node_labels length ({len(node_labels)}) must match number of nodes ({num_nodes})")
        final_node_labels = {i: str(label) for i, label in enumerate(node_labels)}
    
    # Handle edge labels
    if edge_labels is None and show_edge_labels:
        # Create default edge labels from edge attributes
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            edge_attrs = data.edge_attr.numpy()
            if len(edge_attrs.shape) > 1 and edge_attrs.shape[1] > 1:
                # For multi-dimensional, show norm or first value
                edge_attr_dim = edge_attrs.shape[1]
                first_col = edge_attrs[:, 0]
                is_binary_first_col = np.all(np.isin(first_col, [0.0, 1.0]))
                
                if is_binary_first_col:
                    edge_values = first_col
                    final_edge_labels = {(u, v): f"{edge_values[i]:.0f}" for i, (u, v) in enumerate(graph.edges())}
                else:
                    edge_values = np.linalg.norm(edge_attrs, axis=1)
                    final_edge_labels = {(u, v): f"{edge_values[i]:.2f}" for i, (u, v) in enumerate(graph.edges())}
            else:
                edge_values = edge_attrs.squeeze()
                final_edge_labels = {(u, v): f"{edge_values[i]:.2f}" for i, (u, v) in enumerate(graph.edges())}
        else:
            final_edge_labels = {}
    elif edge_labels is not None:
        final_edge_labels = edge_labels
    else:
        final_edge_labels = {}
    
    # Draw the graph
    nx.draw(graph, pos, 
            with_labels=final_node_labels if isinstance(final_node_labels, bool) else True,
            labels=final_node_labels if isinstance(final_node_labels, dict) else None,
            node_size=node_size, 
            node_color=final_node_colors, 
            edge_color=final_edge_colors, 
            ax=ax1)
    
    if final_edge_labels and show_edge_labels:
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=final_edge_labels, font_size=8, ax=ax1)
    
    ax1.set_title("Assembly Graph")

    # --- UUID List (Middle Subplot) ---
    node_uuids = data.node_uuids
    # Ensure node_uuids is a flat list of strings
    if isinstance(node_uuids, list) and len(node_uuids) == 1 and isinstance(node_uuids[0], list):
        node_uuids = node_uuids[0]
    uuid_text = "\n".join([f"{i}: {uuid}" for i, uuid in enumerate(node_uuids)])
    ax2.text(0.01, 0.99, uuid_text, transform=ax2.transAxes, fontsize=10, 
             verticalalignment='top', family='monospace')
    ax2.axis('off')
    ax2.set_title("Node UUIDs")

    # --- Edge Attribute Statistics (Right Subplot) ---
    if hasattr(data, 'edge_attr') and data.edge_attr is not None:
        edge_attrs = data.edge_attr.numpy()
        edge_attr_dim = edge_attrs.shape[1] if len(edge_attrs.shape) > 1 else 1
        
        stats_text = f"Edge Attributes Summary:\n"
        stats_text += f"Shape: {edge_attrs.shape}\n"
        stats_text += f"Dimensions: {edge_attr_dim}\n\n"
        
        if edge_attr_dim == 1:
            stats_text += f"Values: {edge_attrs.squeeze()}\n"
        elif edge_attr_dim > 1:
            first_col = edge_attrs[:, 0]
            is_binary_first_col = np.all(np.isin(first_col, [0.0, 1.0]))
            
            if edge_attr_dim in [256, 384]:
                # Pure contact embeddings
                stats_text += f"Contact Embeddings ({edge_attr_dim}-D):\n"
                norms = np.linalg.norm(edge_attrs, axis=1)
                stats_text += f"  Mean norm: {np.mean(norms):.3f}\n"
                stats_text += f"  Std norm: {np.std(norms):.3f}\n"
                stats_text += f"  Min norm: {np.min(norms):.3f}\n"
                stats_text += f"  Max norm: {np.max(norms):.3f}\n"
                stats_text += f"  Zero embeddings: {np.sum(np.all(edge_attrs == 0, axis=1))}\n"
                stats_text += f"  Non-zero embeddings: {np.sum(~np.all(edge_attrs == 0, axis=1))}\n"
            elif is_binary_first_col and edge_attr_dim > 1:
                contact_labels = edge_attrs[:, 0]
                stats_text += f"Contact Labels (col 0):\n"
                stats_text += f"  Contact edges: {np.sum(contact_labels)}\n"
                stats_text += f"  Non-contact edges: {len(contact_labels) - np.sum(contact_labels)}\n\n"
                
                if edge_attr_dim > 1:
                    embeddings = edge_attrs[:, 1:]
                    stats_text += f"Contact Embeddings (cols 1-{edge_attr_dim-1}):\n"
                    stats_text += f"  Embedding dim: {embeddings.shape[1]}\n"
                    stats_text += f"  Mean norm: {np.mean(np.linalg.norm(embeddings, axis=1)):.3f}\n"
                    stats_text += f"  Std norm: {np.std(np.linalg.norm(embeddings, axis=1)):.3f}\n"
                    stats_text += f"  Zero embeddings: {np.sum(np.all(embeddings == 0, axis=1))}\n"
            else:
                for i in range(min(5, edge_attr_dim)):
                    col_data = edge_attrs[:, i]
                    stats_text += f"Dim {i}: μ={np.mean(col_data):.3f}, σ={np.std(col_data):.3f}\n"
                if edge_attr_dim > 5:
                    stats_text += f"... and {edge_attr_dim - 5} more dimensions\n"
        
        ax3.text(0.01, 0.99, stats_text, transform=ax3.transAxes, fontsize=9, 
                verticalalignment='top', family='monospace')
        ax3.axis('off')
        ax3.set_title("Edge Attribute Statistics")
    else:
        ax3.text(0.5, 0.5, "No edge attributes", transform=ax3.transAxes, 
                ha='center', va='center', fontsize=12)
        ax3.axis('off')
        ax3.set_title("Edge Attribute Statistics")

    # --- Save the Plot ---
    if save_path is None:
        logs_dir = os.path.join(os.getcwd(), "logs", "visualize_graphs")
        os.makedirs(logs_dir, exist_ok=True)
        save_path = os.path.join(logs_dir, f'custom_graph_{data.assembly_id}.png')
    else:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Graph visualization saved to {save_path}")
    
    return save_path


def visualize_graph(data, extension):
    data = data.to('cpu')
    print("Data: ", data)
    print("data.assembly_id:", data.assembly_id)
    print("data.x shape: ", data.x.shape)
    # data = data[0]
    # get a graph from data using its assembly_id
    # test_id = "63879_93fa4cbd"
    # test_id = "136339_d3571534"
    # data = data[data.assembly_id == test_id]
    if hasattr(data, 'edge_attr') and data.edge_attr is not None:
        print("data.edge_attr shape: ", data.edge_attr.shape)
        print("data.edge_attr dtype: ", data.edge_attr.dtype)
        print("First few edge_attr values shape:", data.edge_attr[:3].shape if len(data.edge_attr) > 0 else "Empty")
        if len(data.edge_attr.shape) > 1:
            print("data.edge_attr[0] (first edge):", data.edge_attr[0][:10] if data.edge_attr.shape[1] > 10 else data.edge_attr[0])
        else:
            print("data.edge_attr[0] (first edge):", data.edge_attr[0])
    else:
        print("No edge_attr found!")
    
    # Check for other edge attributes in AssemblyGraphDatasetCombined
    if hasattr(data, 'edge_points') and data.edge_points is not None:
        print("Found edge_points shape:", data.edge_points.shape)
    if hasattr(data, 'edge_scalar') and data.edge_scalar is not None:
        print("Found edge_scalar shape:", data.edge_scalar.shape)
    
    # List all attributes to see what's available
    print("All data attributes:", [attr for attr in dir(data) if not attr.startswith('_')])
    graph = to_networkx(data, to_undirected=False)
    print("Graph: ", graph)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8))
    fig.suptitle(f"Graph Visualization - Assembly ID: {data.assembly_id.item() if isinstance(data.assembly_id, torch.Tensor) else data.assembly_id}")

    # --- Graph Visualization (Left Subplot) ---
    if hasattr(data, 'edge_attr') and data.edge_attr is not None:
        edge_attrs = data.edge_attr.numpy()
        print("edge_attrs: ", edge_attrs)
        # exit()
        edge_attrs = np.nan_to_num(edge_attrs, nan=0.0)
        
        # Handle multi-dimensional edge attributes
        if len(edge_attrs.shape) > 1 and edge_attrs.shape[1] > 1:
            # Multi-dimensional edge attributes (e.g., contact embeddings + scalar)
            edge_attr_dim = edge_attrs.shape[1]
            
            # Use different visualization strategies based on the content
            if edge_attr_dim == 1:
                # Single dimension - use directly
                edge_values = edge_attrs.squeeze()
            elif edge_attr_dim > 1:
                # Check if first dimension is scalar contact (0 or 1) and rest are embeddings
                # Only consider it a scalar contact if it's clearly binary AND there are other dimensions
                first_col = edge_attrs[:, 0]
                is_binary_first_col = np.all(np.isin(first_col, [0.0, 1.0]))
                
                # Additional check: if the embedding dimensions (256, 384) are common, treat as pure embeddings
                if edge_attr_dim in [256, 384] and not is_binary_first_col:
                    # Pure contact embeddings (256-D or 384-D)
                    edge_values = np.linalg.norm(edge_attrs, axis=1)
                    print(f"Using L2 norm of {edge_attr_dim}-D contact embeddings for visualization")
                elif is_binary_first_col and edge_attr_dim > 1 and edge_attr_dim not in [256, 384]:
                    # First column is binary contact labels, rest are embeddings
                    edge_values = first_col
                    print(f"Using scalar contact labels (first column) for visualization. Contact edges: {np.sum(edge_values)}/{len(edge_values)}")
                else:
                    # Use L2 norm of the embedding vector for visualization
                    edge_values = np.linalg.norm(edge_attrs, axis=1)
                    print(f"Using L2 norm of {edge_attr_dim}-D edge attributes for visualization")
            
            # Create edge colors and labels
            if np.max(edge_values) > np.min(edge_values):
                normalized_values = (edge_values - np.min(edge_values)) / (np.max(edge_values) - np.min(edge_values))
                edge_colors = cm.viridis(normalized_values)
            else:
                edge_colors = 'gray'
            
            # Create edge labels based on the type of edge values
            if edge_attr_dim == 1 or (edge_attr_dim > 1 and np.all(np.isin(edge_values, [0.0, 1.0]))):
                # Scalar or binary values - show exact values
                edge_labels = {(u, v): f"{edge_values[i]:.0f}" for i, (u, v) in enumerate(graph.edges())}
            else:
                # Continuous values - show with decimals
                edge_labels = {(u, v): f"{edge_values[i]:.2f}" for i, (u, v) in enumerate(graph.edges())}
        else:
            # Single dimensional edge attributes (backward compatibility)
            edge_values = edge_attrs.squeeze()
            max_edge_attr = np.max(edge_values)
            edge_colors = cm.viridis(edge_values / max_edge_attr) if max_edge_attr > 0 else 'gray'
            edge_labels = {(u, v): f"{edge_values[i]:.2f}" for i, (u, v) in enumerate(graph.edges())}
    else:
        edge_colors = 'gray'
        edge_labels = {}
        print("No edge attributes found")

    pos = nx.spring_layout(graph)
    nx.draw(graph, pos, with_labels=True, node_size=400, node_color='skyblue', edge_color=edge_colors, ax=ax1)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=8, ax=ax1)
    ax1.set_title("Assembly Graph")

    # --- UUID List (Middle Subplot) ---
    node_uuids = data.node_uuids
    # Ensure node_uuids is a flat list of strings
    if isinstance(node_uuids, list) and len(node_uuids) == 1 and isinstance(node_uuids[0], list):
        node_uuids = node_uuids[0]
    uuid_text = "\n".join([f"{i}: {uuid}" for i, uuid in enumerate(node_uuids)])
    ax2.text(0.01, 0.99, uuid_text, transform=ax2.transAxes, fontsize=10, verticalalignment='top', family='monospace')
    ax2.axis('off')
    ax2.set_title("Node UUIDs")

    # --- Edge Attribute Statistics (Right Subplot) ---
    if hasattr(data, 'edge_attr') and data.edge_attr is not None:
        edge_attrs = data.edge_attr.numpy()
        edge_attr_dim = edge_attrs.shape[1] if len(edge_attrs.shape) > 1 else 1
        
        stats_text = f"Edge Attributes Summary:\n"
        stats_text += f"Shape: {edge_attrs.shape}\n"
        stats_text += f"Dimensions: {edge_attr_dim}\n\n"
        
        if edge_attr_dim == 1:
            stats_text += f"Values: {edge_attrs.squeeze()}\n"
        elif edge_attr_dim > 1:
            # Show statistics for each dimension
            first_col = edge_attrs[:, 0]
            is_binary_first_col = np.all(np.isin(first_col, [0.0, 1.0]))
            
            if edge_attr_dim in [256, 384]:
                # Pure contact embeddings
                stats_text += f"Contact Embeddings ({edge_attr_dim}-D):\n"
                norms = np.linalg.norm(edge_attrs, axis=1)
                stats_text += f"  Mean norm: {np.mean(norms):.3f}\n"
                stats_text += f"  Std norm: {np.std(norms):.3f}\n"
                stats_text += f"  Min norm: {np.min(norms):.3f}\n"
                stats_text += f"  Max norm: {np.max(norms):.3f}\n"
                stats_text += f"  Zero embeddings: {np.sum(np.all(edge_attrs == 0, axis=1))}\n"
                stats_text += f"  Non-zero embeddings: {np.sum(~np.all(edge_attrs == 0, axis=1))}\n"
            elif is_binary_first_col and edge_attr_dim > 1 and edge_attr_dim not in [256, 384]:
                # First column appears to be binary contact labels
                contact_labels = edge_attrs[:, 0]
                stats_text += f"Contact Labels (col 0):\n"
                stats_text += f"  Contact edges: {np.sum(contact_labels)}\n"
                stats_text += f"  Non-contact edges: {len(contact_labels) - np.sum(contact_labels)}\n\n"
                
                if edge_attr_dim > 1:
                    embeddings = edge_attrs[:, 1:]
                    stats_text += f"Contact Embeddings (cols 1-{edge_attr_dim-1}):\n"
                    stats_text += f"  Embedding dim: {embeddings.shape[1]}\n"
                    stats_text += f"  Mean norm: {np.mean(np.linalg.norm(embeddings, axis=1)):.3f}\n"
                    stats_text += f"  Std norm: {np.std(np.linalg.norm(embeddings, axis=1)):.3f}\n"
                    stats_text += f"  Zero embeddings: {np.sum(np.all(embeddings == 0, axis=1))}\n"
            else:
                # All dimensions are continuous
                for i in range(min(5, edge_attr_dim)):  # Show first 5 dimensions
                    col_data = edge_attrs[:, i]
                    stats_text += f"Dim {i}: μ={np.mean(col_data):.3f}, σ={np.std(col_data):.3f}\n"
                if edge_attr_dim > 5:
                    stats_text += f"... and {edge_attr_dim - 5} more dimensions\n"
        
        ax3.text(0.01, 0.99, stats_text, transform=ax3.transAxes, fontsize=9, 
                verticalalignment='top', family='monospace')
        ax3.axis('off')
        ax3.set_title("Edge Attribute Statistics")
    else:
        ax3.text(0.5, 0.5, "No edge attributes", transform=ax3.transAxes, 
                ha='center', va='center', fontsize=12)
        ax3.axis('off')
        ax3.set_title("Edge Attribute Statistics")

    # --- Save the Plot ---
    logs_dir = os.path.join(os.getcwd(), "logs", "visualize_graphs")
    os.makedirs(logs_dir, exist_ok=True)
    plot_path = os.path.join(logs_dir, f'{extension}_test_{data.assembly_id.item() if isinstance(data.assembly_id, torch.Tensor) else data.assembly_id}.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Graph visualization saved to {plot_path}")


def visualize_graph_3d_interactive(data):
    
    data = data.to('cpu')
    graph = to_networkx(data, to_undirected=True)
    
    # Generate 3D layout
    pos_3d = nx.spring_layout(graph, dim=3, seed=42)
    
    # Extract node positions
    node_x = [pos_3d[node][0] for node in graph.nodes()]
    node_y = [pos_3d[node][1] for node in graph.nodes()]
    node_z = [pos_3d[node][2] for node in graph.nodes()]
    
    # Node trace
    node_trace = go.Scatter3d(
        x=node_x, y=node_y, z=node_z,
        mode='markers+text',
        marker=dict(size=10, color='skyblue', line=dict(width=1, color='black')),
        text=[str(i) for i in graph.nodes()],
        hoverinfo='text'
    )
    
    # Edge traces
    edge_traces = []
    
    if hasattr(data, 'edge_attr') and data.edge_attr is not None:
        edge_attrs = data.edge_attr.numpy()
        edge_attrs = np.nan_to_num(edge_attrs, nan=0.0)
        
        # Handle multi-dimensional edge attributes for visualization
        if len(edge_attrs.shape) > 1 and edge_attrs.shape[1] > 1:
            edge_attr_dim = edge_attrs.shape[1]
            
            # Check if first dimension is scalar contact (0 or 1) and rest are embeddings
            first_col = edge_attrs[:, 0]
            is_binary_first_col = np.all(np.isin(first_col, [0.0, 1.0]))
            
            # Additional check: if the embedding dimensions (256, 384) are common, treat as pure embeddings
            if edge_attr_dim in [256, 384]:
                # Pure contact embeddings (256-D or 384-D)
                edge_values = np.linalg.norm(edge_attrs, axis=1)
                use_binary_coloring = False
                print(f"3D viz: Using L2 norm of {edge_attr_dim}-D contact embeddings for edge coloring")
            elif is_binary_first_col and edge_attr_dim > 1 and edge_attr_dim not in [256, 384]:
                # First column is binary contact labels, rest are embeddings
                edge_values = first_col
                use_binary_coloring = True
                print(f"3D viz: Using scalar contact labels for edge coloring. Contact edges: {np.sum(edge_values)}/{len(edge_values)}")
            else:
                # Use L2 norm of the embedding vector for visualization
                edge_values = np.linalg.norm(edge_attrs, axis=1)
                use_binary_coloring = False
                print(f"3D viz: Using L2 norm of {edge_attr_dim}-D edge attributes for edge coloring")
        else:
            # Single dimensional edge attributes (backward compatibility)
            edge_values = edge_attrs.squeeze()
            use_binary_coloring = np.all(np.isin(edge_values, [0.0, 1.0]))
        
        for i, (u, v) in enumerate(graph.edges()):
            x = [pos_3d[u][0], pos_3d[v][0], None]
            y = [pos_3d[u][1], pos_3d[v][1], None]
            z = [pos_3d[u][2], pos_3d[v][2], None]
            
            # Determine color and width based on edge_attr
            if use_binary_coloring:
                if edge_values[i] > 0:
                    color = 'red'  # Contact
                    width = 4
                    edge_type = "Contact"
                else:
                    color = 'gray'  # No contact
                    width = 2
                    edge_type = "No Contact"
                hover_text = f"Edge {u}-{v}: {edge_type} ({edge_values[i]:.0f})"
            else:
                # Continuous values - use color gradient
                normalized_val = edge_values[i] / np.max(edge_values) if np.max(edge_values) > 0 else 0
                if normalized_val > 0.7:
                    color = 'red'
                    width = 4
                elif normalized_val > 0.3:
                    color = 'orange'
                    width = 3
                else:
                    color = 'gray'
                    width = 2
                hover_text = f"Edge {u}-{v}: {edge_values[i]:.3f}"
            
            edge_trace = go.Scatter3d(
                x=x, y=y, z=z,
                mode='lines',
                line=dict(color=color, width=width),
                hoverinfo='text',
                text=hover_text,
                name=f"Edge {u}-{v}"
            )
            edge_traces.append(edge_trace)
    else:
        # No edge attributes - create simple gray edges
        for i, (u, v) in enumerate(graph.edges()):
            x = [pos_3d[u][0], pos_3d[v][0], None]
            y = [pos_3d[u][1], pos_3d[v][1], None]
            z = [pos_3d[u][2], pos_3d[v][2], None]
            
            edge_trace = go.Scatter3d(
                x=x, y=y, z=z,
                mode='lines',
                line=dict(color='gray', width=2),
                hoverinfo='text',
                text=f"Edge {u}-{v}: No attributes",
                name=f"Edge {u}-{v}"
            )
            edge_traces.append(edge_trace)
    
    # Create figure
    fig = go.Figure(data=[node_trace] + edge_traces)
    
    # Update layout
    fig.update_layout(
        title=f"Interactive 3D Graph - Assembly ID: {data.assembly_id.item() if isinstance(data.assembly_id, torch.Tensor) else data.assembly_id}",
        showlegend=False,
        scene=dict(
            xaxis=dict(showbackground=False),
            yaxis=dict(showbackground=False),
            zaxis=dict(showbackground=False)
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        hovermode='closest'
    )
    
    # Save as HTML
    html_path = os.path.join("C:/Users/jignasa/Repo/assembly_graph_interface/logs/visualize_graphs", f'{data.assembly_id}_3d_interactive.html')
    fig.write_html(html_path)
    print(f"Interactive 3D graph saved to {html_path}")
    
    # Show the plot (optional)
    fig.show()

def test_single_graph(model, graph, uuid_to_embedding, device, data_root=None, visualize=False, k=[1, 2, 3, 4, 5]):
    """
    Tests the model on a single graph and visualizes the results.

    Args:
        model (torch.nn.Module): The trained model.
        graph (torch_geometric.data.Data): The graph data object.
        uuid_to_embedding (dict): A dictionary mapping UUIDs to embeddings.
        device (torch.device): The device to run the model on.
        data_root (str, optional): The root directory of the dataset. Defaults to None.
        visualize (bool, optional): Whether to visualize the results. Defaults to False.
        k (list, optional): A list of top-k values to use for visualization. Defaults to [1, 2, 3, 4, 5].
    """

    model.eval()
    graph = graph.to(device)

    with torch.no_grad():
        output = model(graph)
        global_masked_node_idx = graph.ptr[:-1] + graph.masked_node_idx
        pred = output[global_masked_node_idx]

        # Top-k accuracy using cosine similarity
        if uuid_to_embedding is not None:
            batch_size = pred.shape[0]
            assembly_ids = graph.assembly_id if isinstance(graph.assembly_id, (list, torch.Tensor)) else [graph.assembly_id] * batch_size
            masked_node_idxs = graph.masked_node_idx if isinstance(graph.masked_node_idx, (list, torch.Tensor)) else [graph.masked_node_idx] * batch_size
            for b in range(batch_size):
                pred_embedding = pred[b].cpu()
                pred_embedding_norm = F.normalize(pred_embedding.unsqueeze(0), p=2, dim=1) # [1, D]

                # Prepare embeddings and UUIDs for cosine similarity calculation
                valid_embeddings = []
                valid_uuids = []
                for uuid, embedding in uuid_to_embedding.items():
                    if uuid in graph.node_uuids: # Only use UUIDs present in the current data
                        valid_embeddings.append(embedding)
                        valid_uuids.append(uuid)

                if not valid_embeddings:
                    print("Warning: No valid embeddings found for this batch. Skipping.")
                    continue

                all_embeddings_norm = F.normalize(torch.tensor(valid_embeddings), p=2, dim=1) # [N, D]
                cos_sim = torch.mm(all_embeddings_norm, pred_embedding_norm.t()).squeeze(1) # [N]

                true_assembly_id = str(assembly_ids[b].item() if isinstance(assembly_ids[b], torch.Tensor) else assembly_ids[b])
                true_masked_node_idx = int(masked_node_idxs[b].item() if isinstance(masked_node_idxs[b], torch.Tensor) else masked_node_idxs[b])
                true_uuid = graph.node_uuids[true_masked_node_idx]

                exists = any(uuid == true_uuid for uuid in uuid_to_embedding.keys())
                if not exists:
                    print(f"WARNING: True uuid ({true_uuid}) not in embedding bank!")

                for kk in k:
                    topk_indices = torch.topk(cos_sim, k=kk, largest=True).indices
                    topk_uuids = [valid_uuids[idx] for idx in topk_indices] # Use valid_uuids

                    # --- Visualization of top-k parts for the first sample in the first batch ---
                    if visualize and data_root is not None:
                        reference_part = (true_assembly_id, true_uuid)
                        print(f"Visualizing Top-{kk} retrieved parts for test sample (assembly_id={true_assembly_id}, part_uuid={true_uuid})")
                        save_path = f"top{kk}_retrieved_parts_{true_assembly_id}_{true_uuid}.png"
                        visualize_topk_parts(
                            topk_uuids, data_root, images_subdir="images", cols=5, save_path=save_path,
                            reference_part=reference_part
                        )
                        # Only visualize for the first test sample
                        break


def compute_distance_matrix(cluster_centers):
    return cdist(cluster_centers, cluster_centers)

def to_py(x):
    if isinstance(x, torch.Tensor):
        return x.item() if x.numel() == 1 else x.tolist()
    return str(x) if not isinstance(x, str) else x

def collect_embeddings(dataset, device):
    all_embeddings = []
    all_assembly_ids = []
    all_uuids = []
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    for data in dataloader:
        data = data.to(device)
        assembly_id = str(data.assembly_id.item() if hasattr(data.assembly_id, 'item') else data.assembly_id)
        masked_node_idx = data.masked_node_idx.item()
        for i, uuid in enumerate(data.node_uuids):
            if i != masked_node_idx:
                all_embeddings.append(data.x[i].cpu())
                all_assembly_ids.append(assembly_id)
                all_uuids.append(uuid)
    print(f"Collected {len(all_embeddings)} embeddings from the dataset.")
    return torch.stack(all_embeddings), all_assembly_ids, all_uuids

def get_dataloader(dataset, batch_size=16, shuffle=True, train_ratio=0.7, val_ratio=0.1):
    n = len(dataset)
    if n < 3:
        raise ValueError(
            "Insufficient graphs in the dataset for splitting into train, validation, and test sets."
        )
    n_train = max(1, int(n * train_ratio))
    n_val = max(1, int(n * val_ratio))
    n_test = n - n_train - n_val
    if n_test <= 0:
        n_test = 1
        n_train = max(1, n_train - 1)
    indices = list(range(n))
    random.shuffle(indices)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train+n_val]
    test_indices = indices[n_train+n_val:]
    train_dataset = dataset[train_indices]
    val_dataset = dataset[val_indices]
    test_dataset = dataset[test_indices]
    print("Using batch size: ", batch_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Save train, val, and test indices with or without _RE suffix
    suffix = "_RE" if getattr(dataset, "random_edges", False) else ""
    np.save(os.path.join(dataset.root, f'train_indices{suffix}.npy'), train_indices)
    np.save(os.path.join(dataset.root, f'val_indices{suffix}.npy'), val_indices)
    np.save(os.path.join(dataset.root, f'test_indices{suffix}.npy'), test_indices)

    return train_loader, val_loader, test_loader

def generate_dummy_dataset(num_graphs=100, num_nodes_range=(5, 20), num_edges_range=(5, 50), output_size=256):
    graphs = []
    for _ in range(num_graphs):
        num_nodes = np.random.randint(*num_nodes_range)
        # num_edges = np.random.randint(*num_edges_range)

        # Generate random node features (256-dimensional)
        x = torch.randn((num_nodes, 256))
        edge_index = torch.randint(0, num_nodes, (2, num_nodes))
        edge_attr = torch.randint(0, 2, (num_nodes, 1))
        y = torch.randn((num_nodes,output_size))
        graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        graphs.append(graph)

    return graphs


def generate_dummy_assembly(output_path, num_nodes=10, num_edges=15, embedding_dim=256):
    """
    Generate a dummy assembly JSON file with the specified structure.
    """
    import json
    import random
    from datetime import datetime

    # Generate random UUIDs
    def generate_uuid():
        return f"{random.randint(100000, 999999)}-{random.randint(1000, 9999)}-{random.randint(1000, 9999)}-{random.randint(1000, 9999)}-{random.randint(1000000000, 9999999999)}"

    # Root structure
    root_component_id = generate_uuid()
    root_bodies = {generate_uuid(): {"is_visible": random.choice([True, False])} for _ in range(num_nodes)}
    root = {
        "component": root_component_id,
        "bodies": root_bodies
    }

    # Occurrences, components, and tree
    occurrences = {}
    components = {}
    tree = {"root": {}}
    for _ in range(num_nodes):
        occ_uuid = generate_uuid()
        occurrences[occ_uuid] = {
            "name": f"Component{_ + 1}",
            "type": "Occurrence",
            "component": root_component_id, # Link to the root component
            "is_grounded": random.choice([True, False]),
            "is_visible": random.choice([True, False]),
            "physical_properties": {
                "center_of_mass": {"x": random.random(), "y": random.random(), "z": random.random()},
                "area": random.uniform(1.0, 10.0),
                "volume": random.uniform(1.0, 10.0),
                "density": random.uniform(1.0, 10.0),
                "mass": random.uniform(1.0, 10.0)
            },
            "transform": {
                "origin": {"x": random.random(), "y": random.random(), "z": random.random()},
                "x_axis": {"x": 1.0, "y": 0.0, "z": 0.0},
                "y_axis": {"x": 0.0, "y": 1.0, "z": 0.0},
                "z_axis": {"x": 0.0, "y": 0.0, "z": 1.0}
            }
        }
        components[root_component_id] = {
            "name": "root",
            "type": "Component",
            "part_number": "Untitled",
            "bodies": list(root_bodies.keys())
        }
        tree["root"][occ_uuid] = {}

    # Bodies
    bodies = {
        body_id: {
            "name": f"Body{index + 1}",
            "type": "BRepBody",
            "physical_properties": {
                "area": random.uniform(1.0, 10.0),
                "volume": random.uniform(1.0, 10.0),
                "density": random.uniform(1.0, 10.0),
                "mass": random.uniform(1.0, 10.0)
            },
            "appearance": {"color": f"#{random.randint(0, 0xFFFFFF):06x}"},
            "material": {"type": "Material", "name": "Steel"}
        } for index, body_id in enumerate(root_bodies.keys())
    }

    # Joints
    joints = {
        generate_uuid(): {
            "name": f"Joint{index + 1}",
            "type": "Joint",
            "parent_component": root_component_id,
            "occurrence_one": random.choice(list(occurrences.keys())),
            "occurrence_two": random.choice(list(occurrences.keys())),
            "joint_motion": {
                "joint_type": "RigidJointType"
            },
            "geometry_or_origin_one": {}, # Add this line
            "geometry_or_origin_two": {}  # Add this line
        } for index in range(num_edges)
    }

    # Contacts
    contacts = [
        {
            "entity_one": {"type": "BRepFace", "body": random.choice(list(bodies.keys()))},
            "entity_two": {"type": "BRepFace", "body": random.choice(list(bodies.keys()))}
        } for _ in range(num_edges)
    ]

    # Holes
    holes = [
        {
            "type": "RoundHoleWithThroughBottom",
            "body": random.choice(list(bodies.keys())),
            "diameter": random.uniform(0.5, 2.0),
            "length": random.uniform(0.5, 2.0),
            "origin": {"x": random.random(), "y": random.random(), "z": random.random()},
            "direction": {"x": 0.0, "y": 0.0, "z": 1.0}
        } for _ in range(num_edges)
    ]

    # Properties
    properties = {
        "name": "Untitled",
        "area": random.uniform(10.0, 100.0),
        "volume": random.uniform(10.0, 100.0),
        "density": random.uniform(1.0, 10.0),
        "mass": random.uniform(1.0, 10.0),
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat()
    }

    # Combine all into the final assembly structure
    assembly = {
        "tree": tree,
        "root": root,
        "occurrences": occurrences,
        "components": components,
        "bodies": bodies,
        "joints": joints,
        "as_built_joints": {},
        "contacts": contacts,
        "holes": holes,
        "properties": properties
    }

    # Save to the specified output path
    with open(output_path, "w") as f:
        json.dump(assembly, f, indent=4)

    print(f"Dummy assembly JSON saved to {output_path}")

def make_directory(input_log_arg):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(input_log_arg, timestamp)
    os.makedirs(log_dir, exist_ok=True)
    return log_dir

def create_log_loss_file(log_dir):
    log_file_path = os.path.join(log_dir, 'loss_log.txt')
    if not os.path.exists(log_file_path):
        with open(log_file_path, 'w') as log_file:
            log_file.write("Epoch\tTrain Loss\tValidation Loss\tTest Loss\n")
    return log_file_path

def save_config(model_config, training_config, save_path):
    with open(os.path.join(save_path, 'model_config.json'), 'w') as f:
        json.dump(model_config, f, indent=4)
    
    with open(os.path.join(save_path, 'training_config.json'), 'w') as f:
        json.dump(training_config, f, indent=4)
    
    print(f"Configuration files saved to {save_path}")

def log_loss(log_path, epoch=None, train_loss=None, val_loss=None, test_loss=None, **kwargs):
    log_entry = {
        'epoch': epoch,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'test_loss': test_loss
    }
    log_entry.update(kwargs)  # Add additional metrics like top-k accuracies
    with open(log_path, 'a') as f:
        f.write(json.dumps(log_entry) + '\n')

def plot_train_val_accuracy(train_accuracies, val_accuracies_top1, val_accuracies_top2, val_accuracies_top3, val_balanced_accuracies=None, save_path=None):
    """Plots training and validation accuracy curves."""
    epochs = np.arange(1, len(train_accuracies) + 1)

    plt.figure(figsize=(12, 6))
    plt.plot(epochs, train_accuracies, label='Training Accuracy', color='blue', marker='o')
    plt.plot(epochs, val_accuracies_top1, label='Validation Top-1 Accuracy', color='green', marker='x')
    plt.plot(epochs, val_accuracies_top2, label='Validation Top-2 Accuracy', color='red', marker='x')
    plt.plot(epochs, val_accuracies_top3, label='Validation Top-3 Accuracy', color='purple', marker='x')
    if val_balanced_accuracies:
        plt.plot(epochs, val_balanced_accuracies, label='Validation Balanced Accuracy', color='cyan', marker='s')

    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()

def plot_train_val_loss(train_losses, val_losses, save_path=None):
    epochs = np.arange(1, len(train_losses) + 1)
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, label='Training Loss', color='blue', marker='o')
    plt.plot(epochs, val_losses, label='Validation Loss', color='orange', marker='x')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    else:
        pass
    plt.close()