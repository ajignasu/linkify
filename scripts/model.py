import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GATConv


class GATv2Classification(nn.Module):
    def __init__(self, node_attr_size, edge_attr_size, hidden_size, num_clusters, num_gat_layers=2, dropout=0.3, gat_heads=1, attn_drop=0.0, activation='relu', residual='none', layer_norm=False):
        super(GATv2Classification, self).__init__()
        self.num_gat_layers = num_gat_layers
        self.gat_heads = gat_heads
        self.residual = residual
        self.layer_norm = layer_norm
        self.gat_layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        
        # Output of GATv2Conv with `concat=True` is (hidden_size * heads)
        feature_size = hidden_size * gat_heads
        
        self.node_feat_proj = nn.Linear(node_attr_size, feature_size)
        
        # Add layer normalization if requested
        if self.layer_norm:
            self.norms = nn.ModuleList([nn.LayerNorm(feature_size) for _ in range(num_gat_layers)])
        else:
            self.norms = nn.ModuleList([nn.BatchNorm1d(feature_size) for _ in range(num_gat_layers)])
        
        # Choose activation function
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'elu':
            self.activation = F.elu
        elif activation == 'gelu':
            self.activation = F.gelu
        elif activation == 'leaky_relu':
            self.activation = F.leaky_relu
        else:
            self.activation = F.relu

        for i in range(num_gat_layers):
            self.gat_layers.append(GATv2Conv(feature_size, hidden_size, heads=gat_heads, edge_dim=edge_attr_size, concat=True, dropout=attn_drop))
        
        self.classifier = nn.Linear(feature_size, num_clusters)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # Project input features to the model's feature size
        x = self.node_feat_proj(x)
        
        for i, gat_layer in enumerate(self.gat_layers):
            identity = x
            
            # Pre-Normalization
            x_norm = self.norms[i](x)
            
            # GAT Layer
            x_gat = gat_layer(x_norm, edge_index, edge_attr)

            # Activation
            x_act = self.activation(x_gat)

            # Dropout
            x_drop = self.dropout(x_act)
            
            # Apply residual connection
            if self.residual == 'add':
                x = x_drop + identity
            else: # No residual
                x = x_drop
        
        node_embeddings = x
        
        if self.classifier:
            classification_output = self.classifier(x)
        else:
            classification_output = x
        
        return classification_output, node_embeddings


class GATv2ClassificationNoEdgeAttr(nn.Module):
    def __init__(self, node_attr_size, hidden_size, num_clusters, num_gat_layers=2, dropout=0.3, gat_heads=1, attn_drop=0.0, activation='relu', residual='none', layer_norm=False):
        super(GATv2ClassificationNoEdgeAttr, self).__init__()
        self.num_gat_layers = num_gat_layers
        self.gat_heads = gat_heads
        self.residual = residual
        self.layer_norm = layer_norm
        self.gat_layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)

        feature_size = hidden_size * gat_heads
        
        self.node_feat_proj = nn.Linear(node_attr_size, feature_size)

        if self.layer_norm:
            self.norms = nn.ModuleList([nn.LayerNorm(feature_size) for _ in range(num_gat_layers)])
        else:
            self.norms = nn.ModuleList([nn.BatchNorm1d(feature_size) for _ in range(num_gat_layers)])

        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'elu':
            self.activation = F.elu
        elif activation == 'gelu':
            self.activation = F.gelu
        elif activation == 'leaky_relu':
            self.activation = F.leaky_relu
        else:
            self.activation = F.relu

        for i in range(num_gat_layers):
            self.gat_layers.append(GATv2Conv(feature_size, hidden_size, heads=gat_heads, concat=True, dropout=attn_drop))
        
        self.classifier = nn.Linear(feature_size, num_clusters)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        x = self.node_feat_proj(x)
        
        for i, gat_layer in enumerate(self.gat_layers):
            identity = x
            
            x_norm = self.norms[i](x)
            
            x_gat = gat_layer(x_norm, edge_index)

            x_act = self.activation(x_gat)

            x_drop = self.dropout(x_act)
            
            if self.residual == 'add':
                x = x_drop + identity
            else:
                x = x_drop
        
        node_embeddings = x
        
        if self.classifier:
            classification_output = self.classifier(x)
        else:
            classification_output = x
        
        return classification_output, node_embeddings


class GATClassification(nn.Module):
    def __init__(self, node_attr_size, edge_attr_size, hidden_size, num_clusters, num_gat_layers=2, dropout=0.3, gat_heads=1, attn_drop=0.0, activation='relu', residual='none'):
        super(GATClassification, self).__init__()
        self.num_gat_layers = num_gat_layers
        self.gat_heads = gat_heads
        self.residual = residual
        self.gat_layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        
        # Output of GATv2Conv with `concat=True` is (hidden_size * heads)
        feature_size = hidden_size * gat_heads
        self.norms = nn.ModuleList([nn.BatchNorm1d(feature_size) for _ in range(num_gat_layers)])
        
        # Choose activation function
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'elu':
            self.activation = F.elu
        elif activation == 'gelu':
            self.activation = F.gelu
        elif activation == 'leaky_relu':
            self.activation = F.leaky_relu
        else:
            self.activation = F.relu

        for i in range(num_gat_layers):
            in_size = node_attr_size if i == 0 else feature_size
            self.gat_layers.append(GATConv(in_size, hidden_size, heads=gat_heads, edge_dim=edge_attr_size, concat=True, dropout=attn_drop))
        
        self.classifier = nn.Linear(feature_size, num_clusters)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        for i, gat_layer in enumerate(self.gat_layers):
            identity = x if self.residual != 'none' and x.size(-1) == self.gat_layers[i].out_channels * self.gat_heads else None
            
            x = self.activation(gat_layer(x, edge_index, edge_attr))
            x = self.norms[i](x)
            x = self.dropout(x)
            
            # Apply residual connection
            if self.residual == 'add' and identity is not None:
                x = x + identity
            elif self.residual == 'concat' and identity is not None:
                x = torch.cat([x, identity], dim=-1)
        
        node_embeddings = x
        
        if self.classifier:
            classification_output = self.classifier(x)
        else:
            classification_output = x
        
        return classification_output, node_embeddings