#!/usr/bin/env python3

import os
import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pickle
from Bio import SeqIO
import argparse
from collections import defaultdict
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.utils import from_networkx
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split

# ------------- De Bruijn Graph Construction -------------

def read_fasta(file_path):
    """Read sequences from a FASTA file."""
    sequences = []
    for record in SeqIO.parse(file_path, "fasta"):
        sequences.append(str(record.seq).upper())
    return sequences

def generate_kmers(sequence, k):
    """Generate k-mers from a sequence."""
    return [sequence[i:i+k] for i in range(len(sequence) - k + 1)]

def build_debruijn_graph(sequences, k):
    """Build a De Bruijn graph from sequences using k-mers."""
    G = nx.DiGraph()
    
    for sequence in sequences:
        kmers = generate_kmers(sequence, k)
        
        # Add edges between (k-1)-mers
        for i in range(len(kmers) - 1):
            # For each k-mer, we create an edge from its prefix to its suffix
            prefix = kmers[i][:-1]
            suffix = kmers[i][1:]
            
            # Add nodes if they don't exist
            if not G.has_node(prefix):
                G.add_node(prefix)
            if not G.has_node(suffix):
                G.add_node(suffix)
            
            # Add edge or increment weight if edge exists
            if G.has_edge(prefix, suffix):
                G[prefix][suffix]['weight'] += 1
            else:
                G.add_edge(prefix, suffix, weight=1)
    
    return G

# ------------- Graph Neural Network Models -------------

class GCN(torch.nn.Module):
    """Graph Convolutional Network model for De Bruijn graph analysis."""
    def __init__(self, num_node_features, hidden_channels, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.linear = torch.nn.Linear(hidden_channels, num_classes)
    
    def forward(self, x, edge_index, edge_weight=None):
        # First Graph Convolution layer
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=0.1, training=self.training)
        
        # Second Graph Convolution layer
        x = self.conv2(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=0.1, training=self.training)
        
        # Third Graph Convolution layer
        x = self.conv3(x, edge_index, edge_weight)
        x = F.relu(x)
        
        # Final linear layer
        x = self.linear(x)
        
        return x

class GAT(torch.nn.Module):
    """Graph Attention Network model for De Bruijn graph analysis."""
    def __init__(self, num_node_features, hidden_channels, num_classes, heads=8):
        super(GAT, self).__init__()
        self.conv1 = GATConv(num_node_features, hidden_channels, heads=heads)
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=1)
        self.linear = torch.nn.Linear(hidden_channels, num_classes)
    
    def forward(self, x, edge_index):
        # First Graph Attention layer
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.1, training=self.training)
        
        # Second Graph Attention layer
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        
        # Final linear layer
        x = self.linear(x)
        
        return x

# ------------- Graph Processing and Training Functions -------------

def prepare_graph_for_gnn(G):
    """Convert a NetworkX graph to a PyTorch Geometric Data object."""
    # Create node features based on the node labels (k-mers)
    node_mapping = {node: i for i, node in enumerate(G.nodes())}
    
    # Create one-hot encoding for the nucleotides in each node
    nucleotides = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
    
    # Initialize node features
    node_features = []
    for node in G.nodes():
        # One-hot encode each nucleotide in the node (k-mer)
        feature = np.zeros(5 * len(node))  # 5 possible nucleotides
        for i, nuc in enumerate(node):
            if nuc in nucleotides:
                feature[i * 5 + nucleotides[nuc]] = 1
            else:
                # Handle non-standard nucleotides
                feature[i * 5 + 4] = 1  # 'N' or other
        node_features.append(feature)
    
    # Create edge list and edge weights
    edge_list = []
    edge_weights = []
    for u, v, data in G.edges(data=True):
        edge_list.append([node_mapping[u], node_mapping[v]])
        edge_weights.append(data.get('weight', 1.0))
    
    # Convert to PyTorch tensors
    x = torch.tensor(np.array(node_features), dtype=torch.float)
    edge_index = torch.tensor(np.array(edge_list).T, dtype=torch.long)
    edge_weight = torch.tensor(np.array(edge_weights), dtype=torch.float)
    
    # Create PyG Data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_weight)
    
    return data, node_mapping

def train_model(model, data, num_epochs=200, lr=0.01):
    """Train the graph neural network model."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    criterion = torch.nn.MSELoss()  # Can be changed based on the task
    
    model.train()
    
    # Create a target based on node centrality
    G_nx = nx.DiGraph()
    edge_index = data.edge_index.numpy()
    for i in range(edge_index.shape[1]):
        G_nx.add_edge(edge_index[0, i], edge_index[1, i])
    
    # Calculate centrality as target
    centrality = np.array(list(nx.pagerank(G_nx).values()))
    y = torch.tensor(centrality, dtype=torch.float).view(-1, 1)
    
    # Training loop
    losses = []
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        if hasattr(data, 'edge_attr'):
            out = model(data.x, data.edge_index, data.edge_attr)
        else:
            out = model(data.x, data.edge_index)
        
        loss = criterion(out, y)
        losses.append(loss.item())
        
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 20 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.6f}')
    
    return model, losses

def save_graph_and_model(G, model, graph_file, model_file, node_mapping_file):
    """Save the graph, trained model, and node mapping."""
    # Save the graph
    with open(graph_file, 'wb') as f:
        pickle.dump(G, f)
    
    # Save the model
    torch.save(model.state_dict(), model_file)
    
    # Save the node mapping
    with open(node_mapping_file, 'wb') as f:
        pickle.dump(node_mapping, f)
    
    print(f"Graph saved to {graph_file}")
    print(f"Model saved to {model_file}")
    print(f"Node mapping saved to {node_mapping_file}")

def load_graph_and_model(graph_file, model_file, node_mapping_file, model_class, model_params):
    """Load the graph, trained model, and node mapping."""
    # Load the graph
    with open(graph_file, 'rb') as f:
        G = pickle.load(f)
    
    # Load the node mapping
    with open(node_mapping_file, 'rb') as f:
        node_mapping = pickle.load(f)
    
    # Initialize the model with the same parameters
    model = model_class(**model_params)
    model.load_state_dict(torch.load(model_file))
    model.eval()
    
    return G, model, node_mapping

def visualize_training(losses, output_file=None):
    """Visualize the training loss."""
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    if output_file:
        plt.savefig(output_file, format='png', dpi=300)
        print(f"Training loss plot saved to {output_file}")
    else:
        plt.show()

def visualize_graph_with_predictions(G, model, data, node_mapping, output_file=None):
    """Visualize the graph with node predictions from the model."""
    model.eval()
    
    # Get model predictions
    with torch.no_grad():
        if hasattr(data, 'edge_attr'):
            pred = model(data.x, data.edge_index, data.edge_attr)
        else:
            pred = model(data.x, data.edge_index)
    
    pred = pred.numpy().flatten()
    
    # Map predictions back to original nodes
    reverse_mapping = {v: k for k, v in node_mapping.items()}
    node_predictions = {reverse_mapping[i]: float(pred[i]) for i in range(len(pred))}
    
    # Visualize graph
    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(G, seed=42)
    
    # Draw nodes with color based on predictions
    node_colors = [node_predictions[node] for node in G.nodes()]
    nodes = nx.draw_networkx_nodes(G, pos, node_size=500, node_color=node_colors, cmap=plt.cm.viridis)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.7, arrows=True, arrowsize=15)
    
    # Draw node labels
    nx.draw_networkx_labels(G, pos, font_size=8, font_family='sans-serif')
    
    plt.colorbar(nodes, label='Model Prediction')
    plt.axis('off')
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, format='png', dpi=300)
        print(f"Graph visualization with predictions saved to {output_file}")
    else:
        plt.show()

# ------------- Main Function -------------

def main():
    parser = argparse.ArgumentParser(description='Train a GNN on a De Bruijn graph from FASTA sequences.')
    parser.add_argument('-f', '--fasta', help='Path to FASTA file')
    parser.add_argument('-k', '--kmer', type=int, default=3, help='k-mer size (default: 3)')
    parser.add_argument('-m', '--model', type=str, default='gcn', choices=['gcn', 'gat'], 
                        help='GNN model to use (default: gcn)')
    parser.add_argument('-e', '--epochs', type=int, default=200, help='Number of training epochs (default: 200)')
    parser.add_argument('-l', '--learning_rate', type=float, default=0.01, help='Learning rate (default: 0.01)')
    parser.add_argument('-hc', '--hidden_channels', type=int, default=64, 
                        help='Number of hidden channels in GNN (default: 64)')
    parser.add_argument('-o', '--output_dir', default='output', help='Output directory (default: output)')
    parser.add_argument('--load', action='store_true', help='Load saved graph and model instead of creating new ones')
    parser.add_argument('--graph_file', default='output/debruijn_graph.pkl', help='File to save/load graph')
    parser.add_argument('--model_file', default='output/gnn_model.pth', help='File to save/load model')
    parser.add_argument('--mapping_file', default='output/node_mapping.pkl', help='File to save/load node mapping')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        if args.load:
            # Load existing graph and model
            print(f"Loading graph from {args.graph_file} and model from {args.model_file}...")
            
            # Determine model parameters based on the saved model file name
            model_class = GCN if 'gcn' in args.model_file.lower() else GAT
            
            # For simplicity, we'll use default parameters
            # In a real-world scenario, you would save and load these parameters as well
            model_params = {
                'num_node_features': 5 * (args.kmer - 1),  # For k-mers of length k-1
                'hidden_channels': args.hidden_channels,
                'num_classes': 1  # Predicting one value per node
            }
            
            if model_class == GAT:
                model_params['heads'] = 8
            
            G, model, node_mapping = load_graph_and_model(
                args.graph_file, args.model_file, args.mapping_file, model_class, model_params
            )
            
            # Convert graph to PyG data
            data, _ = prepare_graph_for_gnn(G)
            
        else:
            # Create new graph and train model
            if not args.fasta:
                print("Error: FASTA file is required when not loading an existing graph.")
                return
            
            # Read sequences and build graph
            print(f"Reading sequences from {args.fasta}...")
            sequences = read_fasta(args.fasta)
            if not sequences:
                print("No sequences found in the FASTA file.")
                return
            
            print(f"Building De Bruijn graph with k={args.kmer}...")
            G = build_debruijn_graph(sequences, args.kmer)
            print(f"Created graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
            
            # Prepare graph for GNN
            print("Converting graph to PyTorch Geometric format...")
            data, node_mapping = prepare_graph_for_gnn(G)
            
            # Create and train the model
            print(f"Creating {args.model.upper()} model...")
            num_node_features = data.x.size(1)
            
            if args.model.lower() == 'gcn':
                model = GCN(num_node_features=num_node_features, 
                           hidden_channels=args.hidden_channels, 
                           num_classes=1)  # Predicting one value per node
            else:  # GAT
                model = GAT(num_node_features=num_node_features, 
                           hidden_channels=args.hidden_channels, 
                           num_classes=1,
                           heads=8)
            
            print(f"Training model for {args.epochs} epochs...")
            model, losses = train_model(model, data, num_epochs=args.epochs, lr=args.learning_rate)
            
            # Save graph and model
            print("Saving graph and model...")
            graph_file = os.path.join(args.output_dir, 'debruijn_graph.pkl')
            model_file = os.path.join(args.output_dir, f'{args.model}_model.pth')
            mapping_file = os.path.join(args.output_dir, 'node_mapping.pkl')
            
            save_graph_and_model(G, model, graph_file, model_file, mapping_file)
            
            # Plot training loss
            loss_plot_file = os.path.join(args.output_dir, 'training_loss.png')
            visualize_training(losses, loss_plot_file)
        
        # Visualize graph with predictions
        print("Visualizing graph with model predictions...")
        viz_file = os.path.join(args.output_dir, 'graph_predictions.png')
        visualize_graph_with_predictions(G, model, data, node_mapping, viz_file)
        
        print("Done!")
        
    except FileNotFoundError as e:
        print(f"Error: {str(e)}")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()