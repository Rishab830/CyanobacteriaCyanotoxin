#!/usr/bin/env python3

import os
import glob
import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pickle
from Bio import SeqIO
import argparse
from collections import defaultdict
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, GATConv
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import shutil
from datetime import datetime

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
    
    def forward(self, x, edge_index, edge_weight=None, batch=None):
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
    
    def forward(self, x, edge_index, batch=None):
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
    
    # Handle empty graphs or graphs with no edges
    if not edge_list:
        x = torch.tensor(np.array(node_features), dtype=torch.float)
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_weight = torch.zeros(0, dtype=torch.float)
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_weight)
        return data, node_mapping
    
    # Convert to PyTorch tensors
    x = torch.tensor(np.array(node_features), dtype=torch.float)
    edge_index = torch.tensor(np.array(edge_list).T, dtype=torch.long)
    edge_weight = torch.tensor(np.array(edge_weights), dtype=torch.float)
    
    # Create PyG Data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_weight)
    
    return data, node_mapping

def calculate_target_values(data):
    """Calculate target values for graph nodes."""
    # Create a NetworkX graph from the PyG data
    G_nx = nx.DiGraph()
    edge_index = data.edge_index.numpy()
    
    # Handle empty graphs
    if edge_index.size == 0:
        return torch.zeros((data.x.size(0), 1), dtype=torch.float)
    
    for i in range(edge_index.shape[1]):
        G_nx.add_edge(edge_index[0, i], edge_index[1, i])
    
    # Calculate centrality as target
    if G_nx.number_of_nodes() > 0:
        centrality = np.array(list(nx.pagerank(G_nx).values()))
        y = torch.tensor(centrality, dtype=torch.float).view(-1, 1)
    else:
        # Handle disconnected graphs
        y = torch.zeros((data.x.size(0), 1), dtype=torch.float)
    
    return y

def train_model(model, dataset, num_epochs=200, lr=0.01):
    """Train the graph neural network model on multiple graphs."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    criterion = torch.nn.MSELoss()
    
    model.train()
    
    # Calculate targets for each graph in the dataset
    targets = [calculate_target_values(data) for data in dataset]
    
    # Training loop
    losses = []
    for epoch in range(num_epochs):
        epoch_loss = 0
        
        for i, data in enumerate(dataset):
            optimizer.zero_grad()
            
            # Forward pass
            if hasattr(data, 'edge_attr') and data.edge_attr.size(0) > 0:
                out = model(data.x, data.edge_index, data.edge_attr)
            else:
                out = model(data.x, data.edge_index)
            
            # Calculate loss
            target = targets[i]
            loss = criterion(out, target)
            epoch_loss += loss.item()
            
            # Backward pass
            loss.backward()
            optimizer.step()
        
        avg_loss = epoch_loss / len(dataset)
        losses.append(avg_loss)
        
        if (epoch + 1) % 20 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}, Avg Loss: {avg_loss:.6f}')
    
    return model, losses

def create_output_structure(base_dir):
    """Create the output directory structure."""
    # Create timestamp for unique folder name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(base_dir, f"run_{timestamp}")
    
    # Create main output directory
    os.makedirs(out_dir, exist_ok=True)
    
    # Create subdirectories
    graphs_dir = os.path.join(out_dir, "graphs")
    models_dir = os.path.join(out_dir, "models")
    plots_dir = os.path.join(out_dir, "plots")
    mappings_dir = os.path.join(out_dir, "mappings")
    
    os.makedirs(graphs_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(mappings_dir, exist_ok=True)
    
    return {
        "main": out_dir,
        "graphs": graphs_dir,
        "models": models_dir,
        "plots": plots_dir,
        "mappings": mappings_dir
    }

def save_graph_and_model(G, model, node_mapping, file_basename, output_dirs):
    """Save the graph, trained model, and node mapping."""
    # Save the graph
    graph_file = os.path.join(output_dirs["graphs"], f"{file_basename}_graph.pkl")
    with open(graph_file, 'wb') as f:
        pickle.dump(G, f)
    
    # Save the model
    model_file = os.path.join(output_dirs["models"], f"{file_basename}_model.pth")
    torch.save(model.state_dict(), model_file)
    
    # Save the node mapping
    mapping_file = os.path.join(output_dirs["mappings"], f"{file_basename}_mapping.pkl")
    with open(mapping_file, 'wb') as f:
        pickle.dump(node_mapping, f)
    
    return graph_file, model_file, mapping_file

def load_graph_and_model(graph_file, model_file, mapping_file, model_class, model_params):
    """Load the graph, trained model, and node mapping."""
    # Load the graph
    with open(graph_file, 'rb') as f:
        G = pickle.load(f)
    
    # Load the node mapping
    with open(mapping_file, 'rb') as f:
        node_mapping = pickle.load(f)
    
    # Initialize the model with the same parameters
    model = model_class(**model_params)
    model.load_state_dict(torch.load(model_file))
    model.eval()
    
    return G, model, node_mapping

def visualize_training(losses, output_file):
    """Visualize the training loss."""
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.savefig(output_file, format='png', dpi=300)
    plt.close()
    print(f"Training loss plot saved to {output_file}")

def visualize_graph_with_predictions(G, model, data, node_mapping, output_file):
    """Visualize the graph with node predictions from the model."""
    model.eval()
    
    # Check if graph is empty
    if G.number_of_nodes() == 0:
        print(f"Cannot visualize empty graph for {output_file}")
        return
    
    # Get model predictions
    with torch.no_grad():
        if hasattr(data, 'edge_attr') and data.edge_attr.size(0) > 0:
            pred = model(data.x, data.edge_index, data.edge_attr)
        else:
            pred = model(data.x, data.edge_index)
    
    pred = pred.numpy().flatten()
    
    # Map predictions back to original nodes
    reverse_mapping = {v: k for k, v in node_mapping.items()}
    node_predictions = {reverse_mapping[i]: float(pred[i]) for i in range(len(pred))}
    
    # Visualize graph
    plt.figure(figsize=(12, 10))
    
    try:
        pos = nx.spring_layout(G, seed=42)
        
        # Draw nodes with color based on predictions
        node_colors = [node_predictions[node] for node in G.nodes()]
        nodes = nx.draw_networkx_nodes(G, pos, node_size=500, node_color=node_colors, cmap=plt.cm.viridis)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.7, arrows=True, arrowsize=15)
        
        # Draw node labels only if there are fewer than 50 nodes (for readability)
        if G.number_of_nodes() < 50:
            nx.draw_networkx_labels(G, pos, font_size=8, font_family='sans-serif')
        
        plt.colorbar(nodes, label='Model Prediction')
        plt.axis('off')
        plt.tight_layout()
        
        plt.savefig(output_file, format='png', dpi=300)
        plt.close()
        print(f"Graph visualization saved to {output_file}")
    except Exception as e:
        print(f"Error visualizing graph: {str(e)}")

def save_run_info(output_dir, args, file_list, num_graphs):
    """Save information about the run to a text file."""
    info_file = os.path.join(output_dir, "run_info.txt")
    
    with open(info_file, 'w') as f:
        f.write("De Bruijn Graph Neural Network Training Run\n")
        f.write("==========================================\n\n")
        f.write(f"Date and Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("Parameters:\n")
        for arg, value in vars(args).items():
            f.write(f"  {arg}: {value}\n")
        
        f.write(f"\nNumber of FASTA files processed: {len(file_list)}\n")
        f.write(f"Number of graphs generated: {num_graphs}\n")
        
        f.write("\nFASTA files:\n")
        for i, file_path in enumerate(file_list, 1):
            f.write(f"  {i}. {os.path.basename(file_path)}\n")
    
    print(f"Run information saved to {info_file}")

# ------------- Main Function -------------

def main():
    parser = argparse.ArgumentParser(description='Train a GNN on multiple De Bruijn graphs from FASTA files.')
    parser.add_argument('-d', '--fasta_dir', help='Directory containing FASTA files')
    parser.add_argument('-f', '--fasta', help='Single FASTA file (alternative to directory)')
    parser.add_argument('-k', '--kmer', type=int, default=3, help='k-mer size (default: 3)')
    parser.add_argument('-m', '--model', type=str, default='gcn', choices=['gcn', 'gat'], 
                        help='GNN model to use (default: gcn)')
    parser.add_argument('-e', '--epochs', type=int, default=200, help='Number of training epochs (default: 200)')
    parser.add_argument('-l', '--learning_rate', type=float, default=0.01, help='Learning rate (default: 0.01)')
    parser.add_argument('-hc', '--hidden_channels', type=int, default=64, 
                        help='Number of hidden channels in GNN (default: 64)')
    parser.add_argument('-o', '--output_dir', default='output', help='Base output directory (default: output)')
    parser.add_argument('--load', action='store_true', help='Load saved graph and model instead of creating new ones')
    parser.add_argument('--run_dir', help='Directory of a previous run to load from')
    
    args = parser.parse_args()
    
    # Check for valid input
    if not args.load and not args.fasta_dir and not args.fasta:
        print("Error: Either --fasta_dir, --fasta, or --load with --run_dir must be specified.")
        return
    
    if args.load and not args.run_dir:
        print("Error: --run_dir must be specified when using --load.")
        return
    
    # Create output directory structure
    output_dirs = create_output_structure(args.output_dir)
    print(f"Created output directory structure at {output_dirs['main']}")
    
    try:
        if args.load:
            # Load existing graphs and models
            print(f"Loading graphs and model from {args.run_dir}...")
            
            # Find all graph files
            graph_files = glob.glob(os.path.join(args.run_dir, "graphs", "*_graph.pkl"))
            
            if not graph_files:
                print(f"No graph files found in {os.path.join(args.run_dir, 'graphs')}")
                return
            
            # Load the first model to get the model class
            model_type = 'gcn' if 'gcn' in args.model.lower() else 'gat'
            model_files = glob.glob(os.path.join(args.run_dir, "models", f"*_{model_type}_model.pth"))
            
            if not model_files:
                print(f"No model files found in {os.path.join(args.run_dir, 'models')}")
                return
            
            # Determine model parameters
            model_class = GCN if model_type == 'gcn' else GAT
            
            # Default parameters
            model_params = {
                'num_node_features': 5 * (args.kmer - 1),  # For k-mers of length k-1
                'hidden_channels': args.hidden_channels,
                'num_classes': 1  # Predicting one value per node
            }
            
            if model_class == GAT:
                model_params['heads'] = 8
            
            # Load each graph and its corresponding model
            graphs = []
            models = []
            node_mappings = []
            
            for graph_file in graph_files:
                base_name = os.path.basename(graph_file).replace("_graph.pkl", "")
                model_file = os.path.join(args.run_dir, "models", f"{base_name}_model.pth")
                mapping_file = os.path.join(args.run_dir, "mappings", f"{base_name}_mapping.pkl")
                
                if os.path.exists(model_file) and os.path.exists(mapping_file):
                    G, model, node_mapping = load_graph_and_model(
                        graph_file, model_file, mapping_file, model_class, model_params
                    )
                    graphs.append(G)
                    models.append(model)
                    node_mappings.append(node_mapping)
                    print(f"Loaded graph and model for {base_name}")
                else:
                    print(f"Missing files for {base_name}, skipping...")
            
            # Convert graphs to PyG data
            dataset = []
            for G in graphs:
                data, _ = prepare_graph_for_gnn(G)
                dataset.append(data)
            
            print(f"Loaded {len(graphs)} graphs and models")
            
        else:
            # Create new graphs and train model
            fasta_files = []
            
            if args.fasta_dir:
                # Get all FASTA files in the directory
                fasta_files = glob.glob(os.path.join(args.fasta_dir, "*.fasta")) + \
                              glob.glob(os.path.join(args.fasta_dir, "*.fa")) + \
                              glob.glob(os.path.join(args.fasta_dir, "*.fna")) + \
                              glob.glob(os.path.join(args.fasta_dir, "*.txt"))
            elif args.fasta:
                fasta_files = [args.fasta]
            
            if not fasta_files:
                print("No FASTA files found.")
                return
            
            print(f"Found {len(fasta_files)} FASTA files")
            
            # Save list of FASTA files for reference
            save_run_info(output_dirs["main"], args, fasta_files, len(fasta_files))
            
            # Process each FASTA file
            graphs = []
            node_mappings = []
            dataset = []
            file_basenames = []
            
            for fasta_file in fasta_files:
                file_basename = os.path.splitext(os.path.basename(fasta_file))[0]
                file_basenames.append(file_basename)
                
                print(f"Processing {file_basename}...")
                
                # Read sequences and build graph
                sequences = read_fasta(fasta_file)
                if not sequences:
                    print(f"No sequences found in {file_basename}, skipping...")
                    continue
                
                print(f"Building De Bruijn graph for {file_basename} with k={args.kmer}...")
                G = build_debruijn_graph(sequences, args.kmer)
                print(f"Created graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
                
                # Store graph
                graphs.append(G)
                
                # Prepare graph for GNN
                data, node_mapping = prepare_graph_for_gnn(G)
                dataset.append(data)
                node_mappings.append(node_mapping)
            
            if not graphs:
                print("No valid graphs were created from the FASTA files.")
                return
            
            # Create and train the model
            print(f"Creating {args.model.upper()} model...")
            
            # Determine feature size from the first graph's data
            num_node_features = dataset[0].x.size(1)
            
            if args.model.lower() == 'gcn':
                model = GCN(num_node_features=num_node_features, 
                          hidden_channels=args.hidden_channels, 
                          num_classes=1)
            else:  # GAT
                model = GAT(num_node_features=num_node_features, 
                          hidden_channels=args.hidden_channels, 
                          num_classes=1,
                          heads=8)
            
            print(f"Training model on {len(dataset)} graphs for {args.epochs} epochs...")
            model, losses = train_model(model, dataset, num_epochs=args.epochs, lr=args.learning_rate)
            
            # Save all graphs and the trained model
            print("Saving graphs and model...")
            
            for i, (G, node_mapping) in enumerate(zip(graphs, node_mappings)):
                file_basename = file_basenames[i] if i < len(file_basenames) else f"graph_{i}"
                model_type = args.model.lower()
                
                graph_file, model_file, mapping_file = save_graph_and_model(
                    G, model, node_mapping, f"{file_basename}_{model_type}", output_dirs
                )
            
            # Plot training loss
            loss_plot_file = os.path.join(output_dirs["plots"], "training_loss.png")
            visualize_training(losses, loss_plot_file)
            
            # Save the final combined model separately
            final_model_file = os.path.join(output_dirs["models"], f"combined_{args.model}_model.pth")
            torch.save(model.state_dict(), final_model_file)
            print(f"Combined model saved to {final_model_file}")
        
        # Visualize all graphs with predictions
        print("Visualizing graphs with model predictions...")
        
        for i, (G, data, node_mapping) in enumerate(zip(graphs, dataset, node_mappings)):
            try:
                file_basename = file_basenames[i] if 'file_basenames' in locals() and i < len(file_basenames) else f"graph_{i}"
                viz_file = os.path.join(output_dirs["plots"], f"{file_basename}_prediction.png")
                
                # Use the last trained model (or the loaded model if in load mode)
                if args.load:
                    current_model = models[i]
                else:
                    current_model = model
                
                visualize_graph_with_predictions(G, current_model, data, node_mapping, viz_file)
            except Exception as e:
                print(f"Error visualizing graph {i}: {str(e)}")
        
        print(f"Processing complete! All outputs saved to {output_dirs['main']}")
        
    except FileNotFoundError as e:
        print(f"Error: {str(e)}")
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()