#!/usr/bin/env python3

import os
import torch
import numpy as np
import pickle
from Bio import SeqIO
import argparse
import matplotlib.pyplot as plt
import glob
from graph_nn_trainer import GAT, GCN, prepare_graph_for_gnn
from debruijn import build_debruijn_graph

# De Bruijn Graph functions from original script
def read_fasta(file_path):
    """Read sequences from a FASTA file."""
    sequences = []
    for record in SeqIO.parse(file_path, "fasta"):
        sequences.append(str(record.seq).upper())
    return sequences

def generate_kmers(sequence, k):
    """Generate k-mers from a sequence."""
    return [sequence[i:i+k] for i in range(len(sequence) - k + 1)]

# New classification functions
def load_models_and_graphs(model_dir, graph_dir, mapping_dir):
    """Load all saved models, graphs, and mappings from directories."""
    model_files = glob.glob(os.path.join(model_dir, "*_model.pth"))
    toxin_models = {}
    toxin_graphs = {}
    toxin_mappings = {}
    
    for model_file in model_files:
        base_name = os.path.basename(model_file).split('_model.pth')[0]
        toxin_name = base_name.split('_')[0]  # Assumes filename format: toxinname_modeltype_model.pth
        
        graph_file = os.path.join(graph_dir, f"{base_name}_graph.pkl")
        mapping_file = os.path.join(mapping_dir, f"{base_name}_mapping.pkl")
        
        if os.path.exists(graph_file) and os.path.exists(mapping_file):
            # Determine model type
            if "_gcn_" in model_file.lower():
                model_class = GCN
            else:
                model_class = GAT
            
            # Load graph
            with open(graph_file, 'rb') as f:
                G = pickle.load(f)
            
            # Load mapping
            with open(mapping_file, 'rb') as f:
                node_mapping = pickle.load(f)
            
            # Determine model parameters from the graph
            sample_node = list(G.nodes())[0] if G.nodes() else ""
            k = len(sample_node) + 1  # k-1 mers are stored as nodes
            num_node_features = 5 * len(sample_node)  # 5 possible nucleotides
            
            # Create model
            model_params = {
                'num_node_features': num_node_features,
                'hidden_channels': 64,  # Default
                'num_classes': 1
            }
            
            if model_class == GAT:
                model_params['heads'] = 8
            
            model = model_class(**model_params)
            
            # Load model weights
            model.load_state_dict(torch.load(model_file))
            model.eval()
            
            toxin_models[toxin_name] = model
            toxin_graphs[toxin_name] = G
            toxin_mappings[toxin_name] = node_mapping
            
            print(f"Loaded model for toxin: {toxin_name}")
    
    return toxin_models, toxin_graphs, toxin_mappings

def classify_sequence(sequence, k, toxin_models, toxin_graphs, toxin_mappings):
    """Classify a given gene sequence to identify which cyanotoxin gene it belongs to."""
    # Build De Bruijn graph for the input sequence
    G_input = build_debruijn_graph([sequence], k)
    
    # If the graph is empty, return early
    if G_input.number_of_nodes() == 0:
        return "Unable to classify: sequence too short or invalid"
    
    # Prepare graph for GNN
    data, _ = prepare_graph_for_gnn(G_input)
    
    # Calculate similarity scores for each toxin
    scores = {}
    embeddings = {}
    
    for toxin_name, model in toxin_models.items():
        # Get model predictions
        with torch.no_grad():
            if hasattr(data, 'edge_attr') and data.edge_attr.size(0) > 0:
                pred = model(data.x, data.edge_index, data.edge_attr)
            else:
                pred = model(data.x, data.edge_index)
        
        # Calculate graph similarity based on model predictions
        pred_vector = pred.numpy().flatten()
        embeddings[toxin_name] = pred_vector
        
        # Calculate a similarity score based on graph structure
        graph_similarity = calculate_graph_similarity(G_input, toxin_graphs[toxin_name])
        model_score = np.mean(pred_vector)  # Use mean prediction as a score
        
        # Combine scores (weighted average)
        combined_score = 0.7 * graph_similarity + 0.3 * model_score
        scores[toxin_name] = combined_score
    
    # Find the toxin with the highest score
    if scores:
        best_match = max(scores.items(), key=lambda x: x[1])
        toxin_name = best_match[0]
        confidence = best_match[1]
        
        return {
            "classification": toxin_name,
            "confidence": confidence,
            "all_scores": scores,
            "embeddings": embeddings
        }
    else:
        return {
            "classification": "Unknown",
            "confidence": 0,
            "all_scores": {},
            "embeddings": {}
        }

def calculate_graph_similarity(G1, G2):
    """Calculate similarity between two graphs based on node and edge overlap."""
    # Get nodes and edges from both graphs
    nodes1 = set(G1.nodes())
    nodes2 = set(G2.nodes())
    edges1 = set(G1.edges())
    edges2 = set(G2.edges())
    
    # Calculate Jaccard similarity for nodes and edges
    nodes_intersection = len(nodes1.intersection(nodes2))
    nodes_union = len(nodes1.union(nodes2))
    
    if nodes_union == 0:
        node_similarity = 0
    else:
        node_similarity = nodes_intersection / nodes_union
    
    # Handle case where one or both graphs have no edges
    if not edges1 or not edges2:
        if not edges1 and not edges2:
            edge_similarity = 1  # Both have no edges, consider them similar
        else:
            edge_similarity = 0  # One has edges, one doesn't
    else:
        edges_intersection = len(edges1.intersection(edges2))
        edges_union = len(edges1.union(edges2))
        edge_similarity = edges_intersection / edges_union
    
    # Weighted similarity
    similarity = 0.6 * node_similarity + 0.4 * edge_similarity
    
    return similarity

def visualize_classification_results(sequence_name, results, output_dir):
    """Visualize classification results."""
    # Create scores plot
    plt.figure(figsize=(10, 6))
    toxin_names = list(results["all_scores"].keys())
    scores = list(results["all_scores"].values())
    
    # Sort by scores
    sorted_indices = np.argsort(scores)
    toxin_names = [toxin_names[i] for i in sorted_indices]
    scores = [scores[i] for i in sorted_indices]
    
    plt.barh(toxin_names, scores, color='skyblue')
    plt.xlabel('Similarity Score')
    plt.ylabel('Cyanotoxin Type')
    plt.title(f'Classification Results for {sequence_name}')
    
    # Highlight the best match
    best_idx = toxin_names.index(results["classification"])
    plt.barh([toxin_names[best_idx]], [scores[best_idx]], color='tomato')
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, f"{sequence_name}_classification.png")
    plt.savefig(output_file, dpi=300)
    plt.close()
    
    # If there are embeddings, visualize them using PCA
    if results["embeddings"] and len(results["embeddings"]) > 1:
        try:
            from sklearn.decomposition import PCA
            
            # Prepare embeddings for PCA
            embedding_arrays = []
            labels = []
            
            for toxin_name, embedding in results["embeddings"].items():
                # Ensure embeddings have same length by padding if necessary
                embedding_arrays.append(embedding)
                labels.append(toxin_name)
            
            # Find minimum length and truncate all to that length
            min_len = min(len(e) for e in embedding_arrays)
            embedding_arrays = [e[:min_len] for e in embedding_arrays]
            
            # Convert to numpy array
            X = np.array(embedding_arrays)
            
            # Perform PCA
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X)
            
            # Plot
            plt.figure(figsize=(8, 6))
            for i, label in enumerate(labels):
                plt.scatter(X_pca[i, 0], X_pca[i, 1], label=label)
            
            plt.xlabel('PC1')
            plt.ylabel('PC2')
            plt.title(f'Embedding Space for {sequence_name}')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            
            output_file = os.path.join(output_dir, f"{sequence_name}_embedding.png")
            plt.savefig(output_file, dpi=300)
            plt.close()
        except Exception as e:
            print(f"Error creating embedding visualization: {str(e)}")
    
    return output_file

def write_classification_report(sequence_name, results, output_dir):
    """Write classification results to a text file."""
    output_file = os.path.join(output_dir, f"{sequence_name}_report.txt")
    
    with open(output_file, 'w') as f:
        f.write(f"Classification Report for Sequence: {sequence_name}\n")
        f.write("="*50 + "\n\n")
        
        f.write(f"Best Match: {results['classification']}\n")
        f.write(f"Confidence Score: {results['confidence']:.4f}\n\n")
        
        f.write("All Similarity Scores:\n")
        f.write("-"*30 + "\n")
        
        # Sort scores from highest to lowest
        sorted_scores = sorted(results['all_scores'].items(), key=lambda x: x[1], reverse=True)
        
        for toxin_name, score in sorted_scores:
            f.write(f"{toxin_name}: {score:.4f}\n")
    
    return output_file

def main():
    parser = argparse.ArgumentParser(description='Classify gene sequences to identify cyanotoxin genes using trained GNN models.')
    parser.add_argument('-s', '--sequence', help='File containing the sequence to classify (FASTA format)')
    parser.add_argument('-k', '--kmer', type=int, default=5, help='k-mer size (default: 5)')
    parser.add_argument('-m', '--input_dir', required=True, help='Directory containing saved GNN models')
    parser.add_argument('-o', '--output_dir', default='classification_results', help='Output directory for results')
    
    args = parser.parse_args()
    
    # Check required arguments
    if not args.sequence:
        print("Error: --sequence argument is required.")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load models, graphs, and mappings
    print("Loading saved models, graphs, and mappings...")
    toxin_models, toxin_graphs, toxin_mappings = load_models_and_graphs(
        args.input_dir + "/models", args.input_dir + "/graphs", args.input_dir + "/mappings"
    )
    
    if not toxin_models:
        print("Error: No valid models found.")
        return
    
    print(f"Loaded {len(toxin_models)} toxin models.")
    
    # Read the sequence to classify
    sequences = read_fasta(args.sequence)
    if not sequences:
        print(f"Error: No sequences found in {args.sequence}")
        return
    
    # Process each sequence
    for i, sequence in enumerate(sequences):
        sequence_name = f"sequence_{i+1}"
        
        print(f"Classifying {sequence_name}...")
        
        # Get the k value from a sample graph if not specified
        k = args.kmer
        if k <= 0:
            sample_toxin = next(iter(toxin_graphs))
            sample_node = next(iter(toxin_graphs[sample_toxin].nodes()))
            k = len(sample_node) + 1
            print(f"Using k={k} derived from saved graphs")
        
        # Classify the sequence
        results = classify_sequence(sequence, k, toxin_models, toxin_graphs, toxin_mappings)
        
        if results["classification"] == "Unknown":
            print(f"Unable to classify {sequence_name}")
            continue
        
        print(f"Classification result: {results['classification']} (confidence: {results['confidence']:.4f})")
        
        # Visualize results
        viz_file = visualize_classification_results(sequence_name, results, args.output_dir)
        report_file = write_classification_report(sequence_name, results, args.output_dir)
        
        print(f"Visualization saved to: {viz_file}")
        print(f"Report saved to: {report_file}")
    
    print("Classification complete!")

if __name__ == "__main__":
    main()