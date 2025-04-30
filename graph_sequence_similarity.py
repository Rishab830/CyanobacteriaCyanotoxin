import os
import os
import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pickle
import argparse
from debruijn import build_debruijn_graph
from graph_nn_trainer import prepare_graph_for_gnn

def calculate_graph_similarity(G1, G2):
    """Calculate enhanced similarity between two graphs using multiple metrics."""
    # Get nodes and edges from both graphs
    nodes1 = set(G1.nodes())
    nodes2 = set(G2.nodes())
    edges1 = set(G1.edges())
    edges2 = set(G2.edges())
    
    # 1. Jaccard similarity for nodes
    if len(nodes1.union(nodes2)) == 0:
        node_jaccard = 0
    else:
        node_jaccard = len(nodes1.intersection(nodes2)) / len(nodes1.union(nodes2))
    
    # 2. Jaccard similarity for edges
    if len(edges1.union(edges2)) == 0:
        edge_jaccard = 0
    else:
        edge_jaccard = len(edges1.intersection(edges2)) / len(edges1.union(edges2))
    
    # 3. Degree distribution similarity
    # Get degree distributions
    degree_dist1 = [d for _, d in G1.degree()]
    degree_dist2 = [d for _, d in G2.degree()]
    
    # Handle empty graphs
    if not degree_dist1 or not degree_dist2:
        degree_similarity = 0
    else:
        # Calculate histogram overlap
        max_degree = max(max(degree_dist1) if degree_dist1 else 0, 
                         max(degree_dist2) if degree_dist2 else 0)
        
        bins = max(10, max_degree + 1)  # Ensure at least 10 bins
        
        hist1, _ = np.histogram(degree_dist1, bins=bins, range=(0, max_degree), density=True)
        hist2, _ = np.histogram(degree_dist2, bins=bins, range=(0, max_degree), density=True)
        
        # Calculate histogram intersection
        degree_similarity = np.sum(np.minimum(hist1, hist2))
    
    # 4. Graph density comparison
    if G1.number_of_nodes() > 0 and G2.number_of_nodes() > 0:
        density1 = nx.density(G1)
        density2 = nx.density(G2)
        density_diff = 1.0 - abs(density1 - density2)
    else:
        density_diff = 0
    
    # 5. Path length comparison (if graphs are connected)
    path_similarity = 0
    try:
        if nx.is_connected(G1.to_undirected()) and nx.is_connected(G2.to_undirected()):
            avg_path1 = nx.average_shortest_path_length(G1)
            avg_path2 = nx.average_shortest_path_length(G2)
            # Normalize path difference
            path_diff = abs(avg_path1 - avg_path2)
            path_similarity = 1.0 / (1.0 + path_diff)  # Convert to similarity
    except nx.NetworkXError:
        # Graph is not connected, skip this metric
        pass
    
    # Combine all metrics with appropriate weights
    weights = [0.25, 0.25, 0.2, 0.15, 0.15]
    metrics = [node_jaccard, edge_jaccard, degree_similarity, density_diff, path_similarity]
    
    similarity = sum(w * m for w, m in zip(weights, metrics))
    
    return similarity

def calculate_embedding_similarity(embedding1, embedding2):
    """Calculate similarity between embeddings using multiple metrics."""
    # Ensure embeddings are the same length
    min_len = min(len(embedding1), len(embedding2))
    v1 = embedding1[:min_len]
    v2 = embedding2[:min_len]
    
    # Convert to numpy arrays
    v1 = np.array(v1)
    v2 = np.array(v2)
    
    # 1. Cosine similarity
    dot_product = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    
    if norm1 == 0 or norm2 == 0:
        cosine_sim = 0
    else:
        cosine_sim = dot_product / (norm1 * norm2)
    
    # 2. Euclidean distance (converted to similarity)
    euclidean_dist = np.linalg.norm(v1 - v2)
    euclidean_sim = 1.0 / (1.0 + euclidean_dist)  # Convert distance to similarity
    
    # 3. Manhattan distance (converted to similarity)
    manhattan_dist = np.sum(np.abs(v1 - v2))
    manhattan_sim = 1.0 / (1.0 + manhattan_dist)  # Convert distance to similarity
    
    # 4. Distribution similarity (compare statistical properties)
    dist_sim = 1.0 - 0.25 * (
        abs(np.mean(v1) - np.mean(v2)) + 
        abs(np.std(v1) - np.std(v2)) +
        abs(np.min(v1) - np.min(v2)) +
        abs(np.max(v1) - np.max(v2))
    )
    dist_sim = max(0, min(1, dist_sim))  # Clamp to [0,1]
    
    # Combine metrics
    similarity = 0.4 * cosine_sim + 0.2 * euclidean_sim + 0.2 * manhattan_sim + 0.2 * dist_sim
    
    return similarity

def extract_sequence_features(sequence):
    """Extract relevant features from a DNA sequence."""
    # Count nucleotide frequencies
    length = len(sequence)
    if length == 0:
        return {
            'gc_content': 0,
            'at_content': 0,
            'gc_skew': 0,
            'at_skew': 0,
            'entropy': 0
        }
    
    counts = {
        'A': sequence.count('A'),
        'C': sequence.count('C'),
        'G': sequence.count('G'),
        'T': sequence.count('T')
    }
    
    # Calculate GC and AT content
    gc_content = (counts['G'] + counts['C']) / length
    at_content = (counts['A'] + counts['T']) / length
    
    # Calculate GC and AT skew
    if counts['G'] + counts['C'] > 0:
        gc_skew = (counts['G'] - counts['C']) / (counts['G'] + counts['C'])
    else:
        gc_skew = 0
        
    if counts['A'] + counts['T'] > 0:
        at_skew = (counts['A'] - counts['T']) / (counts['A'] + counts['T'])
    else:
        at_skew = 0
    
    # Calculate sequence entropy
    entropy = 0
    for base in ['A', 'C', 'G', 'T']:
        prob = counts[base] / length
        if prob > 0:
            entropy -= prob * np.log2(prob)
    
    return {
        'gc_content': gc_content,
        'at_content': at_content,
        'gc_skew': gc_skew,
        'at_skew': at_skew,
        'entropy': entropy
    }

def calculate_sequence_similarity(seq_features1, seq_features2):
    """Calculate similarity between sequence features."""
    # Calculate weighted similarity across all features
    features = ['gc_content', 'at_content', 'gc_skew', 'at_skew', 'entropy']
    weights = [0.25, 0.25, 0.2, 0.2, 0.1]
    
    similarity = 0
    for feat, weight in zip(features, weights):
        # Convert difference to similarity
        diff = abs(seq_features1[feat] - seq_features2[feat])
        # Use a sigmoid-like function to convert difference to similarity
        feat_sim = 1.0 / (1.0 + 5.0 * diff)  # Scale factor 5.0 can be adjusted
        similarity += weight * feat_sim
    
    return similarity

def classify_sequence(sequence, k, toxin_models, toxin_graphs, toxin_mappings):
    """Classify a given gene sequence with improved similarity scoring."""
    # Build De Bruijn graph for the input sequence
    G_input = build_debruijn_graph([sequence], k)
    
    # If the graph is empty, return early
    if G_input.number_of_nodes() == 0:
        return "Unable to classify: sequence too short or invalid"
    
    # Prepare graph for GNN
    data, _ = prepare_graph_for_gnn(G_input)
    
    # Extract sequence features
    input_seq_features = extract_sequence_features(sequence)
    
    # Sample some sequences from each toxin graph to calculate sequence features
    toxin_seq_features = {}
    for toxin_name, G in toxin_graphs.items():
        # Extract representative k-mers from graph nodes
        if G.nodes():
            # Sample paths through the graph
            sampled_sequences = []
            try:
                # Try to find some paths through the graph
                for i in range(min(5, G.number_of_nodes())):
                    start_node = np.random.choice(list(G.nodes()))
                    path = [start_node]
                    current = start_node
                    
                    # Walk through the graph to create a sequence
                    for _ in range(20):  # Limit path length
                        successors = list(G.successors(current))
                        if not successors:
                            break
                        next_node = np.random.choice(successors)
                        path.append(next_node)
                        current = next_node
                    
                    # Combine path into a sequence
                    seq = path[0]
                    for node in path[1:]:
                        seq += node[-1]
                    
                    sampled_sequences.append(seq)
            except:
                # If random walk fails, just use node labels
                sampled_sequences = list(G.nodes())
            
            # Calculate features from sampled sequences
            toxin_feature_list = [extract_sequence_features(seq) for seq in sampled_sequences]
            
            # Average features
            avg_features = {}
            for feature in ['gc_content', 'at_content', 'gc_skew', 'at_skew', 'entropy']:
                avg_features[feature] = np.mean([f[feature] for f in toxin_feature_list])
            
            toxin_seq_features[toxin_name] = avg_features
    
    # Calculate similarity scores for each toxin
    scores = {}
    embeddings = {}
    detailed_scores = {}
    
    # Store input model prediction for later comparison
    input_predictions = {}
    
    for toxin_name, model in toxin_models.items():
        # Get model predictions for input sequence
        with torch.no_grad():
            if hasattr(data, 'edge_attr') and data.edge_attr.size(0) > 0:
                pred = model(data.x, data.edge_index, data.edge_attr)
            else:
                pred = model(data.x, data.edge_index)
        
        input_pred_vector = pred.numpy().flatten()
        input_predictions[toxin_name] = input_pred_vector
        embeddings[toxin_name] = input_pred_vector
        
        # 1. Calculate graph topology similarity
        graph_sim = calculate_graph_similarity(G_input, toxin_graphs[toxin_name])
        
        # 2. Calculate sequence feature similarity
        if toxin_name in toxin_seq_features:
            seq_sim = calculate_sequence_similarity(input_seq_features, toxin_seq_features[toxin_name])
        else:
            seq_sim = 0.5  # Default if features couldn't be calculated
        
        # 3. Calculate model prediction pattern similarity
        # First, get reference predictions by running random nodes through the model
        ref_embedding = np.zeros(input_pred_vector.shape)
        
        if toxin_graphs[toxin_name].nodes():
            # Sample nodes from this toxin's graph
            sampled_nodes = np.random.choice(
                list(toxin_graphs[toxin_name].nodes()), 
                size=min(10, len(toxin_graphs[toxin_name].nodes())),
                replace=True
            )
            
            # Get average prediction pattern from the model for this toxin
            node_features = []
            for node in sampled_nodes:
                # Create one-hot encoding for the node
                nucleotides = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
                feature = np.zeros(5 * len(node))
                for i, nuc in enumerate(node):
                    if nuc in nucleotides:
                        feature[i * 5 + nucleotides[nuc]] = 1
                    else:
                        feature[i * 5 + 4] = 1
                node_features.append(feature)
            
            if node_features:
                # Run these features through the model
                x = torch.tensor(np.array(node_features), dtype=torch.float)
                with torch.no_grad():
                    sample_pred = model(x, torch.zeros((2, 0), dtype=torch.long))
                ref_embedding = sample_pred.numpy().mean(axis=0)
        
        # Calculate embedding similarity
        emb_sim = calculate_embedding_similarity(input_pred_vector, ref_embedding)
        
        # Combine scores with different weights
        # Weight graph similarity higher as it's more reliable
        combined_score = 0.5 * graph_sim + 0.3 * seq_sim + 0.2 * emb_sim
        
        # Store all scores
        scores[toxin_name] = combined_score
        detailed_scores[toxin_name] = {
            'graph_similarity': graph_sim,
            'sequence_similarity': seq_sim,
            'embedding_similarity': emb_sim,
            'combined_score': combined_score
        }
    
    # Cross-compare embeddings between all toxins to find relative similarities
    for toxin1 in toxin_models:
        for toxin2 in toxin_models:
            if toxin1 != toxin2:
                cross_sim = calculate_embedding_similarity(
                    input_predictions[toxin1], 
                    input_predictions[toxin2]
                )
                # Penalize toxins with similar predictions (suggesting less distinguishing power)
                scores[toxin1] *= (1.0 - 0.1 * cross_sim)
                scores[toxin2] *= (1.0 - 0.1 * cross_sim)
    
    # Normalize scores to [0,1] range
    if scores:
        min_score = min(scores.values())
        max_score = max(scores.values())
        score_range = max_score - min_score
        
        if score_range > 0:
            for toxin in scores:
                scores[toxin] = (scores[toxin] - min_score) / score_range
    
    # Find the toxin with the highest score
    if scores:
        best_match = max(scores.items(), key=lambda x: x[1])
        toxin_name = best_match[0]
        confidence = best_match[1]
        
        return {
            "classification": toxin_name,
            "confidence": confidence,
            "all_scores": scores,
            "detailed_scores": detailed_scores,
            "embeddings": embeddings
        }
    else:
        return {
            "classification": "Unknown",
            "confidence": 0,
            "all_scores": {},
            "detailed_scores": {},
            "embeddings": {}
        }

def visualize_classification_results(sequence_name, results, output_dir):
    """Visualize classification results with detailed score breakdown."""
    # Create scores plot
    plt.figure(figsize=(12, 8))
    
    # Setup the plot
    toxin_names = list(results["all_scores"].keys())
    scores = list(results["all_scores"].values())
    
    # Sort by scores
    sorted_indices = np.argsort(scores)[::-1]  # Descending order
    toxin_names = [toxin_names[i] for i in sorted_indices]
    scores = [scores[i] for i in sorted_indices]
    
    # Main bar chart of overall scores
    plt.subplot(2, 1, 1)
    bars = plt.barh(toxin_names, scores, color='skyblue')
    
    # Highlight the best match
    best_idx = toxin_names.index(results["classification"])
    bars[best_idx].set_color('tomato')
    
    plt.xlabel('Overall Similarity Score')
    plt.ylabel('Cyanotoxin Type')
    plt.title(f'Classification Results for {sequence_name}')
    
    # Add detailed score breakdown
    plt.subplot(2, 1, 2)
    
    # We'll show detailed scores for top 3 matches
    top_n = min(3, len(toxin_names))
    top_toxins = toxin_names[:top_n]
    
    x = np.arange(len(top_toxins))
    width = 0.2
    
    # Extract detailed scores for plotting
    graph_scores = [results['detailed_scores'][toxin]['graph_similarity'] for toxin in top_toxins]
    seq_scores = [results['detailed_scores'][toxin]['sequence_similarity'] for toxin in top_toxins]
    emb_scores = [results['detailed_scores'][toxin]['embedding_similarity'] for toxin in top_toxins]
    
    # Plot grouped bars
    plt.bar(x - width, graph_scores, width, label='Graph Similarity', color='lightblue')
    plt.bar(x, seq_scores, width, label='Sequence Similarity', color='lightgreen')
    plt.bar(x + width, emb_scores, width, label='Embedding Similarity', color='salmon')
    
    plt.xlabel('Toxin Type')
    plt.ylabel('Score Component')
    plt.title('Detailed Score Breakdown for Top Matches')
    plt.xticks(x, top_toxins)
    plt.legend()
    
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
            colors = plt.cm.tab10(np.linspace(0, 1, len(labels)))
            
            for i, (label, color) in enumerate(zip(labels, colors)):
                plt.scatter(X_pca[i, 0], X_pca[i, 1], label=label, color=color)
                plt.annotate(label, (X_pca[i, 0], X_pca[i, 1]), 
                           textcoords="offset points", xytext=(0,10), ha='center')
            
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
    """Write detailed classification results to a text file."""
    output_file = os.path.join(output_dir, f"{sequence_name}_report.txt")
    
    with open(output_file, 'w') as f:
        f.write(f"Classification Report for Sequence: {sequence_name}\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Best Match: {results['classification']}\n")
        f.write(f"Confidence Score: {results['confidence']:.4f}\n\n")
        
        f.write("Detailed Similarity Scores:\n")
        f.write("-"*60 + "\n")
        
        # Sort toxins by overall score from highest to lowest
        sorted_scores = sorted(results['all_scores'].items(), key=lambda x: x[1], reverse=True)
        
        f.write("Toxin Type        | Overall | Graph Sim | Seq Sim | Emb Sim \n")
        f.write("-"*60 + "\n")
        
        for toxin_name, score in sorted_scores:
            if toxin_name in results['detailed_scores']:
                details = results['detailed_scores'][toxin_name]
                graph_sim = details['graph_similarity']
                seq_sim = details['sequence_similarity']
                emb_sim = details['embedding_similarity']
                
                f.write(f"{toxin_name:<17} | {score:.4f} | {graph_sim:.4f} | {seq_sim:.4f} | {emb_sim:.4f}\n")
        
        f.write("\nScore Interpretation:\n")
        f.write("-"*60 + "\n")
        f.write("Graph Similarity: Measures how similar the De Bruijn graph structure is\n")
        f.write("Sequence Similarity: Compares nucleotide composition and patterns\n")
        f.write("Embedding Similarity: Compares neural network learned features\n")
        f.write("Overall: Weighted combination of all similarity metrics\n")
        
        # Add confidence interpretation
        f.write("\nConfidence Assessment:\n")
        f.write("-"*60 + "\n")
        
        # Get scores of top 2 matches
        if len(sorted_scores) >= 2:
            top_diff = sorted_scores[0][1] - sorted_scores[1][1]
            if top_diff > 0.5:
                confidence_msg = "Very High - Clear distinction from other toxins"
            elif top_diff > 0.3:
                confidence_msg = "High - Significant difference from next best match"
            elif top_diff > 0.1:
                confidence_msg = "Moderate - Some difference from next best match"
            else:
                confidence_msg = "Low - Similar scores with other toxin types"
        else:
            confidence_msg = "Unknown - Only one toxin type available for comparison"
        
        f.write(f"Confidence Assessment: {confidence_msg}\n")
        
        # Add recommendation
        if results['confidence'] < 0.7 or (len(sorted_scores) >= 2 and sorted_scores[0][1] - sorted_scores[1][1] < 0.2):
            f.write("\nRecommendation: Consider additional validation methods to confirm this classification.\n")
    
    return output_file

def main(args):
    print("Starting Cyanotoxin Classification System...")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load pre-trained models and graphs
    print(f"Loading models from: {args.models_dir}")
    toxin_models = {}
    toxin_graphs = {}
    toxin_mappings = {}
    
    # Load all available toxin models
    for filename in os.listdir(args.models_dir + "/models"):
        if filename.endswith(".pth"):
            toxin_name = filename.split('_model')[0]
            model_path = os.path.join(args.models_dir, filename)
            
            print(f"Loading model for {toxin_name}...")
            try:
                # Load the GNN model
                model = torch.load(model_path)
                model.eval()  # Set to evaluation mode
                toxin_models[toxin_name] = model
                
                # Look for corresponding graph file
                graph_path = os.path.join(args.models_dir, f"{toxin_name}_graph.pkl")
                if os.path.exists(graph_path):
                    with open(graph_path, 'rb') as f:
                        toxin_graphs[toxin_name] = pickle.load(f)
                    print(f"  Loaded graph with {toxin_graphs[toxin_name].number_of_nodes()} nodes")
                
                # Look for mapping file (if available)
                mapping_path = os.path.join(args.models_dir, f"{toxin_name}_mapping.pkl")
                if os.path.exists(mapping_path):
                    with open(mapping_path, 'rb') as f:
                        toxin_mappings[toxin_name] = pickle.load(f)
            except Exception as e:
                print(f"  Error loading model: {str(e)}")
    
    print(f"Successfully loaded {len(toxin_models)} toxin models")
    
    # Load test sequences
    test_sequences = {}
    
    # If a specific sequence file is provided
    if args.sequence_file:
        try:
            with open(args.sequence_file, 'r') as f:
                lines = f.readlines()
            
            current_seq_name = None
            current_seq = ""
            
            for line in lines:
                line = line.strip()
                if line.startswith('>'):
                    # Save previous sequence if it exists
                    if current_seq_name:
                        test_sequences[current_seq_name] = current_seq
                    
                    # Start new sequence
                    current_seq_name = line[1:].split()[0]  # Take first word after '>'
                    current_seq = ""
                else:
                    current_seq += line
            
            # Save last sequence
            if current_seq_name:
                test_sequences[current_seq_name] = current_seq
                
            print(f"Loaded {len(test_sequences)} sequences from {args.sequence_file}")
            
        except Exception as e:
            print(f"Error loading sequence file: {str(e)}")
    
    # If no sequences were loaded, add some example sequences
    if not test_sequences:
        print("No sequence file provided or empty file. Using example sequences instead.")
        
        # Example sequence for a microcystin gene (simplified for demonstration)
        test_sequences["example_microcystin"] = "ATGAGAATTTTCACAACCTGGAATTGTAGCAACACTCACCCAGGAGCGAATTGAGCGCATTCGGTAGTTCATTGAGTTCGCAAGCCAGATGCAGTTGCGGAACAGGCGTTAACTCGCTCTCAAATTGATTCGATTTGCCCCCAAACCTTACCAAAGGCTGCTGCTTTAGTAGCTCAACTCAATCGAATGATCGAATCGTTTACTAGTCAATTGATCGAGCTCGTAAAAATTCCTCAACTTCCTCGAGATCGATCAATTACTTCTAGCGCCGGTCAGCGACTCTTACATGCTATACGATTGGGAGCAAGCAACCGAGCCTCTAGCCTTAGGCGC"
        
        # Example sequence for a saxitoxin gene (simplified for demonstration)
        test_sequences["example_saxitoxin"] = "ATGCTTGACTGCATTCTCAGCCCGGAAGTATTCCTCACCGCGATGACCATCGCAGATATCGCCAGGATTGCTGAGCTCATGCTAGCCACAATGGAAGGAGACGGACTCTATTTCTACGTCAAGGGTCGTCTCGAAGAGTACGCCGACATCATGAAGTCGATCCGCATCGCTCACAAGAACGGAGGTAACCAGTACGCTCGCGTGATGTCCGGTCACGGTGTAGCGATTAAGCAACCGGATGCGTTCAAGGCGATCCTGCAATTGCTGGAGTTGGAGCTCAACCTGCCCGGTAGTGGCAGCATCGGTGTCGCCACC"
        
        # Example sequence for a cylindrospermopsin gene (simplified for demonstration)
        test_sequences["example_cylindrospermopsin"] = "ATGGCGCAAATTCCGGAACAGCCCGAACGCCTTGAGCGCGCCCTCTTCGTCAACACCCACGAAGCCGCGCGCACCCTCGCCGAGTTGCGCGAGCAGTTGGCCGAAAACCCCGACCCCGAGCAGCAGTGGCGCCAGCGGCAGGCGCTGGATATCGAGACCGCCCCCGGCCATCCCAACCACGAGCCCGCGTTGCGCGCCCTGTTGCAGTACCTGCGCGCGCAACTGGAGCAGGACCTGACCTCGCTCCTGAATTCGCTCTCGGACAAGTGGGTGCTGCGCCCCGAGGTGAGCGCGCTCGAGGCGGAGAAGCGCGAGTGG"
        
        # Example sequence for a nodularin gene (simplified for demonstration)
        test_sequences["example_nodularin"] = "ATGTCTACCAGTAACACCAACTCTACTGGAGGAAACATAAACACTGATTCGTCGTTAACAACTGAAGCATCCACAGTAGCATTGTTGTCAGTTTCTCATGTCAATCTTGTGCAAAGTTATGGATTAACTGAGCAACTCCAACCTAAAGCATTGACTCGGCACGAATTGGCACGTAGTGATTCTGTGCTAGATGCTGCTGAAATTTATCGTCATGACGGTGAGGCATTTAATGCTGTGTCGGCGCTAGAACTTCTGCGTTCGGAAGCCACTCCACAAGCAACTCGTTTTATTGATCTCGGACAGGCCTTAATTACCCTAGAATATAAGCGCATTGA"
        
        # Create a test sequence that's a mix of two toxins (for demonstration)
        hybrid_seq = test_sequences["example_microcystin"][:100] + test_sequences["example_saxitoxin"][100:200]
        test_sequences["hybrid_toxin"] = hybrid_seq
    
    # Process each sequence
    print("\nBeginning sequence classification...")
    
    # Set k-mer size for De Bruijn graph
    k = args.kmer_size
    
    for seq_name, sequence in test_sequences.items():
        print(f"\nProcessing sequence: {seq_name} (length: {len(sequence)})")
        
        # Make sure sequence is uppercase
        sequence = sequence.upper()
        
        # Skip sequences that are too short
        if len(sequence) < k:
            print(f"  Sequence too short for k-mer size {k}. Skipping.")
            continue
        
        # Classify the sequence
        try:
            results = classify_sequence(sequence, k, toxin_models, toxin_graphs, toxin_mappings)
            
            # Print classification results
            print(f"  Classification result: {results['classification']}")
            print(f"  Confidence score: {results['confidence']:.4f}")
            
            # Show top 3 matches
            print("  Top matches:")
            sorted_scores = sorted(results['all_scores'].items(), key=lambda x: x[1], reverse=True)
            for i, (toxin, score) in enumerate(sorted_scores[:3], 1):
                print(f"    {i}. {toxin}: {score:.4f}")
            
            # Generate visualizations
            image_file = visualize_classification_results(seq_name, results, args.output_dir)
            print(f"  Classification visualization saved to: {image_file}")
            
            # Generate report
            report_file = write_classification_report(seq_name, results, args.output_dir)
            print(f"  Detailed report saved to: {report_file}")
            
        except Exception as e:
            print(f"  Error processing sequence {seq_name}: {str(e)}")
    
    print("\nClassification process complete!")
    print(f"All results saved to: {args.output_dir}")

if __name__ == "__main__":
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='Cyanotoxin Gene Sequence Classifier')
    parser.add_argument('-m', '--models_dir', type=str, default='models', help='Directory containing pre-trained models and graphs')
    parser.add_argument('-o', '--output_dir', type=str, default='results', help='Directory to save results')
    parser.add_argument('-s', '--sequence_file', type=str, default='', help='FASTA file containing sequences to classify')
    parser.add_argument('-k', '--kmer_size', type=int, default=5, help='Size of k-mers for De Bruijn graph construction')
    
    args = parser.parse_args()
    
    # Run the main function
    main(args)