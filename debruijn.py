#!/usr/bin/env python3

import networkx as nx
import matplotlib.pyplot as plt
from Bio import SeqIO
import argparse

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

def visualize_graph(G, output_file=None):
    """Visualize the De Bruijn graph."""
    plt.figure(figsize=(12, 8))
    
    # Position nodes using spring layout
    pos = nx.spring_layout(G, seed=42)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color='lightblue')
    
    # Draw edges with weights as labels
    edge_weights = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.7)
    
    # Draw node labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')
    
    # Draw edge weight labels
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_weights)
    
    plt.axis('off')
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, format='png', dpi=300)
        print(f"Graph saved to {output_file}")
    else:
        plt.show()

def save_graph_info(G, output_file):
    """Save graph information to a text file."""
    with open(output_file, 'w') as f:
        f.write(f"Number of nodes: {G.number_of_nodes()}\n")
        f.write(f"Number of edges: {G.number_of_edges()}\n\n")
        
        f.write("Nodes:\n")
        for node in sorted(G.nodes()):
            f.write(f"  {node}\n")
        
        f.write("\nEdges (with weights):\n")
        for u, v, weight in sorted(G.edges(data='weight')):
            f.write(f"  {u} -> {v}: {weight}\n")

def main():
    parser = argparse.ArgumentParser(description='Convert FASTA sequences to a De Bruijn graph.')
    parser.add_argument('-f', '--fasta', required=True, help='Path to FASTA file')
    parser.add_argument('-k', '--kmer', type=int, default=3, help='k-mer size (default: 3)')
    parser.add_argument('-o', '--output', help='Output file name for graph visualization')
    parser.add_argument('-i', '--info', help='Output file name for graph information')
    
    args = parser.parse_args()
    
    try:
        sequences = read_fasta(args.fasta)
        if not sequences:
            print("No sequences found in the FASTA file.")
            return
        
        print(f"Building De Bruijn graph with k={args.kmer}...")
        G = build_debruijn_graph(sequences, args.kmer)
        
        print(f"Created graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
        
        if args.info:
            save_graph_info(G, args.info)
            print(f"Graph information saved to {args.info}")
        
        visualize_graph(G, args.output)
        
    except FileNotFoundError:
        print(f"Error: File {args.fasta} not found.")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()