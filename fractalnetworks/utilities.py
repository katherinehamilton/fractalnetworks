""" This module contains utility functions used throughout the fractal network toolkit."""

# Network analysis modules
from igraph import Graph
import igraph
import networkx as nx

# Utility modules
import os

# Reading Graphs

def read_graph(filepath):
    """
    Checks the filetype and calls the correct function to read the graph.
    For any file not in .gml format, an additional .gml file is created.
    Currently supports .mtx, .txt, .edge (.EDGE) and .gml filetypes.

    Args:
        filepath (str): Path for the network file.

    Returns:
        G (igraph.Graph): The network from the file stored as a graph.
    """

    # .gml files
    if filepath.endswith(".gml"):
        G = Graph.Load(filepath)

    elif filepath.endswith(".mtx"):
        G = read_mtx_graph_format(filepath)
        new_filepath = filepath.replace(".mtx", ".gml")
        nx.write_gml(G, new_filepath)
        G = Graph.Load(new_filepath)

    # .txt files
    elif filepath.endswith(".txt"):
        G = nx.read_edgelist(filepath)
        new_filepath = filepath.replace(".txt", ".gml")
        nx.write_gml(G, new_filepath)
        G = Graph.Load(new_filepath)

    # .edge or .EDGE files
    elif filepath.endswith(".edges") or filepath.endswith(".EDGES"):
        G = nx.read_weighted_edgelist(filepath)
        new_filepath = filepath.replace(".edges", ".gml")
        new_filepath = new_filepath.replace(".EDGES", ".gml")
        nx.write_gml(G, new_filepath)
        G = Graph.Load(new_filepath)

    # Raise an error if an unknown file type is used.
    else:
        raise ValueError('This filetype is not supported for network analysis.')

    return G