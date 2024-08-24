"""This module includes functions to find and analyse the network's skeleton"""

from .betweennesscentrality import *


def find_skel(G, ebcs=None):
    """
    Finds the skeleton of the given network.

    Args:
        G (igraph.Graph)             : The network to be analysed.
        ebcs (:obj:`list`, optional) : A list of edge betweenness centralities.
                                       If already calculated, passing this parameter prevents duplication.
                                       Default is None, meaning it will be calculated internally.

    Returns:
        H (igraph.Graph) : The skeleton of the network G
    """

    # Find the edge betweenness centralities
    if not ebcs:
        ebcs = find_edge_betweenness_centralities(G)

    # Create a list of tuples of the form (edge ID, edge betweenness centrality)
    edges_to_ebcs = [(G.es()[i].tuple, ebcs[i]) for i in range(G.ecount())]

    # Sort the list in order of decreasing betweenness
    edges_to_ebcs.sort(key=lambda x: x[1], reverse=True)

    # Create a copy of the network with no edges
    H = G.copy()
    H.delete_edges()

    # Iterate through all the edges in the network, in order of decreasing edge betweenness centrality,
    for edge, ebc in edges_to_ebcs:
        # Check if adding this edge would create a cycle
        paths = H.get_shortest_paths(edge[0], to=edge[1], output="vpath")
        if len(paths) == 1 and len(paths[0]) == 0:
            # If there is no cycle, add the edge
            H.add_edge(edge[0], edge[1])
        # When the network is connected, the spanning tree is complete.
        if H.is_connected():
            return H

    # Return the skeleton
    return H


def find_skeleton_eids(G, H=None, ebcs=None):
    """
    Finds a list of edge IDs for edges in the skeleton of G.

    Args:
        G (igraph.Graph)                  : The network to be analysed.
        H (:obj:`igraph.Graph`, optional) : The skeleton of the network, if known. Default is None.
        ebcs (:obj:`list`, optional)      : The edge betweenness centralities of the network, if known. Default is None.

    Returns:
        skeleton_edges (list) : List of IDs of edges in G which are in the skeleton of G.
    """

    # Find the skeleton of the network
    if not H:
        H = find_skel(G, ebcs=ebcs)

    # Initialise an empty list of edge IDs
    skeleton_edges = []

    # Find the edge ID of each edge in the network and add it to the list
    for edge in H.es():
        source = edge.tuple[0]
        target = edge.tuple[1]
        eid = G.get_eid(source, target)
        skeleton_edges.append(eid)

    # Return the complete list.
    return skeleton_edges


def find_skeleton_edge_betweenness(G, H=None, ebcs=None):
    """
    Finds the edge betweenness centralities of edges in the skeleton of the network.

    Args:
        G (igraph.Graph)                  : The network to be analysed.
        H (:obj:`igraph.Graph`, optional) : The skeleton of the network, if known. Default is None.
        ebcs (:obj:`list`, optional)      : The edge betweenness centralities of the network, if known. Default is None.

    Returns:
        skeleton_ebcs (list)     : A list of edge betweenness centralities of edges in the skeleton.
        non_skeleton_ebcs (list) : A list of edge betweenness centralities of edges not in the skeleton.
    """

    # Find the betweenness centrality distribution of the network.
    if not ebcs:
        ebcs = find_edge_betweenness_centralities(G)

    # Find the IDs of edges in the skeleton
    skeleton_edges = find_skeleton_eids(G, H=H, ebcs=ebcs)

    # Find a list of edge betweenness centralities in the skeleton and not in the skeleton
    skeleton_ebcs = [ebcs[i] for i in skeleton_edges]
    non_skeleton_ebcs = [ebcs[i] for i in range(G.ecount()) if i not in skeleton_edges]

    # Return both lists.
    return skeleton_ebcs, non_skeleton_ebcs
