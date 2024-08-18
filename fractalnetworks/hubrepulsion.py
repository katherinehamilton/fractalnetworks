"""This file contains function to analyse the effect of hub-hub repulsion in fractal networks."""

# Mathematics modules
import numpy as np

# Utility imports
import itertools

def find_degree_distribution(G):
    """
    Calculates the degree distribution of the network.

    Returns:
        (list): A list of degrees in the network. The i-th value is the degree of the i-th node.

    """
    return G.degree()

def identify_hubs(G, degrees=None):
    """
    Identify the hubs of the network using the Z-score, where hubs are those nodes with Z > 3.

    Args:
        G (igraph.Graph): The network to be analysed.
        degrees (list) (opt): A list of degrees in the network. If already calculated, this parameter can be passed to prevent duplication. Default is None, in which case the function calculates the distribution internally.

    Returns:
        (list): A list of nodes with Z > 3.
    """
    # Find the degree distribution of the network.
    if not degrees:
        degrees = find_degree_distribution(G)

    # Find the mean and standard deviation of the degree distribution.
    mean_degree = np.mean(degrees)
    stdv_degree = np.std(degrees)

    # Initialise an empty dictionary to store the Z-score for each node.
    z_scores = dict.fromkeys(range(G.vcount()))

    # For each node calculate and store the Z-score
    for i in z_scores:
        z_scores[i] = (degrees[i] - mean_degree) / stdv_degree

    # Find the nodes which have a Z-score greater than 3
    dd_dict_filtered = {i: degrees[i] for i in range(G.vcount()) if z_scores[i] > 3}

    # Return a list of these nodes.
    return list(dd_dict_filtered.keys())


def identify_hubs_by_mean(G, degrees=None, factor=2):
    """
    Identify the hubs of the network using the mean, where hubs are those nodes with degree greater than some factor (usually 2) times the mean.

    Args:
        G (igraph.Graph): The network to be analysed.
        degrees (list) (opt): A list of degrees in the network. If already calculated, this parameter can be passed to prevent duplication. Default is None, in which case the function calculates the distribution internally.
        factor (int) (opt): The factor by which the mean is multiplied to identify hubs. Default is 2.

    Returns:
        (list): A list of hubs.
    """
    # Find the degree distribution of the network.
    if not degrees:
        degrees = find_degree_distribution(G)

    # Find the mean degree
    mean_degree = np.mean(degrees)

    dd_dict_filtered = {i: degrees[i] for i in range(G.vcount()) if degrees[i] > factor * mean_degree}

    return list(dd_dict_filtered.keys())


def identify_hubs_by_percentile(G, degrees=None, percentile=90):
    """
    Identify the hubs of the network by percentile, where hubs are those in some percentile of the degree distribution (normally 90%)

    Args:
        G (igraph.Graph): The network to be analysed.
        degrees (list) (opt): A list of degrees in the network. If already calculated, this parameter can be passed to prevent duplication. Default is None, in which case the function calculates the distribution internally.
        percentile (int) (opt): The percentile hubs belong to. Default is 90.

    Returns:
        (list): A list of hubs.
    """
    # Find the degree distribution of the network.
    if not degrees:
        degrees = find_degree_distribution(G)

    # Create a dictionary of node indexes and degrees.
    dd_dict = {i: degrees[i] for i in range(G.vcount())}

    # Find a list of nodes sorted by degree
    sorted_nodes = [node for node, k in sorted(dd_dict.items(), key=lambda x: x[1])]

    # Find the cutoff index
    cutoff = round(percentile * (G.vcount() / 100))

    # Return nodes in the specified percentile
    return sorted_nodes[cutoff:]


def mean_hub_degree(G, hubs=None, hub_method=identify_hubs, degrees=None):
    """
    Calculates the mean hub degree in the network.

    Args:
        G (igraph.Graph): The network to be analysed.
        hubs (list) (opt): A list of hubs in the network, If already calculated, this parameter can be passed to prevent duplication. Default is None, in which case the function finds the hubs.
        hub_method (func) (opt): If the hubs are to be calculated, then this parameter specifies the method used to find hubs. Default is identify_hubs, which uses the Z score.
        degrees (list) (opt): A list of degrees in the network. If already calculated, this parameter can be passed to prevent duplication. Default is None, in which case the function calculates the distribution internally.
    """

    # Find the degree distribution of the network.
    if not degrees:
        degrees = find_degree_distribution(G)

    # Find the hubs of the network
    if not hubs:
        hubs = hub_method(G, degrees=None)

    # Create a list of degrees of hubs
    hub_degrees = [degrees[i] for i in hubs]

    # Return the mean of the hub degrees.
    return np.mean(hub_degrees)


def find_hub_hub_edges(G, hubs=None, hub_method=identify_hubs, degrees=None):
    """
    Finds the number of edges which connect hubs to other hubs.

    Args:
        G (igraph.Graph): The network to be analysed.
        hubs (list) (opt): A list of hubs in the network, If already calculated, this parameter can be passed to prevent duplication. Default is None, in which case the function finds the hubs.
        hub_method (func) (opt): If the hubs are to be calculated, then this parameter specifies the method used to find hubs. Default is identify_hubs, which uses the Z score.
        degrees (list) (opt): A list of degrees in the network. If already calculated, this parameter can be passed to prevent duplication. Default is None, in which case the function calculates the distribution internally.

    Returns:
        no_of_hub_edges (int): The number of edges between hubs.
    """

    # Find the degree distribution of the network.
    if not degrees:
        degrees = find_degree_distribution(G)

    # Find the hubs of the network
    if not hubs:
        hubs = hub_method(G, degrees=None)

    # Initialise a variable to count the number of hub-hub edges
    no_of_hub_edges = 0

    # Iterate through every pair of hubs
    for hub_u, hub_v in itertools.combinations(hubs, 2):
        # If there is an edge between this pair of hubs, then add one to the total number of hub-hub edges
        if G.are_adjacent(hub_u, hub_v):
            no_of_hub_edges += 1

    # Return the total number of edges between hubs
    return no_of_hub_edges


def find_hub_hub_path_nodes(G, hubs=None, hub_method=identify_hubs, degrees=None):
    """
    Finds the nodes on the shortest paths between pairs of hubs.

    Args:
        G (igraph.Graph): The network to be analysed.
        hubs (list) (opt): A list of hubs in the network, If already calculated, this parameter can be passed to prevent duplication. Default is None, in which case the function finds the hubs.
        hub_method (func) (opt): If the hubs are to be calculated, then this parameter specifies the method used to find hubs. Default is identify_hubs, which uses the Z score.
        degrees (list) (opt): A list of degrees in the network. If already calculated, this parameter can be passed to prevent duplication. Default is None, in which case the function calculates the distribution internally.

    Returns:
        hub_hub_path_nodes (list): A list of nodes on the paths between hubs.
    """

    # Find the degree distribution of the network.
    if not degrees:
        degrees = find_degree_distribution(G)

    # Find the hubs of the network
    if not hubs:
        hubs = hub_method(G, degrees=None)

    # Initialise an empty set to store the nodes.
    hub_hub_path_nodes = set()

    # Iterate through each possible pair of hubs
    for hub_u, hub_v in itertools.combinations(hubs, 2):
        # Find all the shortest paths between this pair of hubs.
        for path in G.get_all_shortest_paths(hub_u, to=hub_v):
            # Add all nodes on this path to the set of nodes.
            hub_hub_path_nodes.update(path)

    # Remove the hubs from the list
    hub_hub_path_nodes = hub_hub_path_nodes.difference(set(hubs))

    # Return the list of nodes.
    return hub_hub_path_nodes


def find_hub_hub_path_node_occurrences(G, hubs=None, hub_method=identify_hubs, degrees=None):
    """
    Finds the nodes on the shortest paths between pairs of hubs, and the number of times they appear.

    Args:
        G (igraph.Graph): The network to be analysed.
        hubs (list) (opt): A list of hubs in the network, If already calculated, this parameter can be passed to prevent duplication. Default is None, in which case the function finds the hubs.
        hub_method (func) (opt): If the hubs are to be calculated, then this parameter specifies the method used to find hubs. Default is identify_hubs, which uses the Z score.
        degrees (list) (opt): A list of degrees in the network. If already calculated, this parameter can be passed to prevent duplication. Default is None, in which case the function calculates the distribution internally.

    Returns:
        hub_hub_path_nodes (list): A list of nodes on the paths between hubs.
        occurrences (dict): A dictionary with nodes as keys and the number of times they appear on shortest paths as values.
    """

    # Find the degree distribution of the network.
    if not degrees:
        degrees = find_degree_distribution(G)

    # Find the hubs of the network
    if not hubs:
        hubs = hub_method(G, degrees=None)

    # Initialise a set to store the nodes.
    hub_hub_path_nodes = set()

    # Initialise a dictionary to store the number of occurrences
    occurrences = dict.fromkeys(range(G.vcount()), 0)

    # Iterate through each pair of hubs
    for hub_u, hub_v in itertools.combinations(hubs, 2):
        # Iterate through every shortest path between that pair of hubs.
        for path in G.get_all_shortest_paths(hub_u, to=hub_v):
            # Add all nodes on this path to the set of nodes.
            hub_hub_path_nodes.update(path)
            # For each non-hub on the path, add the number of occurrences to the dictionary
            for node in path:
                if node not in hubs:
                    occurrences[node] += path.count(node)

    # Remove the hubs from the list
    hub_hub_path_nodes = hub_hub_path_nodes.difference(set(hubs))

    # Remove all nodes from the dictionary with no occurrences
    occurrences = {node: occurrences[node] for node in occurrences if occurrences[node] > 0}

    return hub_hub_path_nodes, occurrences


def calculate_HCS(G, hubs=None, hub_method=identify_hubs, degrees=None, normalise_by_number_of_hubs=False, normalise_by_mean_hub_degree=False, normalise_by_mean_degree=False, normalise_by_number_of_edges=False):
    """
    Calculate the Hub Connectivity Score (HCS) of the network. This is the average number of hubs each hub is adjacent to.

    Args:
        G (igraph.Graph): The network to be analysed.
        hubs (list) (opt): A list of hubs in the network, If already calculated, this parameter can be passed to prevent duplication. Default is None, in which case the function finds the hubs.
        hub_method (func) (opt): If the hubs are to be calculated, then this parameter specifies the method used to find hubs. Default is identify_hubs, which uses the Z score.
        degrees (list) (opt): A list of degrees in the network. If already calculated, this parameter can be passed to prevent duplication. Default is None, in which case the function calculates the distribution internally.
        normalised (bool) (opt): If True, the Hub Connectivity Score is normalised by the number of hubs in the network.

    Returns:
        (float): The Hub Connectivity Score of the network.
    """

    # Find the degree distribution of the network.
    if not degrees:
        degrees = find_degree_distribution(G)

    # Find the hubs of the network
    if not hubs:
        hubs = hub_method(G, degrees=None)

    # If there are no hubs in the network, the HCS has no value.
    if len(hubs) == 0:
        return np.NaN

    # Find the number of hub-hub edges
    E_hub = find_hub_hub_edges(G, hubs=hubs, degrees=degrees)

    # Find the number of hubs
    N_hub = len(hubs)

    # Calculate the HCS
    HCS = 2 * E_hub / N_hub

    if normalise_by_number_of_hubs:
        HCS = HCS / N_hub

    if normalise_by_mean_hub_degree:
        mean_k = mean_hub_degree(G, hubs=hubs, degrees=degrees)
        HCS = HCS / mean_k

    if normalise_by_mean_degree:
        mean_k = np.mean(G.degree())
        HCS = HCS / mean_k

    if normalise_by_number_of_edges:
        HCS = HCS / G.ecount()


    # Return the HCS
    return HCS