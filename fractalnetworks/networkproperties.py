"""This module contains functions to analyse the properties of networks"""

# Mathematics modules
from .hubrepulsion import *


def summarise_graph(G, skip_diam=False, skip_aspl=False):
    """
    Summarises the key attributes of a given network.

    Args:
        G (igraph.Graph): The network to be analysed.
        skip_diam (Bool) (opt): If True, then do not calculate the diameter of the graph. This is recommended for large graphs. The default is False.
        skip_aspl (Bool) (opt): If True, then do not calculate the average shortest path length of the graph. This is recommended for large graphs. The default is False.

    Returns:
        None
    """
    # Display the size and order of the network.
    print("Network has {0} nodes and {1} edges.".format(G.vcount(), G.ecount()))

    # Calculate and display the average degree of the network.
    degree_dist = [G.degree(node) for node in G.vs]
    avg_degree = sum(degree_dist) / len(degree_dist)
    print("The average degree of the network is {0}.".format(avg_degree))

    # If chosen, find the average shortest path length.
    if not skip_aspl:
        print("The average shortest path length is {0}.".format(np.mean(G.distances())))

    # If chosen, find the diameter.
    if not skip_diam:
        print("The diameter is {0}.".format(G.diameter()))


def find_distances(G):
    """
    Finds the diameter and average shortest path length of a network.

    Args:
        G (igraph.Graph): The network to be analysed.

    Returns:
        diam (int): The diameter of the network.
        aspl (float): The average shortest path length of the network.
    """
    # Calculate the diameter
    diam = G.diameter()
    # Calculate the average shortest path length
    aspl = G.average_path_length()

    # Return both values
    return diam, aspl


def mean_hub_distance(G, hubs=None, hub_method=identify_hubs, normalised=False, degrees=None):
    """
    Finds the mean distance between hubs.
    The normalised mean distance is the mean distance between hubs over the mean distance between any pair of nodes in the network.

    Args:
        G (igraph.Graph): The network to be analysed.
        hubs (list) (opt): A list of hubs in the network, If already calculated, this parameter can be passed to prevent duplication. Default is None, in which case the function finds the hubs.
        hub_method (func) (opt): If the hubs are to be calculated, then this parameter specifies the method used to find hubs. Default is identify_hubs, which uses the Z score.
        normalised (Bool) (opt): If true, the distances are normalised over the average distance in the network. Default is False.
        degrees (list) (opt): A list of degrees in the network. If already calculated, this parameter can be passed to prevent duplication. Default is None, in which case the function calculates the distribution internally.

    Returns:
        mean_distance (float): The mean distance between pairs of hubs in the network.
        hub_distances (list): A list of distances between pairs of hubs.
    """
    # Find the hubs of the network
    if not hubs:
        hubs = hub_method(G, degrees=degrees)

    # If there are no hubs in the network, then there is no mean distance.
    if len(hubs) == 0:
        return np.NaN, []

    # Initialise an empty list for hub distances
    hub_distances = []

    # Add the distance between each pair of hubs to the list
    for hub_u, hub_v in itertools.combinations(hubs, 2):
        hub_distances.append(len(G.get_shortest_paths(hub_u, to=hub_v, output="vpath")[0]) - 1)

    # Find the mean
    mean_distance = np.mean(hub_distances)

    # Normalise the mean with the average distance between any pair of nodes.
    if normalised:
        mean_distance = mean_distance / G.average_path_length()

    return mean_distance, hub_distances


def hub_distance_distribution(G, hubs=None, hub_method=identify_hubs, degrees=None):
    """
    Find the distribution of distances between hubs.

    Args:
        G (igraph.Graph): The network to be analysed.
        hubs (list) (opt): A list of hubs in the network, If already calculated, this parameter can be passed to prevent duplication. Default is None, in which case the function finds the hubs.
        hub_method (func) (opt): If the hubs are to be calculated, then this parameter specifies the method used to find hubs. Default is identify_hubs, which uses the Z score.
        degrees (list) (opt): A list of degrees in the network. If already calculated, this parameter can be passed to prevent duplication. Default is None, in which case the function calculates the distribution internally.

    Returns:
        distance_distributions (dict): A dictionary with distances as keys, the probability of two hubs being separated by that distance as the values.
    """
    # Find the hubs of the network
    if not hubs:
        hubs = hub_method(G, degrees=degrees)

    # If there are no hubs in the network, then there is no mean distance.
    if len(hubs) == 0:
        return np.NaN, []

    # Find the number of possible edges.
    no_of_possible_edges = (len(hubs) * (len(hubs) - 1)) / 2

    # Find a list of distances between pairs of hubs in the network
    _, hub_distances = mean_hub_distance(G, hubs=hubs)

    # Initialise an empty dictionary to store the probabilities.
    distance_distribution = dict.fromkeys(hub_distances)

    # For each distance, find the probability of it being the distance between two hubs.
    for distance in hub_distances:
        distance_distribution[distance] = hub_distances.count(distance) / no_of_possible_edges

    # Return the dictionary
    return distance_distribution


def find_clustering_coefficient(G):
    """
    Finds the clustering coefficient T of a given network.

    Args:
        G (igraph.Graph): The network to be analysed.

    Returns:
        (float): The clustering coefficient (transitivity) T of the network.
    """
    return G.transitivity_undirected()