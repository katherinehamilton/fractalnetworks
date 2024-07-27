"""This module contains functions to analyse the betweenness centralities of nodes and edges in fractal and non-fractal networks."""

# Other module imports
from .hubrepulsion import *

# Mathematics modules
import matplotlib.pyplot as plt
import pandas as pd


def find_betweenness_centralities(G):
    """
    Calculates the normalised betweenness centrality distribution of the network.

    Returns:
        (list): A list of betweenness centralities in the network. The i-th value is the betweenness centrality of the i-th node.

    """
    # Calculate the normalising constant for the betweenness centrality.
    N = G.vcount()
    normalising_constant = 2 / ((N - 2) * (N - 1))

    # Calculate the betweenness centralities.
    bc = G.betweenness()

    # Normalise the betweenness centralities
    bc = [v * normalising_constant for v in bc]

    return bc


def plot_degree_distribution(G, degrees=None, save_path=None, plot=True):
    """
    Plots the degree distribution of a given network on a histogram.

    Args:
        G (igraph.Graph): The network to be analysed.
        degrees (list) (opt): A list of degrees in the network. If already calculated, this parameter can be passed to prevent duplication. Default is None, in which case the function calculates the distribution internally.
        save_path (str) (opt): If a save path is given then the histogram is saved as a .png file. Default is None.
        plot (Bool) (opt): If True, the histogram is displayed. Default is True.

    Returns:
        None
    """

    # Find the degree distribution of the network.
    if not degrees:
        degrees = find_degree_distribution(G)

    # Plot a degree distribution histogram
    plt.hist(degrees, bins=100, color="navy")

    # Label the axis
    plt.xlabel("Degree $k$")
    plt.ylabel("Frequency")

    # If a save path is given, then save the file in that location.
    if save_path:
        plt.savefig(save_path)

    # If plot is True display the graph
    if plot:
        plt.show()

    # Close the figure
    plt.close()


def plot_betweenness_centralities(G, bcs=None, save_path=None, plot=True):
    """
    Plots the betweenness centrality distribution of a given network on a histogram.

    Args:
        G (igraph.Graph): The network to be analysed.
        bcs (list) (opt): A list of betweenness centralities in the network. If already calculated, this parameter can be passed to prevent duplication. Default is None, in which case the function calculates the distribution internally.
        save_path (str) (opt): If a save path is given then the histogram is saved as a .png file. Default is None.
        plot (Bool) (opt): If True, the histogram is displayed. Default is True.

    Returns:
        None
    """
    # Find the betweenness centrality distribution of the network.
    if not bcs:
        bcs = find_betweenness_centralities(G)

    # Plot a degree distribution histogram
    plt.hist(bcs, bins=100, color="crimson")

    # Label the axis
    plt.xlabel("Betweenness Centrality $C$")
    plt.ylabel("Frequency")

    # If a save path is given, then save the file in that location.
    if save_path:
        plt.savefig(save_path)

    # If plot is True display the graph
    if plot:
        plt.show()

    # Close the figure
    plt.close()


def calc_betweenness_degree_correlation(G, bcs=None, degrees=None):
    """
    Calculates the Pearson correlation coefficient for the degree and betweenness centrality of the nodes in a network.
    Fractal networks are hypothesised to be less correlated in this regard than non-fractal networks.

    Args:
        G (igraph.Graph): The network to be analysed.
        bcs (list) (opt): A list of betweenness centralities in the network. If already calculated, this parameter can be passed to prevent duplication. Default is None, in which case the function calculates the distribution internally.
        degrees (list) (opt): A list of degrees in the network. If already calculated, this parameter can be passed to prevent duplication. Default is None, in which case the function calculates the distribution internally.


    Returns:
        (float): Pearson's correlation coefficient.
    """

    # Find the betweenness centrality distribution of the network.
    if not bcs:
        bcs = find_betweenness_centralities(G)

    # Find the degree distribution of the network.
    if not degrees:
        degrees = find_degree_distribution(G)

    # Convert both to a Pandas Series
    bc_series = pd.Series(bcs)
    dd_series = pd.Series(degrees)

    # Return the correlation coefficient for the variables.
    return bc_series.corr(dd_series)


def calc_betweenness_degree_correlation_non_hubs(G, hubs=None, hub_method=identify_hubs, bcs=None, degrees=None):
    """
    Calculates the Pearson correlation coefficient for the degree and betweenness centrality of the non-hub nodes in a network.

    Args:
        G (igraph.Graph): The network to be analysed.
        hubs (list) (opt): A list of nodes indexes for hubs in the network. If no list is passed, the function with find the hubs with hub_method.
        hub_method (func): The method by which to calculate hubs, if not give. Default is identify_hubs.
        bcs (list) (opt): A list of betweenness centralities in the network. If already calculated, this parameter can be passed to prevent duplication. Default is None, in which case the function calculates the distribution internally.
        degrees (list) (opt): A list of degrees in the network. If already calculated, this parameter can be passed to prevent duplication. Default is None, in which case the function calculates the distribution internally.

    Returns:
        (float): Pearson's correlation coefficient.
    """
    # Find the hubs of the network
    if not hubs:
        hubs = hub_method(G)

    # Find the betweenness centrality distribution of the network.
    if not bcs:
        bcs = find_betweenness_centralities(G)

    # Find the degree distribution of the network.
    if not degrees:
        degrees = find_degree_distribution(G)

    # Only consider the betweenness centrality for non hubs
    filtered_bc = [bcs[i] for i in range(G.vcount()) if i not in hubs]
    # Only consider the degree for non hubs
    filtered_dd = [degrees[i] for i in range(G.vcount()) if i not in hubs]

    # Convert both to a Pandas Series
    bc_series = pd.Series(filtered_bc)
    dd_series = pd.Series(filtered_dd)

    # Return the correlation coefficient for the variables.
    return bc_series.corr(dd_series)


def calc_betweenness_degree_correlation_hubs(G, hubs=None, hub_method=identify_hubs, bcs=None, degrees=None):
    """
    Calculates the Pearson correlation coefficient for the degree and betweenness centrality of the hub nodes in a network.

    Args:
        G (igraph.Graph): The network to be analysed.
        hubs (list) (opt): A list of nodes indexes for hubs in the network. If no list is passed, the function with find the hubs with hub_method.
        hub_method (func): The method by which to calculate hubs, if not give. Default is identify_hubs.
        bcs (list) (opt): A list of betweenness centralities in the network. If already calculated, this parameter can be passed to prevent duplication. Default is None, in which case the function calculates the distribution internally.
        degrees (list) (opt): A list of degrees in the network. If already calculated, this parameter can be passed to prevent duplication. Default is None, in which case the function calculates the distribution internally.

    Returns:
        (float): Pearson's correlation coefficient.
    """
    # Find the hubs of the network
    if not hubs:
        hubs = hub_method(G)

    # Find the betweenness centrality distribution of the network.
    if not bcs:
        bcs = find_betweenness_centralities(G)

    # Find the degree distribution of the network.
    if not degrees:
        degrees = find_degree_distribution(G)

    # Only consider the betweenness centrality for non hubs
    filtered_bc = [bcs[i] for i in range(G.vcount()) if i in hubs]
    # Only consider the degree for non hubs
    filtered_dd = [degrees[i] for i in range(G.vcount()) if i in hubs]

    # Convert both to a Pandas Series
    bc_series = pd.Series(filtered_bc)
    dd_series = pd.Series(filtered_dd)

    # Return the correlation coefficient for the variables.
    return bc_series.corr(dd_series)


def analyse_betweenness_centrality(G, bcs=None):
    """
    Returns key attributes of the betweenness centrality distribution of a network.
    Namely, finds the maximum, minimum and mean betweenness centrality, as well as the standard deviation and number of outliers with greater than expected betweenness.

    Args:
        G (igraph.Graph): Network to be analysed.
        bcs (list) (opt): A list of betweenness centralities in the network. If already calculated, this parameter can be passed to prevent duplication. Default is None, in which case the function calculates the distribution internally.

    Returns:
        max_bc (float): The maximum betweenness centrality in the network.
        min_bc (float): The minimum betweenness centrality in the network.
        mean_bc (float): The mean betweenness centrality in the network.
        stdv_bc (float): The standard deviation of the betweenness centrality distribution in the network.
        no_of_high_bc_nodes (int): Number of nodes with significantly higher betweenness centrality than expected.
        mean_max_bc (float): The mean betweenness centrality of the top 10% of nodes in the network.
    """
    # Find the betweenness centrality distribution of the network.
    if not bcs:
        bcs = find_betweenness_centralities(G)

    # Find the maximum betweenness centrality
    max_bc = max(bcs)
    # Find the minimum betweenness centrality
    min_bc = min(bcs)
    # Find the mean betweenness centrality
    mean_bc = np.mean(bcs)
    # Find the standard deviation of the betweenness centrality
    stdv_bc = np.std(bcs)

    # Outliers are those with betweenness centrality 3 standard deviations higher than the mean.
    # Calculate the cutoff value
    cutoff = mean_bc + 3 * stdv_bc
    # Find nodes with betweenness centralities greater than this cutoff.
    high_bcs = [v for v in bcs if v > cutoff]
    # The number of nodes with high betweenness centrality is the length of this list.
    no_of_high_bc_nodes = len(high_bcs)

    # Find the top 10% of betweenness centralities.
    cutoff = round(len(bcs) * 0.9)
    bcs.sort()
    top_10_percent = bcs[cutoff:]
    # Find the mean of these values.
    mean_max_bc = np.mean(top_10_percent)

    # Return the maximum, minimum, mean, standard deviation, number of outliers and mean of the top 10% of betweenness centralities.
    return max_bc, min_bc, mean_bc, stdv_bc, no_of_high_bc_nodes, mean_max_bc


def find_edge_betweenness_centralities(G):
    """
    Calculates the normalised betweenness centrality distribution of the network.

    Returns:
        (list): A list of betweenness centralities in the network. The i-th value is the betweenness centrality of the i-th node.

    """
    # Calculate the normalising constant for the betweenness centrality.
    N = G.vcount()
    normalising_constant = 2 / (N * (N - 1))

    # Calculate the betweenness centralities.
    ebc = G.edge_betweenness()

    # Normalise the betweenness centralities
    ebc = [v * normalising_constant for v in ebc]

    return ebc


def edge_vertex_betweenness_correlation(G, bcs=None, ebcs=None):
    """
    Calculates the correlation between the betweenness centrality of edges and the betweenness centrality of its endpoints.

    Args:
        G (igraph.Graph): The network to be analysed.
        bcs (list) (opt): A list of betweenness centralities in the network. If already calculated, this parameter can be passed to prevent duplication. Default is None, in which case the function calculates the distribution internally.
        ebcs (list) (opt): A list of betweenness centralities in the network. If already calculated, this parameter can be passed to prevent duplication. Default is None, in which case the function calculates the distribution internally.

    """

    # Find the betweenness centralities
    if not bcs:
        bcs = find_betweenness_centralities(G)

    # Find the edge betweenness centralities
    if not ebcs:
        ebcs = find_edge_betweenness_centralities(G)

    # Initialise an empty list to store the edge betweenness centralities of each edge
    edge_bcs = []
    # Initialise an empty list to store the node betweenness centralities of every edge endpoint.
    node_bcs = []

    # Iterate through every edge in the network
    for edge_index in range(G.ecount()):
        # Find the endpoints of the edge
        edge = G.es()[edge_index]
        source = edge.tuple[0]
        target = edge.tuple[1]

        # Find the betweenness centrality of the edge and its endpoints.
        edge_bc = ebcs[edge_index]
        source_bc = bcs[source]
        target_bc = bcs[target]

        # Add the betweenness centrality of the edge and its source to the lists.
        edge_bcs.append(edge_bc)
        node_bcs.append(source_bc)

        # Add the betweenness centrality of the edge and its target to the lists.
        edge_bcs.append(edge_bc)
        node_bcs.append(target_bc)

    # Convert the lists to pandas Series
    edge_bcs_series = pd.Series(edge_bcs)
    node_bcs_series = pd.Series(node_bcs)

    # Return the correlation coefficient for the variables.
    return edge_bcs_series.corr(node_bcs_series)


def edge_betweenness_degree_correlation(G, degrees=None, ebcs=None):
    """
    Calculates the correlation between the betweenness centrality of edges and the degree of its endpoints.

    Args:
        G (igraph.Graph): The network to be analysed.
        degrees (list) (opt): A list of degrees in the network. If already calculated, this parameter can be passed to prevent duplication. Default is None, in which case the function calculates the distribution internally.
        ebcs (list) (opt): A list of betweenness centralities in the network. If already calculated, this parameter can be passed to prevent duplication. Default is None, in which case the function calculates the distribution internally.

    """

    # Find the degrees of nodes
    if not degrees:
        degrees = find_degree_distribution(G)

    # Find the edge betweenness centralities
    if not ebcs:
        ebcs = find_edge_betweenness_centralities(G)

    # Initialise an empty list to store the edge betweenness centralities of each edge
    edge_bcs = []
    # Initialise an empty list to store the degree of every edge endpoint.
    node_degrees = []

    # Iterate through every edge in the network
    for edge_index in range(G.ecount()):
        # Find the endpoints of the edge
        edge = G.es()[edge_index]
        source = edge.tuple[0]
        target = edge.tuple[1]

        # Find the betweenness centrality of the edge and the degrees of its endpoints.
        edge_bc = ebcs[edge_index]
        source_k = degrees[source]
        target_k = degrees[target]

        # Add the betweenness centrality of the edge and its source's degree to the lists.
        edge_bcs.append(edge_bc)
        node_degrees.append(source_k)

        # Add the betweenness centrality of the edge and its target's degree to the lists.
        edge_bcs.append(edge_bc)
        node_degrees.append(target_k)

    # Convert the lists to pandas Series
    edge_bcs_series = pd.Series(edge_bcs)
    node_degrees_series = pd.Series(node_degrees)

    # Return the correlation coefficient for the variables.
    return edge_bcs_series.corr(node_degrees_series)


def find_hub_betweenness(G, hubs=None, hub_method=identify_hubs, bcs=None):
    """
    Finds lists of betweenness centrality for the hubs and non-hubs in the network.

    Args:
        G (igraph.Graph): The network to be analysed.
        hubs (list) (opt): A list of hubs in the network, If already calculated, this parameter can be passed to prevent duplication. Default is None, in which case the function finds the hubs.
        hub_method (func) (opt): If the hubs are to be calculated, then this parameter specifies the method used to find hubs. Default is identify_hubs, which uses the Z score.
        bcs (list) (opt): A list of betweenness centralities in the network. If already calculated, this parameter can be passed to prevent duplication. Default is None, in which case the function calculates the distribution internally.

    Returns:
        hub_bcs (list): A list of betweenness centralities of hubs.
        non_hub_bcs (list): A list of betweenness centralities of non-hubs.
    """

    # Find the hubs of the network
    if not hubs:
        hubs = hub_method(G, degrees=None)

    # Find the betweenness centrality distribution of the network.
    if not bcs:
        bcs = find_betweenness_centralities(G)

    # Initialise empty lists for the betweenness centralities of hubs and non-hubs.
    hub_bcs = []
    non_hub_bcs = []

    # Iterate over every node in the network
    for node in [v for v in range(G.vcount())]:
        # If the node is a hub, add its betweenness centrality to the hub list.
        if node in hubs:
            hub_bcs.append(bcs[node])
        # If the node is a non-hub, add its betweenness centrality to the non-hub list.
        else:
            non_hub_bcs.append(bcs[node])

    # Return both lists.
    return hub_bcs, non_hub_bcs


def hub_hub_path_betweenness(G, bcs=None, degrees=None, hubs=None, hub_method=identify_hubs, hub_hub_path_nodes=None):
    """
    Analyses the betweenness of nodes on the paths between hubs with the number of their occurrences.
    Returns a list of the betweenness centralities of these nodes, and the Pearson correlation coefficient between the number of occurrences on such a path and their centrality.

    Args:
        G (igraph.Graph): The network to be analysed.
        bcs (list) (opt): A list of betweenness centralities in the network. If already calculated, this parameter can be passed to prevent duplication. Default is None, in which case the function calculates the distribution internally.
        degrees (list) (opt): A list of degrees in the network. If already calculated, this parameter can be passed to prevent duplication. Default is None, in which case the function calculates the distribution internally.
        hubs (list) (opt): A list of hubs in the network, If already calculated, this parameter can be passed to prevent duplication. Default is None, in which case the function finds the hubs.
        hub_method (func) (opt): If the hubs are to be calculated, then this parameter specifies the method used to find hubs. Default is identify_hubs, which uses the Z score.
        hub_hub_path_nodes (list) (opt): A list of nodes on the paths between hubs. If None, this is calculated internally.
        occurrences (dict) (opt): A dictionary of hub path nodes and their number of occurrences. If None, this is calculated internally.

    Returns:
        bcs_list (list): A list of betweenness centralities belonging to nodes found on the paths between hubs.
    """

    # Find the betweenness centrality distribution of the network.
    if not bcs:
        bcs = find_betweenness_centralities(G)

    # Find the hub-hub path nodes and their number of occurrences.
    if not hub_hub_path_nodes:

        # Find the degree distribution of the network.
        if not degrees:
            degrees = find_degree_distribution(G)

        # Find the hubs of the network
        if not hubs:
            hubs = hub_method(G, degrees=None)

        # Find the hub path nodes and their occurrences
        hub_hub_path_nodes = find_hub_hub_path_nodes(G, hubs=hubs, hub_method=hub_method, degrees=None)

    # Find lists of betweenness centralities for the hub path nodes
    bcs_list = [bcs[i] for i in hub_hub_path_nodes]

    # Return the betweenness centralities
    return bcs_list


def hub_hub_path_betweenness_by_occurrence(G, bcs=None, degrees=None, hubs=None, hub_method=identify_hubs,
                                           hub_hub_path_nodes=None, occurrences=None):
    """
    Analyses the betweenness of nodes on the paths between hubs with the number of their occurrences.
    Returns a list of the betweenness centralities of these nodes, and the Pearson correlation coefficient between the number of occurrences on such a path and their centrality.

    Args:
        G (igraph.Graph): The network to be analysed.
        bcs (list) (opt): A list of betweenness centralities in the network. If already calculated, this parameter can be passed to prevent duplication. Default is None, in which case the function calculates the distribution internally.
        degrees (list) (opt): A list of degrees in the network. If already calculated, this parameter can be passed to prevent duplication. Default is None, in which case the function calculates the distribution internally.
        hubs (list) (opt): A list of hubs in the network, If already calculated, this parameter can be passed to prevent duplication. Default is None, in which case the function finds the hubs.
        hub_method (func) (opt): If the hubs are to be calculated, then this parameter specifies the method used to find hubs. Default is identify_hubs, which uses the Z score.
        hub_hub_path_nodes (list) (opt): A list of nodes on the paths between hubs. If None, this is calculated internally.
        occurrences (dict) (opt): A dictionary of hub path nodes and their number of occurrences. If None, this is calculated internally.

    Returns:
        bcs_list (list): A list of betweenness centralities belonging to nodes found on the paths between hubs.
    """

    # Find the betweenness centrality distribution of the network.
    if not bcs:
        bcs = find_betweenness_centralities(G)

    # Find the hub-hub path nodes and their number of occurrences.
    if not hub_hub_path_nodes:

        # Find the degree distribution of the network.
        if not degrees:
            degrees = find_degree_distribution(G)

        # Find the hubs of the network
        if not hubs:
            hubs = hub_method(G, degrees=None)

        # Find the hub path nodes and their occurrences
        hub_hub_path_nodes, occurrences = find_hub_hub_path_node_occurrences(G, hubs=hubs, hub_method=hub_method,
                                                                             degrees=degrees)

    # Find lists of the occurrences and betweenness centralities
    bcs_list = [bcs[i] for i in hub_hub_path_nodes]
    occ_list = [occurrences[i] for i in hub_hub_path_nodes]

    # Convert the occurrences and the betweenness centralities to a Pandas Series
    bcs_series = pd.Series(bcs_list)
    occ_series = pd.Series(occ_list)

    # Return the correlation coefficient for the variables.
    return bcs_list, bcs_series.corr(occ_series)


def calculate_r1(G, degrees=None):
    # Find the degree distribution
    if not degrees:
        degrees = find_degree_distribution(G)

    # Initialise an empty list to store the reach of nodes.
    r1 = []

    # Find the degree of each node.
    for node in G.vs():
        d = G.degree(node)

        # The reach of node v is d(v)+1
        r1.append(d + 1)

    # Return the r1 distribution
    return r1