"""This module contains code to estimate the reach of nodes in the network"""

from .utilities import *
from .networkproperties import *

def calculate_r1(G, degrees=None):
    """
    Finds the distribution of reach for a network, where reach is number of nodes a node can reach in s=1 step.
    This is the same as the degree of each node, plus one.

    Args:
        G (igraph.Graph): The network to be analysed.
        degrees (list): If known, the degree distribution of the network.

    Returns:
        list: A list of the reach for each node with s=1.
    """

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


def calculate_r2(G):
    """
    Estimates the distribution of reach for a network, where reach is number of nodes a node can reach in s=2 step.

    Args:
        G (igraph.Graph): The network to be analysed.
        degrees (list): If known, the degree distribution of the network.

    Returns:
        list: A list of the reach for each node with s=2.
    """
    # Initialise an empty list to store the distribution
    r2 = []

    # Iterate through all the nodes in the network
    for node in G.vs():
        # Find the degree
        d = G.degree(node)
        # Find the mean neighbour degree
        k = G.knn(node)[0][0]
        # Find the clustering coefficient
        T = G.transitivity_undirected()

        # Estimate the reach and add it to the distribution
        est_reach = 1 + d + (k - 1) * d * (1 - T)
        r2.append(est_reach)

    # Return the complete distribution
    return r2


def check_r1(G, plot=True):
    """
    Check the estimated reach for s=1 against the true distribution.

    Args:
        G (igraph.Graph): The network to be analysed.
        plot (Bool): If True, the histograms are plotted.

    Returns:
        float: The chi-squared score of the estimated distribution
        float: The chi-squared score of the frequency distribution of the estimated reach.

    """
    # Find the estimated distribution
    est_dist = calculate_r1(G)
    # Initialise a list for the true distribution
    true_dist = []

    # Iterate through each node
    for node in G.vs():
        # Find the true reach and add it to the distribution
        true_r1 = G.neighborhood_size(vertices=node, order=1, mode='all', mindist=0)
        true_dist.append(true_r1)

    # Find the minimum and maximum values
    min_r1 = math.floor(min(min(est_dist), min(true_dist)))
    max_r1 = math.ceil(max(max(est_dist), max(true_dist)))
    # Find the width of bins for the histograms
    binwidth = math.ceil((max_r1 - min_r1) / 10)

    # Find the histogram distributions
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 2))
    est_hist = axes[0].hist(est_dist, bins=range(min_r1, max_r1 + binwidth, binwidth))
    true_hist = axes[1].hist(true_dist, bins=range(min_r1, max_r1 + binwidth, binwidth))

    # If plot is true, plot the histograms.
    if plot:
        plt.show()

    plt.close()

    # Return the chi squared errors
    return chi_squared_error(est_dist, true_dist), chi_squared_error(est_hist[0], true_hist[0])


def check_r2(G, plot=True):
    """
    Check the estimated reach for s=2 against the true distribution.

    Args:
        G (igraph.Graph): The network to be analysed.
        plot (Bool): If True, the histograms are plotted.

    Returns:
        float: The chi-squared score of the estimated distribution
        float: The chi-squared score of the frequency distribution of the estimated reach.

    """
    # Find the estimated distribution
    est_dist = calculate_r2(G)
    # Initialise a list for the true distribution
    true_dist = []

    # Iterate through each node
    for node in G.vs():
        # Find the true reach and add it to the distribution
        true_r2 = G.neighborhood_size(vertices=node, order=2, mode='all', mindist=0)
        true_dist.append(true_r2)

    # Find the minimum and maximum values
    min_r2 = math.floor(min(min(est_dist), min(true_dist)))
    max_r2 = math.ceil(max(max(est_dist), max(true_dist)))
    # Find the width of bins for the histograms
    binwidth = math.ceil((max_r2 - min_r2) / 10)

    # Find the histogram distributions
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 2))
    est_hist = axes[0].hist(est_dist, bins=range(min_r2, max_r2 + binwidth, binwidth))
    true_hist = axes[1].hist(true_dist, bins=range(min_r2, max_r2 + binwidth, binwidth))

    # If plot is true, plot the histograms.
    if plot:
        plt.show()

    plt.close()

    # Return the chi squared errors
    return chi_squared_error(est_dist, true_dist), chi_squared_error(est_hist[0], true_hist[0])
