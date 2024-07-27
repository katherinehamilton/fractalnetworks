""" This module contains utility functions used throughout the fractal network toolkit."""

# Network analysis modules
from igraph import Graph
import networkx as nx

# Mathematics modules
import numpy as np

# Utility modules
from scipy.io import mmread

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


def read_mtx_graph_format(filepath):
    """
    Reads graphs stored in the .mtx file format. Use with, for example, graphs from www.networkrepository.com.
    Note: Some files may need to be edited to make sure that scipy.io can read them. Files should have a header starting with %%MatrixMarket and a single line denoted the number of values in each column.

    Args:
        filepath (str): Filepath to .mtx file

    Returns:
        G (networkx.Graph): Network read from file.
    """
    # Read the file using the scipy.io file reader.
    mmf = mmread(filepath)
    # Generate a graph from this file.
    G = nx.from_scipy_sparse_array(mmf)
    # Return the graph.
    return G


def exact_log(x, y):
    """
    Finds the exact logarithm of x base y.

    Args:
        x (int): The value of x.
        y (int): The logarithm base.

    Returns:
        power (int): The floor of the power, where y^power = x.
        x==1 (Bool): True if x is an exact power of y, False otherwise.
    """
    power = 0
    while (x % y == 0):
        x = x / y
        power += 1
    return power, x == 1


def find_best_power_law_fit(x, y, A_min=0, A_max=10000, c_min=0, c_max=12.5, linspace_N=100):
    """
    Finds the best exponential fit according to the sum of squares deviation of the form y = Ax^{-c}

    Args:
        x (list): The values of x in the distribution.
        y (list): The values of y in the distribution.
        A_min (int) (opt): The minimum value of A to be tested. Can be adjusted to find more accurate results. Default is 0.
        A_max (int) (opt): The maximum value of A to be tested. Can be adjusted to find more accurate results. Default is 10000.
        c_min (int) (opt): The minimum value of c to be tested. Can be adjusted to find more accurate results. Default is 0.
        c_max (int) (opt): The maximum value of c to be tested. Can be adjusted to find more accurate results. Default is 12.5.
        linspace_N (int) (opt): The number of values of A and c to be checked in the respective ranges. Default is 100.

    Returns:
        best_fit (tuple): The coefficients A and c from the best power law approximation.
        best_score (float): The sum of squares regression of this approximation.
    """
    # Initialise empty variables for the best fit (i.e. best A and c) and the best SSR score.
    best_fit = (None, None)
    best_score = None

    # Iterate through linspace_N values of A in the range [A_min, A_max].
    for A in np.linspace(A_min, A_max, 100):
        # Iterate through linspace_N values of c in the range [c_min, c_max].
        for c in np.linspace(c_min, c_max, 100):

            # Find the values of y according to the power law fractal model with parameters A and c.
            est_y = [A * i ** (-c) for i in x]
            # Calculate the sum of squares regression.
            score = sum_of_squares_error(y, est_y)

            # If the best score is yet to be updated (i.e. this is the first iteration) then set the current A, c and SSR to the best fit values.
            if best_score == None:
                best_score = score
                best_fit = (A, c)
            # If the new SSR score is smaller than the current best, then update the best score and update the best fit to the current A and c.
            elif score < best_score:
                best_score = score
                best_fit = (A, c)

    # Once all values are tried return the best fit and the best score.
    return best_fit, best_score


def sum_of_squares_error(y, est_y):
    """
    Finds the value SSR for the sum of squares regression method using a true and model distribution.

    Args:
        y (list): The true or measured distribution.
        est_y (list): The model distribution to be compared.

    Returns:
        sum_of_squares (float): The sum of squares regression
    """
    sum_of_squares = 0  # Initialise the sum as zero.
    # Iterate for each pair of values in the true/model distributions.
    for (yi, est_yi) in zip(y, est_y):
        # Add to the sum the square of the difference between the two distributions.
        sum_of_squares += (est_yi - yi) ** 2
    # Return the total sum of the squares.
    return sum_of_squares