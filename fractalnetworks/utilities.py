""" This module contains utility functions used throughout the fractal network toolkit."""

# Network analysis modules
from igraph import Graph
import igraph
import networkx as nx

# Mathematics modules
import numpy as np
import random
import matplotlib.pyplot as plt
import math

# Utility modules
from scipy.io import mmread
import csv

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
        x (float): The value of x.
        y (float): The logarithm base.

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


def swap_edges_at_random(G, number_edges_to_swap):
    """
    Swaps edge of a given network at random, to find an uncorrelated network with the same degree distribution.
    Removes two randomly chosen edges (a,b) and (c,d) and replaces them with edges (a,d) and (c,b).

    Args:
        G (igraph.Graph): The original network.
        number_edges_to_swap (int): The number of edges to swap.

    Returns:
        H (igraph.Graph): The new, uncorrelated network with the same degree distribution as G.
    """
    # Iterate number_edges_to_swap times.
    for i in range(number_edges_to_swap):

        # Create a copy of the graph
        H = G.copy()

        # There are certain bad swaps we could make, e.g. swapping (a,b), (c,a) would create a self loop and an edge (c,b) might already exist.
        # Set this flag to False until we find a good swap.
        good_swap = False

        # Iterate until we find a good swap.
        while not good_swap:
            # Choose two edges at random from the network
            edge1, edge2 = random.sample(list(G.es()), 2)

            # Find the endpoints of the edges.
            a = edge1.tuple[0]
            b = edge1.tuple[1]
            c = edge2.tuple[0]
            d = edge2.tuple[1]

            # Check if (a,d) or (c,b) are self loops.
            if a != d and b != c:
                # Check if (a,d) or (c,b) already exists.
                if a not in G.neighbors(d) and b not in G.neighbors(c):
                    # If neither of these are True, then we have found a good swap.
                    good_swap = True

        # Delete the old edges and add the new ones.
        H.add_edges([(a, d), (c, b)])
        H.delete_edges([(a, b), (c, d)])

    # Once number_edges_to_swap edges have been swapped, return the new network
    return H


def plot_scatter_graph(fractal_attribute, non_fractal_attribute, fractal_Ns, non_fractal_Ns, y_label, save_path=None,
                       plot=True):
    """
    Plots a comparison of the properties of fractal and non-fractal networks on a scatter graph.

    Args:
        fractal_attributes (list): A list of an attribute of fractal networks.
        non_fractal_attributes (list): A list of an attribute of non-fractal networks.
        fractal_Ns (list): A list of fractal network orders.
        non_fractal_Ns (list): A list of non-fractal network orders.
        y_label (str): Label for the y-axis, i.e. the attribute being plotted.
        save_path (str) (opt): The file path to save the figure to, if given. Default is None.
        plot (Bool) (opt): If True, display the network. Default is True.
    """
    # Find the median of both sets
    fractal_median = np.median(fractal_attribute)
    non_fractal_median = np.median(non_fractal_attribute)

    # The x axis ranges from the smallest network order N to the largest.
    x = np.linspace(min(min(fractal_Ns), min(non_fractal_Ns)), max(max(fractal_Ns), max(non_fractal_Ns)), 1001)

    # Plot the fractal attribute
    plt.scatter(fractal_Ns, fractal_attribute, marker="o", facecolors='none', edgecolors='navy', label="Fractal")
    # Plot the non-fractal attribute
    plt.scatter(non_fractal_Ns, non_fractal_attribute, marker="^", facecolors='none', edgecolors='crimson',
                label="Non-fractal")

    # Plot fractal median
    plt.plot(x, [fractal_median] * len(x), ':', color="navy")
    # Plot non-fractal median
    plt.plot(x, [non_fractal_median] * len(x), ':', color="crimson")

    # Label the axes
    plt.xlabel("$|G|$")
    plt.ylabel(y_label)

    # Add a legend
    plt.legend()

    # If a save path is given, then save the file in that location.
    if save_path:
        plt.savefig(save_path)

    # If plot is True display the graph
    if plot:
        plt.show()

    # Close the figure
    plt.close()


def plot_scatter_graph_random_networks(fractal_attribute, random_attribute, fractal_Ns, random_Ns, y_label,
                                       save_path=None, plot=True):
    """
    Plots a comparison of the properties of fractal networks and their random counterparts on a scatter graph.

    Args:
        fractal_attributes (list): A list of an attribute of fractal networks.
        random_attributes (list): A list of an attribute of the random networks.
        fractal_Ns (list): A list of fractal network orders.
        random_Ns (list): A list of random network orders.
        y_label (str): Label for the y-axis, i.e. the attribute being plotted.
        save_path (str) (opt): The file path to save the figure to, if given. Default is None.
        plot (Bool) (opt): If True, display the network. Default is True.
    """
    # Find the median of both sets
    fractal_median = np.median(fractal_attribute)
    random_median = np.median(random_attribute)

    # The x axis ranges from the smallest network order N to the largest.
    x = np.linspace(min(min(fractal_Ns, random_Ns)), max(max(fractal_Ns, random_Ns)), 1001)

    # Plot the fractal attribute
    plt.scatter(fractal_Ns, fractal_attribute, marker="o", facecolors='none', edgecolors='navy', label="Fractal")
    # Plot the non-fractal attribute
    plt.scatter(random_Ns, random_attribute, marker="x", facecolors='seagreen', label="Random")

    # Plot fractal median
    plt.plot(x, [fractal_median] * len(x), ':', color="navy")
    # Plot non-fractal median
    plt.plot(x, [random_median] * len(x), ':', color="seagreen")

    # Label the axes
    plt.xlabel("$|G|$")
    plt.ylabel(y_label)

    # Add a legend
    plt.legend()

    # If a save path is given, then save the file in that location.
    if save_path:
        plt.savefig(save_path)

    # If plot is True display the graph
    if plot:
        plt.show()

    # Close the figure
    plt.close()


def clean_lists_of_NaNs(list1, list2):
    """
    Cleans lists of NaN values before plotting.

    Args:
        list1 (list): The list to be plotted on the x-axis.
        list2 (list): The list to be plotted on the y-axis, to be checked for NaN values.

    Returns:
        list1_clean (list): A clean version of list1.
        list2_clean (list): A clean version of list2.
    """

    # Find a list of indexes where the value in the second list is NaN
    NaN_indexes = [index for (index, value) in enumerate(list2) if math.isnan(value)]

    # Only take the elements of lists not equal to NaN
    list1_clean = [list1[i] for i in range(len(list1)) if i not in NaN_indexes]
    list2_clean = [list2[i] for i in range(len(list2)) if i not in NaN_indexes]

    # Return the cleaned lists.
    return list1_clean, list2_clean


def display_network(G, save_path=None, plot=True):
    """
    Displays a given network.

    Args:
        G (igraph.Graph): The network to be displayed.
        save_path (str) (opt): The filepath to save the figure to, if given. Default is None.
        plot (Bool) (opt): If True, the network is displayed inline. Default is True.
    """
    # Plot the network
    fig, ax = plt.subplots()
    igraph.plot(G,
                vertex_size=10,
                vertex_color='black',
                edge_width=1,
                edge_color='black',
                layout="kamada_kawai",
                target=ax)
    # If a save path is given, then save the file in that location.
    if save_path:
        plt.savefig(save_path)

    # If plot is True display the graph
    if plot:
        plt.show()

    # Close the figure
    plt.close()


def find_best_exp_fit(x, y, A_min=0, A_max=10000, c_min=0, c_max=12.5, linspace_N=100):
    """
    Finds the best exponential fit of the form y = Ae^{-cx} according to the sum of squares deviation.

    Args:
        x (list): The values of x in the distribution.
        y (list): The values of y in the distribution.
        A_min (int) (opt): The minimum value of A to be tested. Can be adjusted to find more accurate results. Default is 0.
        A_max (int) (opt): The maximum value of A to be tested. Can be adjusted to find more accurate results. Default is 100.
        c_min (int) (opt): The minimum value of c to be tested. Can be adjusted to find more accurate results. Default is 0.
        c_max (int) (opt): The maximum value of c to be tested. Can be adjusted to find more accurate results. Default is 2.
        linspace_N (int) (opt): The number of values of A and c to be checked in the respective ranges. Default is 100.

    Returns:
        best_fit (tuple): The coefficients A and c from the best exponential approximation.
        best_score (float): The sum of squares regression of this approximation.
    """
    # Initialise empty variables for the best fit (i.e. best A and c) and the best SSR score.
    best_fit = (None, None)
    best_score = None

    # Iterate through linspace_N values of A in the range [A_min, A_max].
    for A in np.linspace(A_min, A_max, linspace_N):
        # Iterate through linspace_N values of c in the range [c_min, c_max].
        for c in np.linspace(c_min, c_max, linspace_N):

            # Find the values of y according to the exponential model with parameters A and c.
            est_y = [A * math.e ** (-c * i) for i in x]
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


def find_best_fractal_fit(x, y, A_min=0, A_max=10000, c_min=0, c_max=12.5, linspace_N=100):
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


def find_best_linear_fit(x, y, A_min=0, A_max=100, c_min=0, c_max=10, linspace_N=101):
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
    for A in np.linspace(A_min, A_max, linspace_N):
        # Iterate through linspace_N values of c in the range [c_min, c_max].
        for c in np.linspace(c_min, c_max, linspace_N):

            # Find the values of y according to the power law fractal model with parameters A and c.
            est_y = [-c * i + (A) for i in x]
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


def find_best_fit_iteratively(x, y, method, linspace_N=100, A_min=0, A_max=10000, c_min=0, c_max=12.5, iter_num=4):
    """
    Finds the best fit according to a given model by iteratively reducing the range of values checked for A and c.

    Args:
        x (list): The values of x in the distribution.
        y (list): The values of y in the distribution.
        method (func): A function which finds the best fit to the given distribution according to a given model.
        linspace_N (int) (opt): The number of values of A and c to be checked in the respective ranges. Default is 100.
        A_min (int) (opt): The minimum value of A to be tested. Can be adjusted to find more accurate results. Default is 0.
        A_max (int) (opt): The maximum value of A to be tested. Can be adjusted to find more accurate results. Default is 10000.
        c_min (int) (opt): The minimum value of c to be tested. Can be adjusted to find more accurate results. Default is 0.
        c_max (int) (opt): The maximum value of c to be tested. Can be adjusted to find more accurate results. Default is 12.5.
        iter_num (int) (opt): The number of iterations to complete.

    Returns:
        best_fit (tuple): The coefficients A and c from the best approximation.
        best_score (float): The sum of squares regression of this approximation.
    """

    # Find the current range of the intervals for A and c.
    A_diff = A_max - A_min
    c_diff = c_max - c_min

    # Iterate as many times as dictated by the iter_num variable.
    for i in range(iter_num):
        # Find the best fit in this range according to the intervals given.
        best_fit, best_score = method(x, y, A_min=A_min, A_max=A_max, c_min=c_min, c_max=c_max, linspace_N=linspace_N)

        # Reduce the interval range for A by a factor of 10 and for c by a factor of 5.
        A_diff = A_diff / 10
        c_diff = c_diff / 5

        # Extract the values of A and c from the current best fit.
        best_A_approximation = best_fit[0]
        best_c_approximation = best_fit[1]

        # Adjust the interval [A_min, A_max] to have a range of A_diff about the current best estimate for A.
        A_min = max(0, best_A_approximation - (A_diff / 2))
        A_max = best_A_approximation + (A_diff / 2)

        # Adjust the interval [c_min, c_max] to have a range of c_diff about the current best estimate for c.
        c_min = max(0, best_c_approximation - (c_diff / 2))
        c_max = best_c_approximation + (c_diff / 2)

    # Once iter_num iterations are complete, return the best fit and best score.
    return best_fit, best_score


def plot_lB_NB(lB, NB):
    """
    Plots the distribution of the optimal number of boxes NB against the diameter of the boxes lB.

    Args:
        lB (list): List of the values of lB.
        NB (list): List of the corresponding values of NB.

    Returns:
        None
    """
    # Plots distribution using matplotlib.
    plt.plot(lB, NB, color='#C00000')
    plt.xlabel('$\ell_B$')
    plt.ylabel('$N_B$')
    plt.title('The optimal number of boxes $N_B$ against the diameter $\ell_B$.')
    plt.show()


def plot_loglog_lB_NB(lB, NB):
    """
    Plots the distribution of the optimal number of boxes NB against the diameter of the boxes lB on a log log scale.

    Args:
        lB (list): List of the values of lB.
        NB (list): List of the corresponding values of NB.

    Returns:
        None
    """
    # Plots distribution on log log scale using matplotlib.
    plt.loglog(lB, NB, color='#C00000')
    plt.xlabel('$\ell_B$')
    plt.ylabel('$N_B$')
    plt.title('The optimal number of boxes $N_B$ against the diameter $\ell_B$.')
    plt.show()


def plot_best_fit_comparison(lB, NB, exp_A, exp_c, exp_score, frac_A, frac_c, frac_score):
    """
    Plots a comparison of the best power law (fractal) and exponential (non-fractal) fit for a given distribution.

    Args:
        lB (list): List of the values of lB.
        NB (list): List of the corresponding values of NB.
        exp_A (float): The optimal value of A according to the exponential fit.
        exp_c (float): The optimal value of c according to the exponential fit.
        exp_score (float): The sum of squares regression for the exponential best fit.
        frac_A (float): The optimal value of A according to the power law fit.
        frac_c (float): The optimal value of c according to the power law fit.
        frac_score (float): The sum of squares regression for the power law best fit.
    """
    # The lists lB and NB need to be converted to numpy arrays.
    lB = np.array(lB)
    NB = np.array(NB)

    est_NB_exp = [exp_A * math.e ** (-exp_c * i) for i in lB]  # Find the exponential fit according to A and c given.
    est_NB_frac = [frac_A * i ** (-frac_c) for i in lB]  # Find the power law fit according to A and c given.

    # Initialise a plot.
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 3))

    # Plot the distribution against the power law fit.
    # Note we only use one set of labels for the legend as both plots use the same colours.
    axes[0].plot(lB, NB, color='#000066', label="Empirical Data")
    axes[0].plot(lB, est_NB_frac, color='#C00000', label="Best Fit")

    # Plot the distribution against the exponential fit.
    axes[1].plot(lB, NB, color='#000066')
    axes[1].plot(lB, est_NB_exp, color='#C00000')

    # fig.suptitle('Non-Fractal Network Model', fontsize=16) # Title the plot.

    # Find the maximum x and y values.
    max_lB = max(lB)
    max_NB = max(NB)

    # Label the axes and title the subplot.
    axes[0].set_xlabel('$\ell_B$')
    axes[0].set_title('Power-Law Relation')
    axes[0].set_ylabel('$N_B$')
    axes[0].text(0.7 * max_lB, 0.95 * max_NB,
                 r"$SSR \approx ${0}".format(frac_score.round(4)))  # Write the SSR score on the plot.

    # Label the axes and title the subplot.
    axes[1].set_xlabel('$\ell_B$')
    axes[1].set_title('Exponential Relation')
    axes[1].set_ylabel('$N_B$')
    axes[1].text(0.7 * max_lB, 0.95 * max_NB,
                 r"$SSR \approx ${0}".format(exp_score.round(4)))  # Write the SSR score on the plot.

    fig.legend(loc="upper left")  # Add the legend.
    fig.tight_layout()


def plot_best_fit_comparison_by_logarithm(lB, NB, exp_A, exp_c, exp_score, frac_A, frac_c, frac_score):
    """
    Plots a comparison of the best power law (fractal) and exponential (non-fractal) fit for a given distribution.

    Args:
        lB (list): List of the values of lB.
        NB (list): List of the corresponding values of NB.
        exp_A (float): The optimal value of A according to the exponential fit.
        exp_c (float): The optimal value of c according to the exponential fit.
        exp_score (float): The sum of squares regression for the exponential best fit.
        frac_A (float): The optimal value of A according to the power law fit.
        frac_c (float): The optimal value of c according to the power law fit.
        frac_score (float): The sum of squares regression for the power law best fit.
    """
    loglB = [math.log(l) for l in lB]
    logNB = [math.log(n) for n in NB]

    # The lists lB and NB need to be converted to numpy arrays.
    lB = np.array(lB)
    loglB = np.array(loglB)
    logNB = np.array(logNB)

    est_NB_exp = [-exp_c * i + exp_A for i in lB]  # Find the exponential fit according to A and c given.
    est_NB_frac = [-frac_c * i + frac_A for i in loglB]  # Find the power law fit according to A and c given.

    # Initialise a plot.
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 3))
    fig.tight_layout()

    # Plot the distribution against the power law fit.
    # Note we only use one set of labels for the legend as both plots use the same colours.
    axes[0].plot(lB, logNB, color='#000066', label="Empirical Data")
    axes[0].plot(lB, est_NB_frac, color='#C00000', label="Best Fit")

    # Plot the distribution against the exponential fit.
    axes[1].plot(loglB, logNB, color='#000066')
    axes[1].plot(loglB, est_NB_exp, color='#C00000')

    # fig.suptitle('Non-Fractal Network Model', fontsize=16) # Title the plot.

    # Find the maximum x and y values.
    max_lB = max(lB)
    max_NB = max(NB)

    # Label the axes and title the subplot.
    axes[0].set_xlabel('$\ell_B$')
    axes[0].set_title('Power-Law Relation')
    axes[0].set_ylabel('$N_B$')
    axes[0].text(0.7 * max_lB, 0.95 * max_NB,
                 r"$SSR \approx ${0}".format(frac_score.round(4)))  # Write the SSR score on the plot.

    # Label the axes and title the subplot.
    axes[1].set_xlabel('$\ell_B$')
    axes[1].set_title('Exponential Relation')
    axes[1].set_ylabel('$N_B$')
    axes[1].text(0.7 * max_lB, 0.95 * max_NB,
                 r"$SSR \approx ${0}".format(exp_score.round(4)))  # Write the SSR score on the plot.

    fig.legend(loc="upper left")  # Add the legend.
    fig.tight_layout()


def draw_box_covering(G, nodes_to_boxes):
    """
    Displays the network with nodes coloured according to the box covering.

    Args:
        G (networkx.Graph): The network to be analysed.
        nodes_to_boxes (dict): A dictionary with nodes as keys and their corresponding boxes as values.

    Returns:
        None
    """

    G = G.to_networkx()

    # Initialise an empty colour map for the box covering visualiation.
    colourmap = []

    # For each node, assign it the colour of its box ID.
    for node in G.nodes():
        colourmap.append(nodes_to_boxes[str(node)])

    # If draw is True then display the graph with the colours indicating the box the node belongs to.
    nx.draw_kamada_kawai(G, node_color=colourmap)
    plt.show()


def export_to_gephi(G, nodes_to_boxes, file_path):
    """
    Puts graphs in a format readable to Gephi including attributes for the boxes found under box coverings.

    Args:
        G (networkx.Graph): The network to be analysed.
        nodes_to_boxes (dict): A dictionary with nodes as keys and their corresponding boxes as values.
        file_path (str): The path for the gml file to be saved to.

    Returns:
        None
    """
    H = G.copy()  # Create a copy of the network.

    # Assign to each node an attribute according to its box given under the box covering.
    nx.set_node_attributes(H, nodes_to_boxes, 'boxes')

    # Write the graph including the box covering attributes to the file path.
    nx.write_gml(H, file_path)


def chi_squared_error(observed_y, expected_y):
    """
    Finds the value SSR for the sum of squares regression method using a true and model distribution.

    Args:
        y (list): The true or measured distribution.
        est_y (list): The model distribution to be compared.

    Returns:
        sum_of_squares (float): The sum of squares regression
    """
    chi = 0  # Initialise the sum as zero.
    # Iterate for each pair of values in the true/model distributions.
    for (observed_yi, expected_yi) in zip(observed_y, expected_y):
        if expected_yi > 0:
            # Add to the sum the square of the difference between the two distributions.
            chi += (((observed_yi - expected_yi) ** 2) / expected_yi)
    # Return the total sum of the squares.
    return chi


def read_lB_NB_from_csv(filepath):
    with open(filepath, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        count = 1
        for row in reader:
            if count == 1:
                lB = [float(i) for i in row]
            else:
                NB = [float(i) for i in row]
            count += 1

    return lB, NB


def preprocess_network(filepath, save=True):
    # Need to add function to normalise edge weights
    G = Graph.Load(filepath)
    G = G.simplify()
    components = G.connected_components(mode='weak')
    giant_component = G.induced_subgraph(components[0])
    if save:
        save_path = filepath.replace('.gml','_processed.gml')
        giant_component.write_gml(save_path)
    return giant_component