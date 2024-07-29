"""This module contians the functions needed to determine if a given network is fractal or non-fractal"""

from .maximumexcludedmassburning import *
from .utilities import *
from .greedyalgorithm import *

import tqdm

def calculate_lB_NB_dist(G, diam=None, method=degree_based_MEMB, min_lB=3):
    """
    Finds the distribution of the optimal number of boxes NB against the diameter of these boxes lB.

    Args:
        G (igraph.Graph): The network to be analysed.
        diam (int) (opt): The diameter of the network G. The default is None, and note that if no diameter is given the algorithm will calculate it which is expensive for large networks.
        method (func): The method used to calculate the box covering. Default is the degree based amended MEMB algorithm (see [2]).
        min_lB (int): The first value of lB to calculate NB from. Default is 3.

    Returns:
        lB (list): List of values for the box diameters lB.
        NB (list): List of the corresponding optimal number of boxes.
    """

    # If no diameter is given for the network then it is calculated using networkX.
    if diam == None:
        diam = G.diameter()

    # The MEMB algorithm only works for odd numbers (see MEMB function docstrings or [2] for explanation).
    # Therefore, find the next biggest odd number.
    nearest_odd = int(np.ceil(diam) // 2 * 2 + 1)

    # Take all possible (odd) values of lB from 1 to the diameter of the network.
    lB = [i for i in range(min_lB, nearest_odd + 2, 2)]

    # Initialise an empty list for the values of NB.
    NB = []

    N = G.vcount()

    # Iterate through all possible values of lB.
    for l in lB:
        centres = method(G, l) # Find the list of centres using the given method.
        NB.append(len(centres)) # Add the number of boxes (length of the list of centres) to the list NB.

    # Return the complete list of lB and NB values.
    return lB, NB


def is_fractal(G, diam=None, plot=False, verbose=False, normalise=False, lB_min=2, save_path=None,
               colouring_ordering_method=None, step=1):
    """
    Determines whether a network is fractal or not depending on the sum of squares regression score for the fractal and exponential fits.

    Args:
        G (igraph.Graph): The network to be analysed.
        diam (int): The diameter of the graph. If None, calculates the diameter. Default is None.
        plot (Bool): If True, plot a comparison of the best fits. Default is False.
        verbose (Bool): If True, print statements with the fractal and non fractal values. Default is False.

    Returns:
        lB (list): A list of box diameters for the covering.
        NB (list): A list of the number of boxes needed to cover the network.
        (Bool): True if the network is determined to be fractal, False otherwise.
    """

    if diam == None:
        diam = G.diameter()

    if colouring_ordering_method:
        ordering = colouring_ordering_method(G)
    else:
        ordering = [i for i in range(G.vcount())]

    # Calculate NB for all lB with the given method and diameter.
    NB = []
    lB = [l for l in range(lB_min, diam + 2)]

    for l, _ in zip(lB, tqdm(range(1, len(lB) + 1, step))):
        if l % 2 == 0:
            _, N = greedy_box_covering(G, l, node_order=ordering)
            if l > 2:
                previous_N = NB[-1]
                N = min(previous_N, N)
        else:
            centres, _ = accelerated_MEMB(G, l)
            N = len(centres)
        NB.append(N)

    if save_path:
        to_write = np.row_stack((lB, NB))
        np.savetxt(save_path, to_write, delimiter=",")

    if normalise == True:
        N = G.vcount()
        NB = [x / N for x in NB]

    # Find the best fractal fit.
    (frac_A, frac_c), frac_score = find_best_fit_iteratively(lB, NB, find_best_fractal_fit)

    # Find the best exponential fit.
    (exp_A, exp_c), exp_score = find_best_fit_iteratively(lB, NB, find_best_exp_fit)

    print(frac_score / exp_score)
    print(exp_score / frac_score)
    print("Power Law Parameters", frac_A, frac_c)
    print("Exponential Parameters", exp_A, exp_c)

    # If verbose, print a statement with both scores.
    if verbose:
        print("The SSR score for the fractal model is {0} and for the non-fractal model is {1}.".format(frac_score,
                                                                                                        exp_score))

    # If the fractal score is less than the exponential score, then the network is fractal.
    if frac_score < exp_score:
        # If verbose, print such a statement.
        if verbose:
            print("This network is fractal.")
        # If plot is True then plot a comparison of the best fits (see Section 4).
        if plot:
            plot_best_fit_comparison(lB, NB, exp_A, exp_c, exp_score, frac_A, frac_c, frac_score)
        # Return True if the network is fractal.
        return lB, NB, True
    # If the fractal score is greater than the exponential score, then the network is non-fractal.
    else:
        # If verbose, print such a statement.
        if verbose:
            print("This network is non-fractal.")
        # If plot is True then plot a comparison of the best fits (see Section 4).
        if plot:
            plot_best_fit_comparison(lB, NB, exp_A, exp_c, exp_score, frac_A, frac_c, frac_score)
        # Return False if the network is fractal.
        return lB, NB, False


def is_fractal_by_logarithm(G, diam=None, plot=False, verbose=False, normalise=False, lB_min=2, save_path=None,
                            colouring_ordering_method=None):
    """
    Determines whether a network is fractal or not depending on the sum of squares regression score for the fractal and exponential fits.

    Args:
        G (igraph.Graph): The network to be analysed.
        diam (int): The diameter of the graph. If None, calculates the diameter. Default is None.
        plot (Bool): If True, plot a comparison of the best fits. Default is False.
        verbose (Bool): If True, print statements with the fractal and non fractal values. Default is False.

    Returns:
        lB (list): A list of box diameters for the covering.
        NB (list): A list of the number of boxes needed to cover the network.
        (Bool): True if the network is determined to be fractal, False otherwise.
    """

    if diam == None:
        diam = G.diameter()

    if colouring_ordering_method:
        ordering = colouring_ordering_method(G)
    else:
        ordering = [i for i in range(G.vcount())]

    # Calculate NB for all lB with the given method and diameter.
    NB = []
    lB = [l for l in range(lB_min, diam + 2)]

    for l, _ in zip(lB, tqdm(range(1, len(lB) + 1))):
        if l % 2 == 0:
            _, N = greedy_box_covering(G, l, node_order=ordering)
            if l > 2:
                previous_N = NB[-1]
                N = min(previous_N, N)
        else:
            centres, _ = accelerated_MEMB(G, l)
            N = len(centres)
        NB.append(N)

    if save_path:
        to_write = np.row_stack((lB, NB))
        np.savetxt(save_path, to_write, delimiter=",")

    if normalise == True:
        N = G.vcount()
        NB = [x / N for x in NB]

    loglB = [math.log(l) for l in lB]
    logNB = [math.log(n) for n in NB]

    # Find the best fractal fit.
    (frac_A, frac_c), frac_score = find_best_fit_iteratively(loglB, logNB, find_best_linear_fit)

    # Find the best exponential fit.
    (exp_A, exp_c), exp_score = find_best_fit_iteratively(lB, logNB, find_best_linear_fit)

    print(frac_score / exp_score)
    print(exp_score / frac_score)
    print("Power Law Parameters", frac_A, frac_c)
    print("Exponential Parameters", exp_A, exp_c)

    # If verbose, print a statement with both scores.
    if verbose:
        print("The SSR score for the fractal model is {0} and for the non-fractal model is {1}.".format(frac_score,
                                                                                                        exp_score))

    # If the fractal score is less than the exponential score, then the network is fractal.
    if frac_score < exp_score:
        # If verbose, print such a statement.
        if verbose:
            print("This network is fractal.")
        # If plot is True then plot a comparison of the best fits (see Section 4).
        if plot:
            plot_best_fit_comparison_by_logarithm(lB, NB, exp_A, exp_c, exp_score, frac_A, frac_c, frac_score)
        # Return True if the network is fractal.
        return lB, NB, True
    # If the fractal score is greater than the exponential score, then the network is non-fractal.
    else:
        # If verbose, print such a statement.
        if verbose:
            print("This network is non-fractal.")
        # If plot is True then plot a comparison of the best fits (see Section 4).
        if plot:
            plot_best_fit_comparison_by_logarithm(lB, NB, exp_A, exp_c, exp_score, frac_A, frac_c, frac_score)
        # Return False if the network is fractal.
        return lB, NB, False