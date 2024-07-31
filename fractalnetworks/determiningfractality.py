"""This module contians the functions needed to determine if a given network is fractal or non-fractal"""

from .maximumexcludedmassburning import *
from .utilities import *
from .greedyalgorithm import *

from tqdm import tqdm
from sklearn.linear_model import LinearRegression


def calculate_lB_NB_dist(G, diam=None, normalise=False, lB_min=2, save_path=None, colouring_ordering_method=None):
    """
    Finds the distribution of the optimal number of boxes NB against the diameter of these boxes lB.

    Args:
        G (igraph.Graph): The network to be analysed.
        diam (:obj:`int`, optional): The diameter of the network G. The default is None, and note that if no diameter is given the algorithm will calculate it which is expensive for large networks.
        normalise (:obj:`bool`, optional): If True, normalises the number of boxes over the total number of nodes in the network. Default is False.
        lB_min (:obj:`int`, optional): The minimum value of box diameter used to cover the network. Default is 2.
        save_path (:obj:`str`, optional): The filepath to save the results of the lB-NB distribution to. If none given, results are not saved.
        colouring_ordering_method (:obj:`func`, optional): The method by which to order the nodes for the greedy colouring algorithm. If none given, nodes are ordered lexicographically.

    Returns:
        (list): List of values for the box diameters lB.
        (list): List of the corresponding optimal number of boxes NB.
    """

    # If no diameter is given for the network then it is calculated using networkX.
    if diam == None:
        diam = G.diameter()

    # If an ordering method is specified, order nodes according to that scheme.
    # Otherwise the nodes are handled in lexicographical order.
    if colouring_ordering_method:
        ordering = colouring_ordering_method(G)
    else:
        ordering = [i for i in range(G.vcount())]

    # Calculate NB for all lB with the given method and diameter.
    NB = []
    lB = [l for l in range(lB_min, diam + 2)]

    # Iterate through all the possible values of l
    for l, _ in zip(lB, tqdm(range(1, len(lB) + 1))):
        # If l is even, apply the greedy box covering algorithm.
        if l % 2 == 0:
            # Find the number of boxes.
            _, N = greedy_box_covering(G, l, node_order=ordering)
            # Take the minimum of this value and the previous calculated NB as the next NB
            if l > 2:
                previous_N = NB[-1]
                N = min(previous_N, N)
        # If l is odd, use the accelerated MEMB method.
        else:
            centres, _ = accelerated_MEMB(G, l)
            N = len(centres)
        # Add the new value of NB to the list
        NB.append(N)

    # If normalise is True, normalise each result over the number of nodes in the network.
    if normalise == True:
        N = G.vcount()
        NB = [x / N for x in NB]

    # If a save path is given, save the results to a csv file.
    if save_path:
        to_write = np.row_stack((lB, NB))
        np.savetxt(save_path, to_write, delimiter=",")

    return lB, NB

def is_fractal(results_filepath, plot=False, verbose=False):
    """
    Tests if a given lB-NB distribution is a power-law or exponential, thus determining if a network is fractal or non-fractal.

    Args:
        results_filepath (string): The filepath for the csv file storing the lB-NB distribution.
        plot (:obj:`bool`, optional): If True, a comparison of the relationship between lB and NB is plotted on a log-log scale and a log scale.
        verbose (:obj:`bool`, optional): If True, the results are displayed.
    Returns:
        bool: True if the network is fractal, False otherwise.
    """
    # Read the lB-NB distribution from the csv file.
    lB, NB = read_lB_NB_from_csv(results_filepath)

    # Find the logarithms of the box diameter lB and the number of boxes NB.
    loglB = [math.log(l) for l in lB]
    logNB = [math.log(n) for n in NB]

    # Convert the lists to arrays.
    x = np.array(loglB).reshape((-1, 1))
    y = np.array(logNB)

    # Fit a linear model
    model = LinearRegression()
    model.fit(x, y)
    # Find the regression score of the model
    frac_score = model.score(x, y)
    frac_A = model.intercept_
    frac_c = model.coef_[0]

    # Convert the lists to arrays.
    x = np.array(lB).reshape((-1, 1))
    y = np.array(logNB)

    # Fit a linear model
    model = LinearRegression()
    model.fit(x, y)
    # Find the regression score of the model
    exp_score = model.score(x, y)
    exp_A = model.intercept_
    exp_c = model.coef_[0]

    # Plot the exponential and power law relationship
    if plot:
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 3))
        axes[0].plot(loglB, logNB)
        axes[0].plot(loglB, [(frac_c) * l + frac_A for l in loglB])
        axes[1].plot(lB, logNB)
        axes[1].plot(lB, [(exp_c) * l + exp_A for l in lB])
        plt.show()
        plt.close()

    # If the fractal power-law fit is better than the exponential fit, then the network is fractal.
    if frac_score > exp_score:
        # If verbose is True, print the results.
        if verbose:
            print("This network is fractal.")
            print("Power law score: {0}.".format(frac_score))
            print("Exponential score: {0}.".format(exp_score))
        return True
    # If the exponential fit is better than the power-law then the network is non-fractal.
    else:
        # If verbose is True, print the results.
        if verbose:
            print("This network is fractal.")
            print("Power law score: {0}.".format(frac_score))
            print("Exponential score: {0}.".format(exp_score))
        return False