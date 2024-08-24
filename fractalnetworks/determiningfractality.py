"""This module contains the functions needed to determine if a given network is fractal or non-fractal"""

from .maximumexcludedmassburning import *
from .utilities import *
from .greedyalgorithm import *

from tqdm import tqdm
from sklearn.linear_model import LinearRegression


# Calculating lB

def calculate_lB_NB_dist(G, diam=None, normalise=False, lB_min=2, save_path=None, colouring_ordering_method=None):
    """
    Finds the distribution of the optimal number of boxes NB against the diameter of these boxes lB.

    Args:
        G (igraph.Graph)                                      : The network to be analysed.
        diam (:obj:`int`, optional)                           : The diameter of the network G.
                                                                If none given, the diameter is calculated.
                                                                Default is None.
        normalise (:obj:`bool`, optional)                     : If True, normalises the number of boxes over the total
                                                                    number of nodes in the network.
                                                                Default is False.
        lB_min (:obj:`int`, optional)                         : The minimum value of lB used to cover the network.
                                                                Default is 2.
        save_path (:obj:`str`, optional)                      : The filepath to save the results to.
                                                                If none given, results are not saved.
                                                                Default is None.
        colouring_ordering_method (:obj:`function`, optional) : The method by which to order the nodes for the greedy
                                                                    colouring algorithm.
                                                                If none given, nodes are ordered lexicographically.
                                                                Default is None.

    Returns:
        (list): List of values for the box diameters lB.
        (list): List of the corresponding optimal number of boxes NB.
    """

    # If no diameter is given for the network then it is calculated using networkX.
    if not diam:
        diam = G.diameter()

    # If an ordering method is specified, order nodes according to that scheme.
    # Otherwise, the nodes are handled in lexicographical order.
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
    if normalise:
        N = G.vcount()
        NB = [x / N for x in NB]

    # If a save path is given, save the results to a csv file.
    if save_path:
        to_write = np.row_stack((lB, NB))
        np.savetxt(save_path, to_write, delimiter=",")

    return lB, NB


def is_fractal(results_filepath, plot=False, verbose=False, save_path=None, p_list=None):
    """
    Tests if a given lB-NB distribution is a power-law or exponential, thus determining if a network is fractal or not.

    Args:
        results_filepath (str)           : The filepath for the csv file storing the lB-NB distribution.
        plot (:obj:`bool`, optional)     : If True, a comparison of the relationship between lB and NB is plotted.
        verbose (:obj:`bool`, optional)  : If True, the results are printed.
        save_path (:obj:`str`, optional) : The figure generated is saved to the file path, if given. Default is None.
        p_list (:obj:`list`, optional)   : A list of percentages p.
                                           The fractal dimension is given by the gradient of the best fit over p% of the
                                                distribution.
                                           Default is None.
    Returns:
        bool : True if the network is fractal, False otherwise.
    """
    # Initialise p_list as [1.0] if none given.
    if not p_list:
        p_list = [1.0]

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
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 3))
        axes[0].plot(loglB, logNB, color='navy')
        axes[0].plot(loglB, [frac_c * l + frac_A for l in loglB], ':', color='crimson')
        axes[0].set_xlabel('$\ln \ell_B$')
        axes[0].set_ylabel('$\ln N_B$')
        axes[0].set_title('Power-Law Relationship')
        axes[1].plot(lB, logNB, color='navy')
        axes[1].plot(lB, [exp_c * l + exp_A for l in lB], ':', color='crimson')
        axes[1].set_xlabel('$\ell_B$')
        axes[1].set_ylabel('$\ln N_B$')
        axes[1].set_title('Exponential Relationship')
        # If a save path is given, save the png file.
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        plt.show()

    plt.close()

    # If the fractal power-law fit is better than the exponential fit, then the network is fractal.
    if frac_score > exp_score:
        # If verbose is True, print the results.
        if verbose:
            A, dB, _ = find_fractal_dimension(loglB, logNB, p_list=p_list)
            # Plot the fractal dimension
            if plot:
                plt.plot(loglB, logNB, color='navy')
                plt.plot(loglB, [A + dB * l for l in loglB], ':', color='crimson')
                plt.xlabel('$\ln \ell_B$')
                plt.ylabel('$\ln N_B$')
                plt.show()
            plt.close()
            print("This network is fractal with box dimension {:.4f}.".format(-1 * dB))
            print("Power law score: {:.4f}.".format(frac_score))
            print("Exponential score: {:.4f}.".format(exp_score))
        return True
    # If the exponential fit is better than the power-law then the network is non-fractal.
    else:
        # If verbose is True, print the results.
        if verbose:
            print("This network is non-fractal.")
            print("Power law score: {:.4f}.".format(frac_score))
            print("Exponential score: {:.4f}.".format(exp_score))
        return False


def find_fractal_dimension(loglB, logNB, p_list=None):
    """
    Finds the fractal dimension of a fractal network given the log(lB)-log(NB) distribution.
    Fits a linear distribution of the form log(NB) = clog(lB) + A, where c is the fractal dimension dB.

    Args:
        loglB (list)                     : A list of values of log(lB)
        logNB (list)                     : A list of values of log(NB)
        p_list (:obj:`list`, optional)   : A list of percentages p.
                                           The fractal dimension is given by the gradient of the best fit over p% of the
                                                distribution.
                                           Default is None.
    Returns:
        float : The optimal value of the intercept A.
        float : The optimal value of the gradient c=dB, which is the fractal dimension.
        float : The coefficient of determination of the linear fit with parameters A and c.
    """
    # Initialise p_list as [1.0] if none given.
    if not p_list:
        p_list = [1.0]

    # Initialise the variables to store the best fit
    best_score = 0
    best_A = None
    best_c = None

    # Look at sections of the curve at least 70% of the full length.
    for p in p_list:
        # Find the best score over portions of that width
        A, c, score = find_best_range(loglB, logNB, percentage=p)
        # If this is better than the previous portion, then update the variables.
        if score > best_score:
            best_score = score
            best_A = A[0]
            best_c = c[0][0]

    return best_A, best_c, best_score


def find_best_range(x, y, percentage=1.0):
    """
    Finds the portion of the distribution which covers percentage% of the total distribution which best fits a linear
        distribution.

    Args:
        x (list)                              : A list of values of x
        y (list)                              : A list of values of y
        percentage (:obj:`float`, optional)   : The percentage of the distribution to test.
                                                Default is 1.0
    Returns:
        (tuple) : Tuple containing 3 floats.
                    The first is the optimal value of the intercept A.
                    The second is the optimal value of the gradient c=dB, which is the fractal dimension.
                    The third is the coefficient of determination of the linear fit with parameters A and c.
    """
    # Initialise variables as None.
    best_score = None
    best_A = None
    best_c = None

    # Find the width of a window which covers percentage% of the distribution.
    sample_width = int(len(x) * percentage)

    # Initialise a counter to track the window which is being searched.
    i = 0
    # Consider a moving window across the distribution.
    while i + sample_width <= len(x):
        # Find the values of x and y in the window.
        sublist_x = x[i:int(i + sample_width)]
        sublist_y = y[i:int(i + sample_width)]
        sublist_array_x = np.array(sublist_x).reshape((-1, 1))
        sublist_array_y = np.array(sublist_y).reshape((-1, 1))

        # Fit a linear model to the window.
        model = LinearRegression()
        model.fit(sublist_array_x, sublist_array_y)

        # Find the score of the new fit.
        score = model.score(sublist_array_x, sublist_array_y)

        # Ignore false fits.
        if model.coef_ != 0:
            # Update the best fit each time a better one is found.
            if not best_score:
                best_score = score
                best_A = model.intercept_
                best_c = model.coef_

            elif score > best_score:
                best_score = score
                best_A = model.intercept_
                best_c = model.coef_

        # Increment the counter
        i += 1
    # Return the best fits.
    return best_A, best_c, best_score
