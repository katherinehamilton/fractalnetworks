"""This module analyses the properties of known network models."""

# Other function imports
from .modelgenerator import *
from .utilities import *

# Mathematics modules
import random
import math
import matplotlib.pyplot as plt

def SHM_no_of_edges(m, n):
    """
    Finds the number of edges in an n-th generation SHM network with parameter m.

    Args:
        m (int): The number of offspring added at each stage, as defined by the SHM model [2].
        n (int): The number of iterations to perform of the SHM generative process [2].

    Returns:
        (int): The number of edges in the network.
    """
    # A SHM network has (1+2m)^(n-1) edges.
    return (1 + 2 * m) ** (n - 1)


def SHM_no_of_nodes(m, n):
    """
    Finds the number of nodes in an n-th generation SHM network with parameter m.

    Args:
        m (int): The number of offspring added at each stage, as defined by the SHM model [2].
        n (int): The number of iterations to perform of the SHM generative process [2].

    Returns:
        (int): The number of nodes in the network.
    """
    # A SHM network has (1+2m)^(n-1)+1 nodes.
    return (1 + 2 * m) ** (n - 1) + 1


def check_SHM_no_of_edges(m, n):
    """
    Checks the validity of the number of edges of a SHM network calculating according to analytical arguments in [1]

    Args:
        m (int): The parameter m in the SHM network model [3]
         n(int): The parameter n in the SHM network model [3]

    Returns:
        (Bool): True if the analytical and empirical value agree, False otherwise.
    """
    # Generate an SHM network. The number of edges is independent of the probability p, so p is chosen randomly on the interval [0,1]
    G = generate_SHM_model(m, random.random(), n, save=False)

    eG = len(G.edges())  # Empirical number of edges
    est_eG = SHM_no_of_edges(m, n)  # Estimated number of edges

    # Check if the values are equal
    return eG == est_eG


def check_SHM_no_of_nodes(m, n):
    """
    Checks the validity of the number of nodes of a SHM network calculating according to analytical arguments in [1]

    Args:
        m (int): The parameter m in the SHM network model [3]
        n (int): The parameter n in the SHM network model [3]

    Returns:
        (Bool): True if the analytical and empirical value agree, False otherwise.
    """
    # Generate an SHM network. The number of nodes is independent of the probability p, so p is chosen randomly on the interval [0,1]
    G = generate_SHM_model(m, random.random(), n, save=False)

    N = len(G.nodes())  # Empirical number of nodes
    est_N = SHM_no_of_nodes(m, n)  # Estimated number of nodes

    # Check if the values are equal
    return N == est_N


def SHM_p0_degree_prob(m, d, n):
    """
    Calculates the probability P(d) for a given degree d in a SHM network with parameters m, n and p=0.

    Args:
        m (int): The parameter m in the SHM network model [3]
        d (int): The degree to find the probability density P(d) for.
        n (int): The parameter N in the SHM network model [3]

    Returns:
        (float): The probability P(d) from the degree distribution.
    """
    # The maximum degree in a network with parameters m, N is (1+m)^(N-1) [4].
    # Thus for any larger degree d, the probability of a node having degree d is 0.
    if d > (1 + m) ** (n - 1):
        return 0

    # All degrees in a SHM network with p=0 are powers of (1+m)
    power, is_power = exact_log(d, (1 + m))  # Check if d is a power of (1+m) and find the exponent.

    # If d is a power, then we can find P(d).
    if is_power:
        # Nodes added at stage k have degree d = (1+m)^power = (1+m)^(N-k), so k = N-power.
        k = n - power

        # In the first stage, 2 nodes are added.
        if k == 1:
            no_deg_d = 2

        # In every subsequent stage (1+2m)^(k-1) - (1+2m)^(k-2) nodes are added.
        else:
            no_deg_d = (1 + 2 * m) ** (k - 1) - (1 + 2 * m) ** (k - 2)

        # Find the total number of nodes in the network.
        total = SHM_no_of_nodes(m, n)

        # The probability that a node has degree d is the number of nodes with degree d divided by the total number of nodes.
        return no_deg_d / total

    # For all other degrees, return 0.
    else:
        return 0


def check_SHM_p0_degree_prob(m, d, n):
    """
    Checks the accuracy of the calculated degree probability, according to analytical arguments in [1].

    Args:
        m (int): The parameter m in the SHM network model [2]
        d (int): The degree to find the probability density P(d) for.
        n (int): The parameter N in the SHM network model [2]

    Returns:
        (Bool): True if the two values agree, and False if they disagree.
    """
    # Generate a SHM network with parameters m, n, p=0.
    G = generate_SHM_model(m, 0, n, save=False)

    # Find a list of the degrees in the network.
    degrees = [deg[1] for deg in G.degree()]

    # Count the number of nodes with degree d.
    no_deg_d = degrees.count(d)
    # The probability that a node has degree d is the number of nodes with degree d divided by the total number of nodes.
    prob = no_deg_d / len(degrees)

    # Find the calculated probability according to the analytical arguments.
    est_prob = SHM_p0_degree_prob(m, d, n)

    # Return True if the values agree, and False if not.
    return prob == est_prob


def SHM_p0_degree_dist(m, n):
    """
    Calculates the degree distribution of a SHM network with p=0

    Args:
        m (int): The parameter m in the SHM network model to test [2]
        n (int): The parameter N in the SHM network model [2]

    Returns:
        degrees (list): A list of the degrees in the network.
        degree_dist (list): A list of probabilities of a node having a given degree.
    """
    # Initialise empty lists for the degrees and the degree distribution
    degrees = []
    degree_dist = []

    # Iterate through each of the generative stages 1, 2, ..., N
    for k in range(n, 0, -1):
        # The degree of a node added at time k is (1+m)^(N-k)
        d = (1 + m) ** (n - k)
        # Add the degree d to the list of degrees.
        degrees.append(d)
        # Add the probability P(d) to the degree distribution.
        degree_dist.append(SHM_p0_degree_prob(m, d, n))

    # Return the list of degrees and the degree distribution.
    return degrees, degree_dist


def check_SHM_p0_degree_dist(m, n):
    """
    Checks the accuracy of the calculated degree distribution, according to analytical arguments in [4].

    Args:
        m (int): The parameter m in the SHM network model [2]
        n (int): The parameter n in the SHM network model [2]

    Returns:
        (Bool): True if the two distributions agree, and False if they disagree.
    """
    # Generate a SHM network with parameters m, N, p=0.
    G = generate_SHM_model(m, 0, n, save=False)
    # Find the degree distribution.
    degrees = [deg[1] for deg in G.degree()]

    # Find a sorted list of degrees without repetition.
    unique_degrees = list(set(degrees))
    unique_degrees.sort()

    # Initialise an empty list for the degree distribution.
    degree_dist = []

    # For each of the degrees, find the number of occurrences and divide that by the total number of nodes.
    for d in unique_degrees:
        no_deg_d = degrees.count(d)
        prob = no_deg_d / len(degrees)
        # Add the probability to the degree distribution
        degree_dist.append(prob)

    # Find the estimated degree distribution according to the analytical arguments.
    _, est_degree_dist = SHM_p0_degree_dist(m, n)

    # Return True if the distributions are the same, and False otherwise.
    return degree_dist == est_degree_dist


def SHM_p1_degree_prob(m, d, n):
    """
    Calculates the probability P(d) for a given degree d in a SHM network with parameters m, N and p=1.

    Args:
        m (int): The parameter m in the SHM network model [2]
        d (int): The degree to find the probability density P(d) for.
        n (int): The parameter n in the SHM network model [2]

    Returns:
        (float): The probability P(d) from the degree distribution.
    """
    # The maximum degree in a network with parameters m, N is m^(N-1) [4].
    # Thus for any larger degree d, the probability of a node having degree d is 0.
    if d > (m) ** (n - 1):
        return 0

    # All degrees in a SHM network are powers of m or 2 times a power of m.
    power, is_power = exact_log(d, m)  # Check if d is a power of m, and find the exponent.
    double_power, is_double_power = exact_log(d / 2, m)  # Check if 1/2 d is a power of m, and find the exponent.

    # The only case where d and 1/2 are both powers of m is when m = 2.
    # In this case, the calculation of the degree distribution is different.
    if is_power and is_double_power:
        # Find k, where k is the stage that the node is added.
        # The degree of the node is d=m^(N-k), so if d=m^power, then power = N-k
        k = n - power

        # If the node was added in the first stage, or if the node was added in the second stage by rewired, then k=1.
        if k == 1:
            # There are four such nodes, the 2 added in the first stage and the 2 rewired in the second stage.
            no_deg_d = 4

        # If the node was added in any other stage, then calculate the number of nodes with this degree.
        else:
            # There are 2(m-1)(1+2m)^(k-2) nodes added in the k-th stage which aren't rewired,
            #            and 2(1+2m)^(k-1) nodes added in the (k+1)-th stage which are rewired.
            no_deg_d = 2 * (m - 1) * (1 + 2 * m) ** (k - 2) + 2 * (1 + 2 * m) ** (k - 1)

            # Find the total number of nodes.
        total = SHM_no_of_nodes(m, n)
        # The probability of a given node having degree d is the number of nodes with that degree divided by the total number of nodes.
        return no_deg_d / total

    # The remaining code deals with all other cases when m != 2.

    # If d is a power of m, then d is the degree of a node added in the k-th stage which is not rewired,
    #    where k is such that d = m ^ (n-k)
    elif is_power:
        # Find k
        k = n - power

        # If the node was added in the first stage then k=1.
        if k == 1:
            # There are two nodes added in the first stage.
            no_deg_d = 2

        # Otherwise, the number of nodes added in the k-th stage which aren't rewired is 2(m-1)(1+2m)^(k-2).
        else:
            no_deg_d = 2 * (m - 1) * (1 + 2 * m) ** (k - 2)

        # Find the total number of nodes.
        total = SHM_no_of_nodes(m, n)
        # The probability of a given node having degree d is the number of nodes with that degree divided by the total number of nodes.
        return no_deg_d / total

    # If 1/2 d is a power of m, then d is the degree of a node added in the k-th stage which is rewired,
    #    where k is such that d = 2 m ^ (n-k)
    elif is_double_power:
        # Find k
        k = n - power

        # No nodes are rewired in the first stage, so we do not need to consider that case.
        no_deg_d = 2 * (1 + 2 * m) ** (k - 2)

        # Find the total number of nodes.
        total = SHM_no_of_nodes(m, n)
        # The probability of a given node having degree d is the number of nodes with that degree divided by the total number of nodes.
        return no_deg_d / total

    # If the degree d is a not a power of m, or 1/2 d is not a power of m, then no nodes will have degree d.
    # Thus the probability is 0.
    else:
        return 0


def check_SHM_p1_degree_prob(m, d, n):
    """
    Checks the accuracy of the calculated degree probability, according to analytical arguments in [4].

    Args:
        m (int): The parameter m in the SHM network model [2]
        d (int): The degree to find the probability density P(d) for.
        n (int): The parameter n in the SHM network model [2]

    Returns:
        (Bool): True if the two values agree, and False if they disagree.
    """
    # Generate a SHM network with parameters m, N, p=1.
    G = generate_SHM_model(m, 1, n, save=False)

    # Find a list of the degrees in the network.
    degrees = [deg[1] for deg in G.degree()]

    # Count the number of nodes with degree d.
    no_deg_d = degrees.count(d)
    # The probability that a node has degree d is the number of nodes with degree d divided by the total number of nodes.
    prob = no_deg_d / len(degrees)

    # Find the calculated probability according to the analytical arguments.
    est_prob = SHM_p1_degree_prob(m, d, n)

    # Return True if the values agree, and False if not.
    return prob == est_prob


def SHM_p1_degree_dist(m, n):
    """
    Calculates the degree distribution of a SHM network with p=0

    Args:
        m (int): The parameter m in the SHM network model to test [2]
        n (int): The parameter n in the SHM network model [2]

    Returns:
        degrees (list): A list of the degrees in the network.
        degree_dist (list): A list of probabilities of a node having a given degree, where degree_dist[i] is P(degrees[i])
    """
    # Initialise empty lists for the degrees and degree distribution.
    degrees = []
    degree_dist = []

    # The degree distribution behaves different for m=2, so we consider this case separately.
    if m == 2:
        # Iterate through each of the generative stages 1, 2, ..., N
        for k in range(n, 0, -1):
            # The degree of nodes generated at stage k is m^(N-k)
            # For m=2, nodes generates at stage k+1 which are rewired also have degree m^(N-k)
            d = (m) ** (n - k)

            # Add the degree d to the list of degrees, and the probability P(d) to the degree distribution.
            degrees.append(d)
            degree_dist.append(SHM_p1_degree_prob(m, d, n))

    # The following code deals with the case when m != 2.
    else:
        # Iterate through each of the generative stages 1, 2, ..., N
        for k in range(n, 0, -1):

            # The degree of nodes generated at stage k is m^(N-k)
            d = (m) ** (n - k)
            # Add the degree d to the list of degrees, and the probability P(d) to the degree distribution.
            degrees.append(d)
            degree_dist.append(SHM_p1_degree_prob(m, d, n))

            # For all stages except the first, there are also some nodes which are rewired.
            # These have degree 2m^(N-k)
            if not k == 1:
                d = 2 * (m) ** (n - k)

                # Add the degree d to the list of degrees, and the probability P(d) to the degree distribution.
                degrees.append(d)
                degree_dist.append(SHM_p1_degree_prob(m, d, n))

    # Return the list of degrees and the degree distribution.
    return degrees, degree_dist


def check_SHM_p1_degree_dist(m, n):
    """
    Checks the accuracy of the calculated degree distribution, according to analytical arguments in [4].

    Args:
        m (int): The parameter m in the SHM network model [2]
        n (int): The parameter n in the SHM network model [2]

    Returns:
        (Bool): True if the two distributions agree, and False if they disagree.
    """
    # Generate a SHM network with parameters m, N, p=1.
    G = generate_SHM_model(m, 1, n, save=False)
    # Find the degree distribution.
    degrees = [deg[1] for deg in G.degree()]

    # Find a sorted list of degrees without repetition.
    unique_degrees = list(set(degrees))
    unique_degrees.sort()

    # Initialise an empty list for the degree distribution.
    degree_dist = []

    # For each of the degrees, find the number of occurrences and divide that by the total number of nodes.
    for d in unique_degrees:
        no_deg_d = degrees.count(d)
        prob = no_deg_d / len(degrees)
        # Add the probability to the degree distribution
        degree_dist.append(prob)

    # Find the estimated degree distribution according to the analytical arguments.
    _, est_degree_dist = SHM_p1_degree_dist(m, n)

    # Return True if the distributions are the same, and False otherwise.
    return degree_dist == est_degree_dist


def check_SHM_p0_lambda(m_max, n, plot=False, verbose=False):
    """
    Verfies that the analytical result for lambda agrees with the empirical result.

    Args:
        m (int): The maximum value of parameter m in the SHM network model to test [2]
        n (int): The parameter n in the SHM network model [2]
        plot (Bool) (opt): If True, plots a comparison of the results. Default is False.
        verbose (Bool) (opt): If True, prints results to terminal. Default is False.

    Returns:
        empirical (list): A list of the values of lambda found by fitting a power law distribution to empirical data.
        analystical (list): A list of the values of lambda determined by the analytical argument in [4]
    """

    # Initialise empty lists to store the empirical and analytical results.
    empirical = []
    analytical = []
    power = []

    # Create a list of all possible values of m
    ms = [m for m in range(2, m_max + 1)]

    # Iterate over each possible m
    for m in ms:
        # Find the degree distribution
        degrees, degree_dist = SHM_p0_degree_dist(m, n)
        # Fit a power law curve to the distribution and find the parameters characterising the best fit
        best_fit, _ = find_best_power_law_fit(degrees, degree_dist, A_min=0.5, A_max=1.2, c_min=1, c_max=1.5,
                                              linspace_N=1001)
        # Calculate the estimated value of lambda according to analytical results.
        est_lambda = math.log(1 + 2 * m) / math.log(1 + m)

        # Add the results to their respective lists.
        empirical.append(best_fit[1])
        analytical.append(est_lambda)

    # If verbose is True print the results.
    if verbose:
        print("Analytical Result", analytical)
        print("Empirical Result", empirical)

    # If plot is True plot the results.
    if plot:
        plt.figure(figsize=(8, 4))
        plt.plot(ms, analytical, label="Analytical Result", color="navy")
        plt.plot(ms, empirical, label="Empirical Result", color="crimson")
        plt.suptitle("Degree Distribution of the Song-Havlin-Makse Network with $p=0$")
        plt.title("Comparison of Power Law Coefficient $\lambda$ found by Empirical and Analytical Methods",
                  fontsize=10)
        plt.xlabel("$m$")
        plt.ylabel("$\lambda$")
        plt.legend()
        plt.show()

    # Return the results from both methods as lists.
    return empirical, analytical


def HADG_no_of_edges(m, n):
    """
    Finds the number of edges in an n-th generation HADG network with parameter m.

    Args:
        m (int): The number of offspring added at each stage, as defined by the HADG model [2].
        n (int): The number of iterations to perform of the HADG generative process [2].

    Returns:
        (int): The number of edges in the network.
    """
    # A HADG network has (3+2m)^(n-1) edges.
    return (3 + 2 * m) ** (n - 1)


def HADG_no_of_nodes(m, n):
    """
    Finds the number of nodes in an n-th generation HADG network with parameter m.

    Args:
        m (int): The number of offspring added at each stage, as defined by the HADG model [2].
        n (int): The number of iterations to perform of the HADG generative process [2].

    Returns:
        N (int): The number of nodes in the network.
    """
    # Calculate the number of nodes
    N = 2

    for t in range(2, n + 1):
        N += 2 * m * ((2 * m + 3) ** (t - 2))

    return N


def check_HADG_no_of_edges(m, n):
    """
    Checks the validity of the number of edges of a HADG network calculating according to analytical arguments in [1]

    Args:
        m (int): The parameter m in the HADG network model [3]
        n(int): The parameter n in the HADG network model [3]

    Returns:
        (Bool): True if the analytical and empirical value agree, False otherwise.
    """
    # Generate an SHM network. The number of edges is independent of the parameters a, b and T, so they are chosen randomly on the interval [0,1]
    G = generate_HADG_model(m, random.random(), random.random(), random.random(), n, save=False)

    eG = len(G.edges())  # Empirical number of edges
    est_eG = HADG_no_of_edges(m, n)  # Estimated number of edges

    # Check if the values are equal
    return eG == est_eG


def check_HADG_no_of_nodes(m, n):
    """
    Checks the validity of the number of nodes of a HADG network calculating according to analytical arguments in [1]

    Args:
        m (int): The parameter m in the HADG network model [3]
        n(int): The parameter n in the HADG network model [3]

    Returns:
        (Bool): True if the analytical and empirical value agree, False otherwise.
    """
    # Generate an SHM network. The number of edges is independent of the parameters a, b and T, so they are chosen randomly on the interval [0,1]
    G = generate_HADG_model(m, random.random(), random.random(), random.random(), n, save=False)

    N = len(G.nodes())  # Empirical number of edges
    est_N = HADG_no_of_nodes(m, n)  # Estimated number of edges

    # Check if the values are equal
    return N == est_N


def uv_flower_no_of_edges(u, v, n):
    """
    Finds the number of edges in an n-th generation (u,v)-flower network.

    Args:
        u (int): Value of u, i.e. path length of one of the parallel paths.
        v (int): Value of v, i.e. path length of one of the parallel paths.
        n (int): Number of generations.

    Returns:
        (int): The number of edges in the network.
    """
    # A (u,v)-flower has (u+v)^n edges.
    return (u + v) ** n


def uv_flower_no_of_nodes(u, v, n):
    """
    Finds the number of nodes in an n-th generation (u,v)-flower network.

    Args:
        u (int): Value of u, i.e. path length of one of the parallel paths.
        v (int): Value of v, i.e. path length of one of the parallel paths.
        n (int): Number of generations.

    Returns:
        (int): The number of nodes in the network.
    """
    # Calculate the number of nodes

    w = u + v
    return int(((w - 2) / (w - 1)) * (w ** n) + (w / (w - 1)))


def check_uv_flower_no_of_edges(u, v, n):
    """
    Checks the validity of the number of edges of a (u,v)-flower network calculating according to analytical arguments in [1]

    Args:
        u (int): Value of u, i.e. path length of one of the parallel paths.
        v (int): Value of v, i.e. path length of one of the parallel paths.
        n (int): Number of generations.

    Returns:
        (Bool): True if the analytical and empirical value agree, False otherwise.
    """
    # Generate an SHM network. The number of edges is independent of the parameters a, b and T, so they are chosen randomly on the interval [0,1]
    G = generate_uv_flower(u, v, n)

    eG = len(G.edges())  # Empirical number of edges
    est_eG = uv_flower_no_of_edges(u, v, n)  # Estimated number of edges

    # Check if the values are equal
    return eG == est_eG


def check_uv_flower_no_of_nodes(u, v, n):
    """
    Checks the validity of the number of edges of a (u,v)-flower network calculating according to analytical arguments in [1]

    Args:
        u (int): Value of u, i.e. path length of one of the parallel paths.
        v (int): Value of v, i.e. path length of one of the parallel paths.
        n (int): Number of generations.

    Returns:
        (Bool): True if the analytical and empirical value agree, False otherwise.
    """
    # Generate an SHM network. The number of edges is independent of the parameters a, b and T, so they are chosen randomly on the interval [0,1]
    G = generate_uv_flower(u, v, n)

    N = len(G.nodes())  # Empirical number of edges
    est_N = uv_flower_no_of_nodes(u, v, n)  # Estimated number of edges

    # Check if the values are equal
    return N == est_N