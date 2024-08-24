"""This module contains code to generate Song-Havlin-Makse networks"""

# Network analysis modules
import networkx as nx

# Mathematics modules
import random
import numpy as np

# Utility modules
import os


def generate_SHM_model(m, p, n, save=False):
    """
    Generates an n-th generation SHM network with parameters m and p (Song, Havlin, and Makse, 2006).

    Args:
        m (int)                      : The number of offspring added at each stage, as defined by the SHM model.
        p (float)                    : The probability of rewiring an edge, as defined by the SHM model.
                                        The value of p should be between 0 and 1.
        n (int)                      : The number of iterations to perform of the SHM generative process.
        save (:obj:`bool`, optional) : If True, the file is saved to the network-files folder. Default is False.

    Returns:
        (networkx.Graph) : The generated SHM network.
    """

    # In the first generation the network is a path of length 2.
    G = nx.path_graph(2)

    # Generate the next generation n-1 more times.
    for t in range(n - 1):
        G = SHM_iteration(G, m, p)

    # Save the file if save is True.
    if save:
        # Save the file in the format SHM-model-m-p-generationn-examplei.gml
        # For p != 0 the method is non-deterministic and so multiple examples need to be saved.
        count = 1
        # Initialise a Boolean variable to False,
        #   where False means that it is yet to be saved, and True means it has been saved.
        saved = False

        # Iterate while the file is yet to be saved.
        while not saved:
            filename = "SHM-model-" + str(m) + "-" + str(p) + "-generation" + str(n) + "-example" + str(count) + ".gml"
            filepath = "network-files/models/SHM-model/" + filename
            # If the count-th example already exists, increment the count by 1 and try to save again.
            if not os.path.isfile(filepath):
                nx.write_gml(G, filepath)
                saved = True
            else:
                count += 1

    # Return the generated graph
    return G


def SHM_iteration(G, m, p):
    """
    Performs one iteration of the generative process for the SHM model.

    Args:
        G (networkx.Graph) : The network from the previous generation.
        m (int)            : The number of offspring added at each stage, as defined by the SHM model.
        p (float)          : The probability of rewiring an edge, as defined by the SHM model.
                                The value of p should be between 0 and 1.

    Returns:
        (networkx.Graph) : The network in the new generation.
    """
    # Find a list of all existing edges in the network.
    edges = list(G.edges())

    # For each edge, add m offspring to each endpoint of the edge.
    for edge in edges:
        G, source_offspring = add_m_offspring(G, edge[0], m)
        G, target_offspring = add_m_offspring(G, edge[1], m)

        # With probability p, rewire the original edge.
        if random.random() <= p:
            rewire_offspring(G, edge, source_offspring, target_offspring)

    # Return the updated network.
    return G


def add_m_offspring(G, node, m):
    """
    Adds m offspring to a given node, connected by a single edge.

    Args:
        G (networkx.Graph) : The network in its current state.
        node (int)         : The node to which the offspring are to be added.
        m (int)            : The number of offspring.

    Returns:
        (tuple) : Tuple containing a networkx.Graph and list, where the graph is the updated network, and the list is
                     the names of all the new nodes added at the previous stage.
    """
    # Find the number of nodes in the network
    N = len(G.nodes())

    # Add new nodes, named for the next available i integers.
    new_nodes = [N + i for i in range(m)]
    # Create a list of new edges, one connecting each new node to the original node.
    new_edges = zip(new_nodes, [node] * m)

    # Add the new edges and vertices
    G.add_nodes_from(new_nodes)
    G.add_edges_from(new_edges)

    # Return the updated graph, and a list of the new nodes.
    return G, new_nodes


def rewire_offspring(G, edge, source_offspring, target_offspring):
    """
    Rewires a given edge, replacing it with an edge between two randomly chosen offspring of the current endpoints.

    Args:
        G (networkx.Graph)      : The network in its current state.
        edge (tuple)            : The edge to be rewired.
        source_offspring (list) : A list of offspring of the source node.
        target_offspring (list) : A list of offspring of the target node.

    Returns:
        (networkx.Graph) : The updated network.
    """
    # Remove the original edge.
    G.remove_edge(edge[0], edge[1])

    # Choose a new source at random from the original source node's offspring.
    new_source = random.choice(source_offspring)
    # Choose a new target at random from the original target node's offspring.
    new_target = random.choice(target_offspring)

    # Add a new edge between the new source and target.
    G.add_edge(new_source, new_target)

    # Return the updated network.
    return G


def generate_SHM_model_for_all_p(m, n, prob_N=11, example_N=1, save=False):
    """
    Generate SHM networks for a fixed m and n, for multiple values of p.

    Args:
        m (int)                          : The number of offspring added at each stage, as defined by the SHM model.
        n (int)                          : The number of iterations to perform of the SHM generative process.
        prob_N (int)                     : The number of probabilities to generate the graph for.
                                            Default is 11, so will generate networks with p in [0.0, 0.1, ..., 0.9, 1.0]
        example_N (:obj:`int`, optional) : The number of networks of each probability to generate. Default is 1.
        save (:obj:`bool`, optional)     : If True, save each of the networks to a .gml file. Default is False.

    Returns:
        (tuple): Tuple containing numpy.ndarray and list, specifically:
                    an array of the probabilities which are used;
                    a list of lists of networks. The sublist graphs[i] contains graphs with p=probabilities[i].
    """
    # Generate the list of probabilities p.
    probabilities = np.linspace(0, 1, prob_N)

    # Initialise an empty list to store networks.
    graphs = []

    # Iterate for each probability.
    for p in probabilities:
        # Initialise an empty list to store networks with probability p.
        p_graphs = []
        # Generate such a network and add it to the list.
        for i in range(example_N):
            G = generate_SHM_model(m, p, n, save=save)
            p_graphs.append(G)
        # Add the list of networks with probability p to the overall list.
        graphs.append(p_graphs)

    # Return a list of probabilities and graphs.
    return probabilities, graphs


def retrieve_SHM_model(m, p, n, example=1):
    """
    Given the parameters m, p, n and the example number, retrieve the file containing the SHM network.

    Args:
        m (int): The number of offspring added at each stage, as defined by the SHM model [3].
        p (float): The probability of rewiring an edge, as defined by the SHM model [3].
        n (int): The number of iterations to perform of the SHM generative process [3].
        example (int) (opt): In the case of multiple graphs with the same parameters, specifies the example wanted.
                                Default is 1.

    Returns:
        (tuple): A tuple containing a networkx.Graph and str, specifically:
                    the saved SHM network with the above specified parameters;
                    the filepath to the network file.
    """

    # Find the filepath to the model with these parameters.
    filename = "SHM-model-" + str(m) + "-" + str(p) + "-generation" + str(n) + "-example" + str(example) + ".gml"
    filepath = "network-files/models/SHM-model/"

    # Read the network.
    G = nx.read_gml(filepath + filename)

    # Return the networkx graph.
    return G, filepath + filename
