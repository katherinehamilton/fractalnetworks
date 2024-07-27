"""This module contains functions to generate instances of known fractal network models"""

# Network analysis modules
import networkx as nx

# Mathematics modules
import random
import numpy as np

# Utility modules
import os
from itertools import combinations


# SHM Model


def generate_SHM_model(m, p, n, save=False):
    """
    Generates an n-th generation SHM network with parameters m and p.

    Args:
        m (int): The number of offspring added at each stage, as defined by the SHM model [3].
        p (float): The probability of rewiring an edge, as defined by the SHM model [3]. The value of p should be between 0 and 1.
        n (int): The number of iterations to perform of the SHM generative process [3].
        save (bool) (opt): If True, the file is saved to the network-files folder. Default is False.

    Returns:
        G (networkx.Graph): The generated SHM network.
    """

    # In the first generation the network is a path of length 2.
    G = nx.path_graph(2)

    # Generate the next generation n-1 more times.
    for t in range(n - 1):
        G = SHM_iteration(G, m, p)

    # Save the file is save is True.
    if save:
        # Save the file in the format SHM-model-m-p-generationn-examplei.gml
        # For p != 0 the method is non-deterministic and so multiple examples need to be saved.
        count = 1
        # Initialise a Boolean variable to False, where False means that it is yet to be saved, and True means it has been saved.
        saved = False

        # Iterate while the file is yet to be saved.
        while saved == False:
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
        G (networkx.Graph): The network from the previous generation.
        m (int): The number of offspring added at each stage, as defined by the SHM model [3].
        p (float): The probability of rewiring an edge, as defined by the SHM model [3]. The value of p should be between 0 and 1.

    Returns:
        G (networkx.Graph): The network in the new generation.
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
        G (networkx.Graph): The network in its current state.
        node (int): The node to which the offspring are to be added.
        m (int): The number of offspring.

    Returns:
        G (networkx.Graph): The updated network.
        new_nodes (list): A list of the names of all the new nodes added at the previous stage.
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
    Rewires a given edge, replacing it with an edge between two randomly chosen offspring of the current endpoints [3].

    Args:
        G (networkx.Graph): The network in its current state.
        edge (tuple): The edge to be rewired.
        source_offspring (list): A list of offspring of the source node.
        target_offspring (list): A list of offspring of the target node.

    Returns:
        G (networkx.Graph): The updated network.
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
        m (int): The number of offspring added at each stage, as defined by the SHM model [3].
        n (int): The number of iterations to perform of the SHM generative process [3].
        prob_N (int): The number of probabilities to generate the graph for. Default is 11, so will generate networks with p in [0.0, 0.1, 0.2, ..., 0.9, 1.0]
        example_N (int) (opt): The number of networks of each probability to generate. Default is 1.
        save (Bool) (opt): If True, save each of the networks to a .gml file. Default is False.

    Returns:
        probabilities (numpy.ndarray): A list of the probabilities which are used.
        graphs (list): A list of lists of generated networks. The sublist graphs[i] contains graphs with p=probabilities[i].
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
        example (int) (opt): In the case of multiple graphs with the same parameters, specifies the example wanted. Default is 1.

    Returns:
        G (networkx.Graph): The saved SHM network with the above specified parameters.
        filepath + filename (str): The filepath to the network file.
    """

    # Find the filepath to the model with these parameters.
    filename = "SHM-model-" + str(m) + "-" + str(p) + "-generation" + str(n) + "-example" + str(example) + ".gml"
    filepath = "network-files/models/SHM-model/"

    # Read the network.
    G = nx.read_gml(filepath + filename)

    # Return the networkx graph.
    return G, filepath + filename

# HADG Model

def generate_HADG_model(m, a, b, T, n, save=False):
    """
    Generates an nth generation HADG network with parameters m, a, b and T [4][5].

    Args:
        m (int): The number of offspring added at each stage, as according to [5].
        a (float): The probability of rewiring an edge if node degree is over the threshold [5].
        b (float): The probability of rewiring an edge otherwise [5].
        T (float): The threshold as according to [5].
        n (int): The number of iterations to perform of the HADG generative process.
        save (Bool) (opt): If True, saved the resulting network to a gml file. Default is False.

    Returns:
        G (networkx.Graph): The resulting HADG network.
    """
    # In the first generation the network is a path of length 2.
    G = nx.path_graph(2)

    # Apply the generative process n-1 more times.
    for t in range(n - 1):
        G = HADG_iteration(G, m, a, b, T)

    # Save the file is save is True.
    if save == True:
        # Save the file in the format HADG-model-m-a-b-T-generationn-examplei.gml
        # The method is non-deterministic and so multiple examples need to be saved.
        count = 1
        # Initialise a Boolean variable to False, where False means that it is yet to be saved, and True means it has been saved.
        saved = False

        # Iterate while the file is yet to be saved.
        while saved == False:
            filename = "HADG-model-" + str(m) + "-" + str(a) + "-" + str(b) + "-" + str(T) + "-generation" + str(
                n) + "-example" + str(count) + ".gml"
            filepath = "network-files/models/HADG-model/" + filename
            # If the count-th example already exists, increment the count by 1 and try to save again.
            if not os.path.isfile(filepath):
                nx.write_gml(G, filepath)
                saved = True
            else:
                count += 1

    # Return the generated graph.
    return G


def HADG_iteration(G, m, a, b, T):
    """
    Performs one iteration of the HADG generative process [4][5].

    Args:
        G (networkx.Graph): The network in its current state.
        m (int): The number of offspring added at each stage, as according to [5].
        a (float): The probability of rewiring an edge if node degree is over the threshold [5].
        b (float): The probability of rewiring an edge otherwise [5].
        T (float): The threshold as according to [5].

    Returns:
        G (networkx.Graph): The updated network.
    """
    # Initialise a dictionary where each node is a key and each value starts as an empty list.
    # This dictionary will store a list of the newly added nodes to which a previously existing node is adjacent.
    new_node_dict = dict.fromkeys(list(G.nodes()), [])

    # Find a list of the edges in the previous generation.
    edges = list(G.edges())
    # Find a list of the node degrees in the previous generation.
    degrees = dict(G.degree())

    # Iterate through each of the edges in the previous generation.
    for edge in edges:
        # Add m offspring to the source node of the edge.
        G, source_offspring = add_m_offspring(G, edge[0], m)
        # Add m offspring to the target node of the edge.
        G, target_offspring = add_m_offspring(G, edge[1], m)

        # Find the degree of the source node in the previous generation.
        degx = degrees[edge[0]]
        # Find the degree of the target node in the previous generation.
        degy = degrees[edge[1]]
        # Find the maximum degree in the previous generation.
        degmax = max([degrees[v] for v in degrees])

        # If the degree of both the source and target node, relative to the maximum degree, are over the threshold then
        #    rewire the edge with probability a.
        if degx / degmax > T and degy / degmax > T:
            if random.random() <= a:
                rewire_offspring(G, edge, source_offspring, target_offspring)
        # Otherwise rewire the edge with probability b.
        else:
            if random.random() <= b:
                rewire_offspring(G, edge, source_offspring, target_offspring)

        # Update the dictionary of new neighbours for the source node.
        source_neighbours = new_node_dict[edge[0]].copy()
        source_neighbours.extend(source_offspring)
        new_node_dict[edge[0]] = source_neighbours

        # Update the dictionary of new neighbours for the target node.
        target_neighbours = new_node_dict[edge[1]].copy()
        target_neighbours.extend(target_offspring)
        new_node_dict[edge[1]] = target_neighbours

    # Iterate through each of the nodes from the previous generation.
    for node in new_node_dict:
        # Find the list of new neighbours for that node.
        neighbours = new_node_dict[node]
        # Find the degree of the node at the previous generation.
        deg = degrees[node]

        # Find all possible pairs of new neighbours, i.e. all possible edges between the new offspring of the node.
        possible_edges = [i for i in combinations(neighbours, 2)]

        # Choose d random new edges where d is the degree of the node at the previous stage.
        new_edges = random.sample(possible_edges, min(deg, len(possible_edges)))
        # Add these new edges to the network.
        G.add_edges_from(new_edges)

    # Return the updated network.
    return G


# (u,v)-Flower

def generate_uv_flower(u, v, n, save=False):
    """
    Generates an n-th generation (u,v)-flower.

    Args:
        u (int): Value of u, i.e. path length of one of the parallel paths.
        v (int): Value of v, i.e. path length of one of the parallel paths.
        n (int): Number of generations.
        save (bool) (opt): If True, the file is saved to the network-files folder. Default is False.

    Returns:
        G (networkx.Graph): The generated (u,v)-flower.
    """
    # Initialise a cycle graph of length w = u + v
    G = nx.cycle_graph(u + v)

    # For each of the n-1 remaining generations, perform one iteration of the generative process.
    for i in range(n - 1):
        G = uv_iteration(G, u, v)

    # Save the file in the format uvflower-generationn.gml
    if save:
        filename = str(u) + "_" + str(v) + "_flower-generation" + str(n) + ".gml"
        filepath = "network-files/models/uv-flowers/" + filename
        nx.write_gml(G, filepath)

    return G


def uv_iteration(G, u, v):
    """
    Performs one iteration in the (u,v)-flower generation process.

    Args:
        G (networkx.Graph): The (u,v)-flower network in its current (t-1)-th generation.
        u (int): Value of u, i.e. path length of one of the parallel paths.
        v (int): Value of v, i.e. path length of one of the parallel paths.

    Returns:
        G (networkx.Graph): The (u,v)-flower network in the t-th generation.

    """
    # Find a list of all the nodes and edges in the network at the (t-1)-th generation.
    nodes = list(G.nodes())
    edges = list(G.edges())

    # Remove all the existing edges.
    G.remove_edges_from(G.edges())

    # Iterate through each of the edges from the network in the (t-1)-th generation.
    for edge in edges:
        # n is used to store the smallest integer which isn't yet a node label.
        # The nodes are labelled 0, ..., n-1, so this is n.
        N = len(G.nodes())

        # Replace the edge with a path of length u.
        # First find a path graph using these vertices.
        Hu, N = add_new_path(u, N, edge)
        # Then merge this path graph with the existing network.
        G = nx.compose(G, Hu)

        # Replace the edge with a path of length v.
        # First find a path graph using these vertices.
        Hv, N = add_new_path(v, N, edge)
        # Then merge this path graph with the existing network.
        G = nx.compose(G, Hv)

    # Return the graph after all iterations.
    return G


def generate_non_fractal_uv_flower(v, n):
    """
    Generates a non-fractal (u,v)-flower with u=1.

    Args:
        v (int): Value of v, i.e. path length of the parallel paths.
        n (int): Number of generations.

    Returns:
        (networkx.Graph): The generated (u,v)-flower.
    """
    # Returns the (u,v)-flower found by the fractal generator, but with u hardcoded as 1.
    return generate_uv_flower(1, v, n)

def add_new_path(l, N, edge):
    """
    Adds a new parallel path to the network, a step in the (u, v)-flower generation process.

    Args:
        l (int): The length of the path to be added to the network.
        N (int): A counter which stores the next unused integer to label nodes.
        edge (tuple): The edge from the network being replaced with parallel paths.

    Returns:
        Hl (networkx.Graph): A path graph of length l with vertices labelled correctly.
        N (int): A counter which stores the next unused integer to label nodes.
    """
    # Generate a path graph with l edges (and l+1 vertices).
    Hl = nx.path_graph(l + 1)

    # Create an empty dictionary to be used to relabel the nodes in the path.
    l_rlbl = {key: None for key in list(Hl.nodes)}

    # The nodes in the path graph are labelled from 0 to l.
    # Thus, the node 0 in this path corresponds to the source node of the original edge,
    #   and the node l in this path corresponds to the target node of the original edge.

    # Iterate through all the nodes in the path.
    for node in list(Hl.nodes()):
        # If the node is 0 in the path, relabel it as the source of the original edge.
        if node == 0:
            l_rlbl[node] = edge[0]
        # If the node is l in the path, relabel it as the source of the original edge.
        elif node == l:
            l_rlbl[node] = edge[1]
        # For all other nodes, relabel it as the next unused integer.
        else:
            l_rlbl[node] = N
            # Increment the counter n, so that n is now the next unused integer.
            N += 1

    # Relabel the nodes according to the scheme described above.
    Hl = nx.relabel_nodes(Hl, l_rlbl)

    # Return the path graph to be merged with the (u, v)-flower graph, and the counter for the next unused integer for node labels.
    return Hl, N


# Nested Barabasi-Albert Network

def generate_nested_BA_model_subnetwork(m, kmax):
    """
    Generates a Barabasi-Albert (BA) network with maximum degree kmax.
    Some of this code is from the NetworkX barabasi_albert_graph function.

    Args:
        m (int): The number of edges attached to a newly created node in the generation of a BA network.
        kmax (int): The maximum degree of the BA network.

    Returns:
        G (networkx.Graph): The generated BA network.
    """

    # Begin with a complete graph on m+1 nodes so that each node has m edges.
    G = nx.complete_graph(m + 1)

    # Generate a list of nodes where each node is repeated as many times as its degree.
    degrees = G.degree()
    repeated_nodes = [n for n, d in degrees for _ in range(d)]

    # The label of the next node is the next available integer, which is N (nodes are labelled from 0 to N-1)
    next_node = len(G)

    # Find the maximum degree in the network
    k = max(degrees, key=lambda x: x[1])[1]

    # Iterate while the maximum degree is less than kmax
    while k < kmax:
        # Choose m target nodes in proportion to their degree.
        target_nodes = preferential_choice(repeated_nodes, m)
        # Add edges between the new node and the m randomly chosen neighbours.
        G.add_edges_from(zip([next_node] * m, target_nodes))

        # Add each of the target nodes to the list of repeated nodes once, to represent their newly added edge.
        repeated_nodes.extend(target_nodes)
        # Add the new node m times to the list of repeated nodes.
        repeated_nodes.extend([next_node] * m)

        # Calculate the new maximum degree
        degrees = G.degree()
        k = max(degrees, key=lambda x: x[1])[1]

        # Increment the label for the next node.
        next_node += 1

    # Once the maximum degree is kmax, return the generated network.
    return G


def preferential_choice(repeated_nodes, m):
    """
    Chooses m nodes at random with probability proportionate to their degree.
    Some of this code is from the NetworkX barabasi_albert_graph function.

    Args:
        repeated_nodes (list): A list of nodes where each node is repeated as many times as its degree.
        m (int): The number of nodes to be chosen.

    Returns:
        target_nodes (set): A set of m chosen nodes.
    """
    # Initialise an empty set for the target nodes.
    # We use a set instead of a list so that the m nodes chosen are unique.
    target_nodes = set()

    # Iterate until m nodes are chosen.
    while len(target_nodes) < m:
        # Choose a node at random from the list.
        # Because each node with degree k is repeated k times the probability of it being chosen is k/2e(G)
        x = random.choice(repeated_nodes)
        # Add the chosen node to the set.
        # If the node has been chosen already, it will not be added again.
        target_nodes.add(x)

    # Return the list of chosen nodes.
    return target_nodes

