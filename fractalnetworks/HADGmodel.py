"""This module contains code to generate Hub Attraction Dynamical Growth networks"""

# Utility modules
import itertools

# Other module imports
from .SHMmodel import *


def generate_HADG_model(m, a, b, T, n, save=False):
    """
    Generates an n-th generation HADG network with parameters m, a, b and T  (Kuang et al., 2013).

    Args:
        m (int)                      : The number of offspring added at each stage.
        a (float)                    : The probability of rewiring an edge if node degree is over the threshold.
        b (float)                    : The probability of rewiring an edge otherwise.
        T (float)                    : The threshold for hubs.
        n (int)                      : The number of iterations to perform of the HADG generative process.
        save (:obj:`bool`, optional) : If True, saved the resulting network to a gml file. Default is False.

    Returns:
        G (networkx.Graph) : The resulting HADG network.
    """
    # In the first generation the network is a path of length 2.
    G = nx.path_graph(2)

    # Apply the generative process n-1 more times.
    for t in range(n - 1):
        G = HADG_iteration(G, m, a, b, T)

    # Save the file if save is True.
    if save:
        # Save the file in the format HADG-model-m-a-b-T-generationn-examplei.gml
        # The method is non-deterministic and so multiple examples need to be saved.
        count = 1
        # Initialise a Boolean variable to False,
        #   where False means that it is yet to be saved, and True means it has been saved.
        saved = False

        # Iterate while the file is yet to be saved.
        while not saved:
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
    Performs one iteration of the HADG generative process.

    Args:
        G (networkx.Graph) : The network in its current state.
        m (int)            : The number of offspring added at each stage.
        a (float)          : The probability of rewiring an edge if node degree is over the threshold.
        b (float)          : The probability of rewiring an edge otherwise.
        T (float)          : The threshold for hubs.

    Returns:
        G (networkx.Graph) : The updated network.
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
        deg_x = int(degrees[edge[0]])
        # Find the degree of the target node in the previous generation.
        deg_y = int(degrees[edge[1]])
        # Find the maximum degree in the previous generation.
        deg_max = int(max([degrees[v] for v in degrees]))

        # If the degree of both the source and target node, relative to the maximum degree, are over the threshold then
        #    rewire the edge with probability = a.
        if deg_x / deg_max > T and deg_y / deg_max > T:
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
        possible_edges = [i for i in itertools.combinations(neighbours, 2)]

        # Choose d random new edges where d is the degree of the node at the previous stage.
        new_edges = random.sample(possible_edges, min(deg, len(possible_edges)))
        # Add these new edges to the network.
        G.add_edges_from(new_edges)

    # Return the updated network.
    return G
