"""Functions to generate nested Barabasi-Albert networks."""

# Network analysis modules
import networkx as nx

# Utility module imports
import random
import os


def generate_nested_BA_model(m, N, kmax, save=False):
    """
    Generated a nested Barabasi-Albert (BA) network of order > N with parameters m, k [6].
    Some of this code is taken from the NetworkX barabasi_albert_graph function.

    Args:
        m (int): The number of edges attached to a newly created node in the generation of a BA network.
        N (int): The minimum order (number of nodes) of the network.
        kmax (int): The maximum degree of the original BA network [6].
        save (Bool) (opt): If True, saved the resulting network to a gml file. Default is False.

    Returns:
        G (networkx.Graph): The resulting nested BA network.

    """
    # Generate a BA network with maximum degree kmax
    G = generate_nested_BA_model_subnetwork(m, kmax)

    # Find the degree distribution of the network
    degrees = G.degree()
    # Create a list of nodes in the network where each node is repeated as many times as it has edges.
    # This is used to select a node with probability proportional to its degree.
    repeated_nodes = [v for v, d in G.degree() for _ in range(d)]

    # Continue to generate the network while there are less than N nodes.
    while len(G) < N:
        # Choose a node at random from the list of repeated nodes, with probability proportionate to its degree.
        node = random.choice(repeated_nodes)

        # Generate a BA network with maximum degree k, where k is the degree of the chosen node.
        H = generate_nested_BA_model_subnetwork(m, G.degree(node))

        # Relabel the nodes in the new subnetwork, so that they can be added to the existing graph.
        # Nodes are labeled from 0 to N-1, so the next nodes should be numbered N, N+1, ...
        new_nodes = list(H.nodes())
        H = nx.relabel_nodes(H, dict(zip(new_nodes, [i + len(G) for i in new_nodes])))
        new_nodes = list(H.nodes())  # Take a list of the newly added nodes.

        # Each of the edges of the chosen node need to be removed and reattached to new nodes chosen at random.
        to_be_reconnected = [i for i in G.neighbors(node)]  # Store each edges' source node
        G.remove_node(node)  # Remove the chosen node
        G = nx.union(G, H)  # Add the new subnetwork
        # For each source node, chose a random node from the newly added subnetwork and add an edge between them.
        for source in to_be_reconnected:
            target = random.choice(new_nodes)
            G.add_edge(source, target)

        # Relabel the nodes so that they are numbered from 0 to N-1
        G = nx.convert_node_labels_to_integers(G, first_label=0)

        # Update the list of repeated nodes.
        repeated_nodes = [v for v, d in G.degree() for _ in range(d)]

    # Save the file is save is True.
    if save == True:
        # Save the file in the format nested-BA-model-m-N-kmax-examplei.gml
        # For p != 0 the method is non-deterministic and so multiple examples need to be saved.
        count = 1
        # Initialise a Boolean variable to False, where False means that it is yet to be saved, and True means it has been saved.
        saved = False

        # Iterate while the file is yet to be saved.
        while saved == False:
            filename = "nested-BA-model-" + str(m) + "-" + str(N) + "-" + str(kmax) + "-example" + str(count) + ".gml"
            filepath = "network-files/models/nested-BA-model/" + filename
            # If the count-th example already exists, increment the count by 1 and try to save again.
            if not os.path.isfile(filepath):
                nx.write_gml(G, filepath)
                saved = True
            else:
                count += 1

    return G


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