"""Implements the greedy colouring box covering algorithm"""

# Network analysis modules
import igraph

# Mathematics modules
import numpy as np


def smallest_last(G):
    """
    Orders nodes using the smallest last (Matula and Beck, 1983) method.

    Args:
        G (igraph.Graph) : The network being ordered.

    Returns:
        (list) : A list of the nodes in order.
    """
    # Find the number of nodes.
    N = G.vcount()

    # Find a list of the nodes.
    nodes = [i for i in range(N)]

    # Initialise an empty list for the ordering.
    ordering = []

    # Create a local copy of the network
    H = G.copy()

    # Iterate whilst there are still nodes that need to be ordered.
    while len(ordering) < N:
        # Find the degrees of all the nodes.
        degrees = H.degree()
        # Find the index of the node with the smallest degree.
        min_i = np.argmin(np.array(degrees))
        node = nodes[min_i]
        # Add that node to the start of the ordering.
        ordering.insert(0, node)
        # Remove that node from the graph.
        nodes.pop(min_i)
        H.delete_vertices(min_i)

    return ordering


def greedy_box_covering(G, lB, node_order=None):
    """
    Colours the network in boxes of diameter lB using the greedy algorithm (Song, Gallos, et al., 2007).

    Args:
        G (igraph.Graph)                   : The network to be analysed.
        lB (int)                           : The diameter of the boxes for the box covering.
        node_order (:obj:`list`, optional) : The order of the nodes in which the greedy colouring is applied.
                                             If None, the nodes are coloured in lexicographical order.
                                             Default is None.

    Returns:
        (tuple) : Tuple containing a dict and int.
                    The dict represents the greedy colouring, with the nodes as keys and the colours as values.
                    The int is the number of boxes of diameter lB needed to cover the network.
    """
    # Find the dual graph
    dual_G = make_dual_graph(G, lB)

    # Find the graph colouring
    colouring = greedy_colouring(dual_G, node_order=node_order)

    # The number of boxes is the number of colours used in the colouring.
    NB = max(list(colouring.values())) + 1

    # Return the colouring dictionary and the number of boxes.
    return colouring, NB


def greedy_colouring(G, node_order=None):
    """
    Colour a graph by the greedy colouring algorithm (Song, Gallos, et al., 2007).

    Args:
        G (igraph.Graph)                   : The graph to be analysed.
        node_order (:obj:`list`, optional) : The order of the nodes in which the greedy colouring is applied.
                                             If None, the nodes are coloured in lexicographical order.
                                             Default is None.

    Returns:
        (dict): A dictionary with nodes as keys and the colours they are assigned as values.
    """

    # Find the number of nodes.
    N = G.vcount()

    # Find a list of the nodes.
    nodes = [i for i in range(N)]
    # Initialise an empty dictionary with the nodes as keys.
    colouring = dict.fromkeys(nodes)

    # Apply the ordering method to the nodes, if one is given.
    if not node_order:
        node_order = nodes

    # Iterate through all the nodes.
    for node in node_order:
        # Initialise an empty list for the forbidden colours.
        forbidden_colours = []
        # Find the neighbours of the given node.
        neighbours = G.neighbors(node)

        # Iterate through each of the neighbours of that node.
        for neighbour in neighbours:
            # Find the colour each neighbour is assigned.
            neighbour_colour = colouring[neighbour]

            # If the neighbour has a colour, then add it to the list of forbidden colours.
            if neighbour_colour is not None:
                forbidden_colours.append(neighbour_colour)

        # If there are no forbidden colours, i.e. none of the neighbours have colours, then assign the node colour 0.
        if len(forbidden_colours) == 0:
            colour = 0
        # Otherwise find the next minimum value and assign that colour to the node.
        else:
            colours = [i for i in range(max(forbidden_colours) + 2)]
            possible_colours = list(set(colours) - set(forbidden_colours))
            colour = min(possible_colours)
        # Assign the chosen colour to the node
        colouring[node] = colour

    # Return the complete covering.
    return colouring


def make_dual_graph(graph, lB):
    """
    Finds the dual graph as defined under the greedy colouring algorithm (Song, Gallos, et al., 2007).
    In the dual graph, two nodes are connected if the distance between them is at least lB.

    Args:
        graph (igraph.Graph) : The graph to be analysed.
        lB (int)             : The diameter of the boxes for the box covering.

    Returns:
        (igraph.Graph) : The dual graph.
    """
    # Calculate the matrix of shortest paths between each pair of nodes in the network, and convert it to a numpy array.
    distance_matrix = graph.distances()
    distance_np_array = np.array(distance_matrix)

    # The following lines of code convert the distance matrix into a matrix where the entries are:
    #   one if the nodes are a distance of at least lB apart, and
    #   zero otherwise.
    distance_np_array[distance_np_array < lB] = 0
    distance_np_array[distance_np_array > 0] = 1

    # Create the dual graph based on the adjacency matrix defined above.
    dual_graph = igraph.Graph.Adjacency(distance_np_array)
    dual_graph.to_undirected()

    # Return the dual graph.
    return dual_graph
