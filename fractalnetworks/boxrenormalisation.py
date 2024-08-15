"""This module contains the code to renormalise a network using a box covering"""

import random
from igraph import Graph
import networkx as nx
import igraph
import matplotlib.pyplot as plt
import os
import datetime

from .maximumexcludedmassburning import *


def find_central_distance(G, centres):
    """
    Finds the central distance for each node in a network given a list of centres.

    Args:
        G (igraph.Graph): The network to be analysed.
        centres (list): A list of centre nodes from the MEMB algorithm.

    Returns:
        central_distance (dict): A dictionary containing nodes as keys and their central distance as values.
    """

    # Initialise an empty dictionary to store the values for the central distance.
    central_distance = {}

    # Iterate through each of the nodes in the network.
    for v in G.vs():

        # Initialise an empty variable for the shortest path length.
        shortest_path_len = None

        # If the node v is a centre then it must have central distance 0, so check for this case to speed up the algorithm.
        if v["label"] in centres:
            central_distance[v["label"]] = 0

        # For all non-centre nodes v, iterate through the list of all centres.
        else:
            for u in G.vs():
                if u["label"] in centres:
                    # Find the shortest path length between the node v and a centre u.
                    path_len = G.distances(source=v, target=u)

                    # If this is a new minimum, then update the shortest path variable
                    if shortest_path_len == None or shortest_path_len > path_len:
                        shortest_path_len = path_len

            # Assign the value of the shortest path length to the node in the dictionary.
            central_distance[v["label"]] = shortest_path_len[0][0]

    # Once all nodes are checked return the values in the dictionary.
    return central_distance


def assign_nodes_to_boxes(G, centres, central_distance):
    """
    Generates a dictionary assigning each node to a box under the MEMB algorithm.

    Args:
        G (igraph.Graph): The network to be analysed.
        centres (list): A list of centre nodes according to the MEMB algorithm.
        central_distance (dict): A dictionary containing nodes as keys and their central distance as values.

    Returns:
        nodes_to_boxes (dict): A dictionary containing nodes as keys and the box they are assigned to as the value.
    """

    # Initialise an empty dictionary to store the boxes for each node.
    nodes_to_boxes = {}

    # Find the list of all nodes which are non-centres.
    nodes = G.vs()
    node_labels = [v["label"] for v in G.vs()]
    non_centres = list(set(node_labels) - set(centres))

    # The following section of code produces a list of non-centres in order of increasing central distance.
    sorted_non_centres = []  # Initialise an empty list of non-centres.
    sorted_dict = dict(sorted(central_distance.items(),
                              key=itemgetter(1)))  # Sort the dictionary of central distances into increasing order.
    # Add each node to the list of sorted non-centres in order.
    for key in sorted_dict:
        if not key in centres:
            sorted_non_centres.append(key)

    id = 0  # The ID of the first box is zero.
    # For each of the centres assign a unique box ID.
    for node in centres:
        nodes_to_boxes[node] = id
        id += 1  # Increment the box ID.

    # Iterate through each of the non-centres
    for node in sorted_non_centres:
        # Initialise an empty list of possible boxes the node can belong to.
        possible_boxes = []
        # Find the neighbours which have central distance strictly less than the current node.
        for neighbour in G.neighbors(int(node)):
            if central_distance[node] > central_distance[str(neighbour)]:
                # For each, add their box to the list of possible boxes for the current node.
                possible_boxes.append(nodes_to_boxes[str(neighbour)])
        # Make a random choice from the list of possible boxes and assign that box to the node.
        nodes_to_boxes[node] = random.choice(possible_boxes)

    # Once all nodes have been checked return a dictionary containing the mapping from all of the nodes to a box.
    return nodes_to_boxes


def find_boxes(nodes_to_boxes, centres):
    """
    Finds a list of nodes assigned to each box in a network.

    Args:
        nodes_to_boxes (dict): A dictionary with nodes as keys and their corresponding boxes as values.
        centres (list): A list of the nodes found as centres under the MEMB algorithm.

    Returns:
        boxes (dict): A dictionary with boxes as keys and a list of nodes in that box as the value.
    """
    # Initialise an empty dictionary to store the boxes.
    boxes = {}

    # The box IDs are 0, ..., k-1 where k is the number of centres.
    for i in range(len(centres)):  # Iterate over the box IDs.
        # Initialise an empty list of nodes.
        nodes = []
        # Check if each node belongs in the current box.
        for node in nodes_to_boxes:
            # If it does, add it to the list of nodes.
            if nodes_to_boxes[node] == i:
                nodes.append(node)

        # Assign the list of nodes to the box.
        boxes[i] = nodes

    # Return the dictionary of boxes to nodes.
    return boxes


def renormalise_graph(G, boxes, nodes_to_boxes, draw=False):
    """
    Renormalise a graph under a given box-covering.

    Args:
        G (igraph.Graph): The network to be analysed.
        boxes (dict): A dictionary with boxes as keys and a list of nodes in that box as the value.
        nodes_to_boxes (dict): A dictionary with nodes as keys and their corresponding boxes as values.
        draw (Bool) (opt): If True then displays the renormalised graph. Default is False.

    Returns:
        renormalisedG (networkx.Graph): The network under renormalisation.
    """
    # Initialise an empty graph to be the renormalised graph of G.
    renormalisedG = Graph()

    # Add one supernode for each of the boxes found under the MEMB algorithm.
    box_list = list(boxes.keys())
    for box in box_list:
        renormalisedG.add_vertices([box])

    for node in renormalisedG.vs():
        node["label"] = str(node["name"])
        node["id"] = float(node["name"])

    # Iterate through each of the edges in the original graph.
    for edge in G.es():
        # Find the nodes originally connected by the edge.
        source = edge.source
        target = edge.target

        # Find the supernodes these nodes now belong to.
        renormalised_source = nodes_to_boxes[str(source)]
        renormalised_target = nodes_to_boxes[str(target)]

        # Create a new edge between the supernodes.
        renormalisedG.add_edges([(renormalised_source, renormalised_target)])

    # Simplify the graph by removing any self loops (edges from a supernode to itself).
    renormalisedG.simplify()

    if draw:
        nxG = renormalisedG.to_networkx()

        nx.draw_kamada_kawai(nxG, node_color=list(nxG.nodes()))

    # Return the renormalised graph.
    return renormalisedG


def find_boxes_and_renormalise_iteration(G, lB, iter_count=1, filepath="graph", method=accelerated_MEMB, draw=False):
    """
    Performs one iteration of box covering and renormalisation.
    Stores all results in new files.

    Args:
        G (igraph.Graph): The network to be analysed.
        lB (int): The diameter of the boxes for the box covering.
        iter_count (int) (opt): The current iteration number. Default is 1 if no value is given.
        filepath (str) (opt): The path to which the resulting box covered and renormalised graphs will be saved. Default is "graph".
        method (func) (opt): The MEMB method used to find the box covering. Default is degree_based_MEMB.
        draw (Bool) (opt): If True then display the networks. Default is False.

    Returns:
        renormalisedG (igraph.Graph): The network after box renormalisation.
    """

    # Find the list of centres using the given MEMB method.
    centres = method(G, lB)

    # Calculate the central distance for each node.
    central_distance = find_central_distance(G, centres)

    # Assign each node to a box.
    nodes_to_boxes = assign_nodes_to_boxes(G, centres, central_distance)

    # Initialise an empty colour map for the box covering visualiation.
    colourmap = []

    # For each node, assign it the colour of its box ID.
    for node in range(G.vcount()):
        colourmap.append(nodes_to_boxes[str(node)])

    nxG = G.to_networkx()

    # If draw is True then display the graph with the colours indicating the box the node belongs to.
    if draw:
        plt.figure(1)
        nx.draw_kamada_kawai(nxG, node_color=colourmap)
        plt.figure(2)  # Start a second figure for the renormalised graph

    # Find a list of nodes for each of the boxes.
    boxes = find_boxes(nodes_to_boxes, centres)

    # Create a file path to store the graph with the box covering.
    boxes_file_path = filepath + "/boxes_iter_" + str(iter_count) + ".gml"
    # Export the graph to gephi using the boxes to colour the nodes.
    # export_to_gephi(nxG, nodes_to_boxes, boxes_file_path)

    # Create a file path to store the renormalised graph.
    renormalised_file_path = filepath + "/renormalised_iter_" + str(iter_count) + ".gml"
    # Find the renormalised graph and draw it if draw is true.
    renormalisedG = renormalise_graph(G, boxes, nodes_to_boxes, draw=draw)

    # The nodes in the renormalised graph should be coloured the same as the boxes they originated from.
    # Create a dictionary which assigns each node the colour of the box (which is the same as the name of the supernode/box)
    gephi_dict = {}
    for node in range(renormalisedG.vcount()):
        gephi_dict[node] = node

    nx_renormalisedG = renormalisedG.to_networkx()

    # Export the renormalised graph to gephi.
    export_to_gephi(renormalisedG, gephi_dict, renormalised_file_path)

    # If draw is True then display the graphs.
    if draw:
        plt.show()

    # Return the renormalised graph.
    return renormalisedG


def renormalise_iteratively(filepath, lB, method=accelerated_MEMB, draw=False):
    """
    Iteratively finds the box covering and then renormalises the network until only one node is left.

    Args:
        filepath (str): The filepath to the network file.
        lB (int): The diameter of the boxes for the box covering.
        method (func) (opt): The MEMB method used to find the box covering. Default is degree_based_MEMB.
        draw (Bool) (opt): If True then display the networks. Default is False.

    Returns:
        None
    """

    # Read the graph in from the given filepath.
    G = Graph.Load(filepath)

    # Take the name of the file without the file type extension and folders as the path to save the results to.
    savepath = filepath.split('.')[0]  # Remove type extension
    savepath = savepath.split('/', 1)[1]  # Remove the network-files folder
    savepath = "result-files/" + savepath  # Add the path to the result files
    savepath = savepath + "_" + datetime.today().strftime(
        '%d-%m-%Y')  # Add todays date to the filepath in case of duplicates.

    if not os.path.exists(savepath):
        os.makedirs(savepath)  # Make a new folder to store the results
    else:  # If such a folder already exists then raise an error.
        raise ValueError(
            'A folder {0} already exists for this graph today. Please change the name of this folder manually and try again.'.format(
                savepath))

    # Start with the graph given.
    current_graph = G.copy()
    # Set a counter for the number of iterations.
    iter_count = 1

    # Keep renormalising while there are multiple nodes in the graph.
    while current_graph.vcount() > 1:
        # Find the box covering and renormalise the graph.
        new_graph = find_boxes_and_renormalise_iteration(current_graph, lB, iter_count=iter_count, filepath=savepath,
                                                         method=method, draw=draw)

        # Update the current graph.
        current_graph = new_graph.copy()
        # Increment the counter.
        iter_count += 1
