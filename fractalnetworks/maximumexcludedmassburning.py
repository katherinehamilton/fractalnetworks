"""This module contains MEMB algorithms"""

# Mathematics modules
import random
import numpy as np

# Utilities modules
from operator import itemgetter
import time


def traditional_MEMB(G, lB, deterministic=True):
    """
    Implements the Maximal Excluded Mass Burning algorithm as according to [1].
    Note: Only works for odd numbers of lB, otherwise it takes rB to be the floor of (lB-1)/2 which is the same as for the odd lB-1.

    Args:
        G (igraph.Graph): The network the algorithm is to be applied to.
        lB (int): The diameter of the boxes used to cover the network.
        deterministic (Bool) (opt): If False, choose fom nodes with equal excluded mass randomly. If True, choose the first lexicographically.

    Returns:
        centres (list): A list of nodes assigned to be centres under the MEMB algorithm.
    """

    # If the diameter lB is less than or equal to 2, then the maximum radius is 0 and so every node is in its own box.
    if lB == 1 or lB == 2:
        return list([v["label"] for v in G.vs()])

    number_EM_calcs = 0

    # Start with all nodes being uncovered and non-centres.
    uncovered = G.vs()["label"]
    non_centres = G.vs()

    # Initialise empty lists for the covered and centre nodes.
    covered = []
    centres = []

    # Each box can have diameter of up to lB, so the maximum radius is rB = (lB-1)/2.
    rB = int((lB - 1) / 2)

    # Iterate while there are still nodes uncovered in the graph.
    while len(uncovered) > 0:

        # Start with a maximum excluded mass of zero, and no node p [1].
        p = None
        maximum_excluded_mass = 0

        # For the non-deterministic method, keep a list of nodes with equal maximum excluded mass
        possible_p = []

        # For each node that isn't a centre, find the excluded mass.
        for node in G.vs():
            if node["label"] not in centres:  # Check that the node isn't a centre.
                # The excluded mass is the number of uncovered nodes in within a radius of rB.
                excluded_mass = len(list(set(G.neighborhood([node], order=rB)[0]) - set(covered)))
                number_EM_calcs += 1
                # If the excluded mass of this node is greater than the current excluded mass, choose this node.
                if excluded_mass > maximum_excluded_mass:
                    p = node  # Update p.
                    maximum_excluded_mass = excluded_mass  # Update maximum excluded mass.
                    possible_p = [node]  # Update list of possible nodes for non-deterministic method.
                # If the excluded mass of this node is equal to the current maximum excluded mass, then add this node to the list of possible p.
                elif excluded_mass == maximum_excluded_mass:
                    possible_p.append(node)

        # If the non-deterministic method is chosen, then randomly choose a node from the list of possible p.
        if not deterministic:
            p = random.choice(possible_p)

        # Add the chosen p to the list of centres.
        centres.append(p["label"])

        # Find the graph centred on the node p with radius rB.
        H = G.neighborhood([p], order=rB)[0]
        # Iterate through the nodes in this subgraph.
        for node in H:
            covered.append(node)  # Cover the nodes in the subgraph.
            # Remove these nodes from the list of uncovered nodes.
            if str(node) in uncovered:
                uncovered.remove(str(node))

                # Once all the nodes are covered, return the list of centres.
    return centres, number_EM_calcs


def MEMB(G, lB, deterministic=True):
    """
    Implements the Maximal Excluded Mass Burning algorithm as according to [1], with some time improvements.
    In this version, the nodes within a radius r_B of a node are only found once, and then updated using the list of covered nodes.
    Note: Only works for odd numbers of lB, otherwise it takes rB to be the floor of (lB-1)/2 which is the same as for the odd lB-1.

    Args:
        G (igraph.Graph): The network the algorithm is to be applied to.
        lB (int): The diameter of the boxes used to cover the network.
        deterministic (Bool) (opt): If False, choose fom nodes with equal excluded mass randomly. If True, choose the first lexicographically.

    Returns:
        centres (list): A list of nodes assigned to be centres under the MEMB algorithm.
    """

    # If the diameter lB is less than or equal to 2, then the maximum radius is 0 and so every node is in its own box.
    if lB == 1 or lB == 2:
        return list([v["label"] for v in G.vs()])

    # Start with all nodes being uncovered and non-centres.
    uncovered = G.vs()["label"]
    non_centres = G.vs()

    # Initialise empty lists for the covered and centre nodes.
    covered = []
    centres = []

    # Each box can have diameter of up to lB, so the maximum radius is rB = (lB-1)/2.
    rB = int((lB - 1) / 2)

    # Initialise an empty dictionary to store nodes and a list of nodes in the graphs centred on these nodes with a radius rB.
    # Doing this stops us from having to generate the same subgraphs multiple times, which is expensive.
    eg_dict = {}

    # For each node find the neighbours of that node within a radius rB.
    for node in G.vs():
        [neighbors] = G.neighborhood([node], order=rB)
        eg_dict[node["label"]] = neighbors  # Add the list of nodes to the dictionary.

    # Iterate while there are still nodes uncovered in the graph.
    while len(uncovered) > 0:

        # Start with a maximum excluded mass of zero, and no node p [1].
        p = None
        maximum_excluded_mass = 0

        # For the non-deterministic method, keep a list of nodes with equal maximum excluded mass
        possible_p = []

        # For each node that isn't a centre, find the excluded mass.
        for node in G.vs():
            if node["id"] not in centres:  # Check that the node isn't a centre.
                # The excluded mass is the number of uncovered nodes in within a radius of rB.
                excluded_mass = len(list(set(eg_dict[node["label"]]) - set(covered)))
                # If the excluded mass of this node is greater than the current excluded mass, choose this node.
                if excluded_mass > maximum_excluded_mass:
                    p = node  # Update p.
                    maximum_excluded_mass = excluded_mass  # Update maximum excluded mass.
                    possible_p = [node]  # Update list of possible nodes for non-deterministic method.
                # If the excluded mass of this node is equal to the current maximum excluded mass, then add this node to the list of possible p.
                elif excluded_mass == maximum_excluded_mass:
                    possible_p.append(node)

        # If the non-deterministic method is chosen, then randomly choose a node from the list of possible p.
        if not deterministic:
            p = random.choice(possible_p)

        # Add the chosen p to the list of centres.
        centres.append(p["label"])

        # Find the graph centred on the node p with radius rB.
        H = eg_dict[p["label"]]
        # Iterate through the nodes in this subgraph.
        for node in H:
            covered.append(node)  # Cover the nodes in the subgraph.
            # Remove these nodes from the list of uncovered nodes.
            if str(node) in uncovered:
                uncovered.remove(str(node))

                # Once all the nodes are covered, return the list of centres.
    return centres


def degree_based_MEMB(G, lB, deterministic=True, N=10, verbose=False):
    """
    Implements the Maximal Excluded Mass Burning algorithm as according to [1], adjusted to prioritise nodes with high degree.
    This reduces the running time of the traditional MEMB without losing accuracy.
    Note: Only works for odd numbers of lB, otherwise it takes rB to be the floor of (lB-1)/2 which is the same as for the odd lB-1.

    Args:
        G (igraph.Graph): The network the algorithm is to be applied to.
        lB (int): The diameter of the boxes used to cover the network.
        deterministic (Bool) (opt): If False, choose fom nodes with equal excluded mass randomly. If True, choose the first lexicographically.
        N (int): Takes the top N-th nodes by degree to find the centred subgraph of.

    Returns:
        centres (list): A list of nodes assigned to be centres under the MEMB algorithm.
    """

    # If the diameter lB is less than or equal to 2, then the maximum radius is 0 and so every node is in its own box.
    if lB == 1 or lB == 2:
        return list([v["label"] for v in G.vs()])

    # Start with all nodes being uncovered and non-centres.
    uncovered = G.vs()["label"]

    # Initialise empty lists for the covered and centre nodes.
    centres = []
    covered = []

    # Each box can have diameter of up to lB, so the maximum radius is rB = (lB-1)/2.
    rB = int((lB - 1) / 2)

    # Find the N nodes with the greatest degree centrality.
    dc = degree_centrality(G)
    top_N_dc = dict(sorted(dc.items(), key=itemgetter(1), reverse=True)[:N])

    # Initialise an empty dictionary to store nodes and a list of nodes in the graphs centred on these nodes with a radius rB.
    # Doing this stops us from having to generate the same subgraphs multiple times, which is expensive.
    eg_dict = {}
    for node in G.vs():
        # Only check the top N graphs according to degree centrality.
        if node["label"] in top_N_dc:
            [neighbors] = G.neighborhood([node], order=rB)
            eg_dict[node["label"]] = neighbors  # Add the list of nodes to the dictionary.

    # On the initial iteration, set maiden to True.
    maiden = True

    # This variable checks if the algorithm does not find a solution, and then looks at the next N nodes.
    failed = False

    # Iterate while there are still nodes uncovered in the graph.
    while len(uncovered) > 0:

        start = time.time()

        # For all iterations except the first, calculate the new dictionary of nodes in each subgraph of radius rB.
        if not maiden:
            eg_dict = calc_next_iter(G, uncovered, N, dc, eg_dict, rB, centres, p, failed)
        maiden = False

        # Start with a maximum excluded mass of zero, and no node p [1].
        p = None
        maximum_excluded_mass = 0

        # For the non-deterministic method, keep a list of nodes with equal maximum excluded mass
        possible_p = []

        # For each of the top N nodes by degree, find the excluded mass.
        for node in eg_dict:
            # The excluded mass is the number of uncovered nodes in within a radius of rB.
            excluded_mass = len(list(set(eg_dict[node]) - set(covered)))
            # If the excluded mass of this node is greater than the current excluded mass, choose this node.
            if excluded_mass > maximum_excluded_mass:
                p = node  # Update p.
                maximum_excluded_mass = excluded_mass  # Update maximum excluded mass.
                possible_p = [node]  # Update list of possible nodes for non-deterministic method.
            # If the excluded mass of this node is equal to the current maximum excluded mass, then add this node to the list of possible p.
            elif excluded_mass == maximum_excluded_mass:
                possible_p.append(node)

        # If the non-deterministic method is chosen, then randomly choose a node from the list of possible p.
        if not deterministic:
            p = random.choice(possible_p)

        # Check if the method fails to find a node p.
        # The method only fails if every single node has zero uncovered nodes within a radius rB.
        if p == None:
            # If it does fail, remove all the nodes that have already been tried from the list of top N nodes.
            failed = True  # Set the failed variable.
            for tried_node_label in eg_dict:
                dc.pop(tried_node_label)
        else:
            # Otherwise, add the new p to the list of centre nodes.
            centres.append(p)
            # Update the list so that the newly covered nodes are in the list of covered nodes and not in the list of uncovered nodes.
            for node in eg_dict[p]:
                if str(node) in uncovered:
                    uncovered.remove(str(node))
                    covered.append(int(node))
            failed = False  # Reset the failed variable.

        end = time.time()

        if verbose:
            print("\n {0} centres found, {1} uncovered nodes remaining.".format(len(centres), len(uncovered)))
            print("This round took {0} seconds.".format(end - start))

    # Once all the nodes are covered return the list of centres found in the graph.
    return centres


def degree_centrality(G):
    """
    Calculates a dictionary of the degree centralities for all nodes in a graph.
    Analagous to the NetworkX degree_centrality method.

    Args:
        G (igraph.Graph): The network the algorithm is to be applied to.

    Returns:
        dc (dict): A dictionary with node labels as keys and their degree centralities as the values.
    """

    # Initialise an empty dictionary
    dc = {}

    # Iterate through each node in the graph
    for node in G.vs():
        # Find the degree of the node
        d = G.degree(node)
        # Normalise by the maximum possible degree (n-1) and add to dictionary
        dc[node["label"]] = d / (G.vcount() - 1)

    return dc


def calc_next_iter(G, uncovered, N, dc, eg_dict, rB, centres, p, failed):
    """
    Finds the top N nodes to check in the next iteration of the algorithm.

    Args:
        G (igraph.Graph): The network being analysed.
        uncovered (list): A list of nodes in the network which are uncovered at this stage.
        N (int): The number of nodes to check in the next iteration of the algorithm.
        dc (dict): Dictionary containing the degree centrality of each node which is yet to be checked as a centre.
        eg_dict (dict): Dictionary with keys as nodes and values as the list of nodes within a distance of rB from that node.
        rB (int): The radius rB to be checked.
        centres (list): List of nodes identified as centres.
        p (str): Name of the node chosen as the most recent p [1].
        failed (Bool): True if the previous iteration of the algorithm failed, and False otherwise.

    Returns:
        eg_dict (dict): Returns an updated version of eg_dict with the nodes to be checked for the next iteration.
    """

    # If the algorithm failed in the previous iteration, then remove the most recent node p from the dictionary.
    # This node was chosen as a centre in the last stage, and so it does not need to be considered again.
    if not failed:
        dc.pop(p)

    # Choose the top N nodes by degree centrality.
    top_N_dc = dict(sorted(dc.items(), key=itemgetter(1), reverse=True)[:N])

    # Initialise an empty updated version of eg_dict.
    new_eg_dict = {}

    # For each of the top N nodes, assign the list of nodes in the subgraph of radius rB centred around the node to the new dictionary.
    for node in G.vs():
        if node["label"] in top_N_dc:
            # If the subgraph has already been found then reference the old dictionary to prevent recalculating the subgraph.
            if node["label"] in eg_dict:
                new_eg_dict[node["label"]] = eg_dict[node["label"]]
            # If not, then find the graph and add it to the new dictionary.
            else:
                [neighbors] = G.neighborhood([node], order=rB)
                new_eg_dict[node["label"]] = neighbors  # Add the list of nodes to the dictionary.

    # Once this is done for all the relevant nodes, return the new updated dictionary.
    return new_eg_dict


def accelerated_MEMB(G, lB):
    """
    Implements the Maximal Excluded Mass Burning algorithm as according to [1], with accelerated performance.
    Rather than calculating the excluded mass for all nodes at each stage, calculate the excluded mass of the node with the next highest excluded mass in the previous stage.
    Then reject all nodes with excluded mass in the previous stage less than this value (excluded mass can only decrease between stages)
    Repeat until only one node is left, and this becomes the next p.
    Note: Only works for odd numbers of lB, otherwise it takes rB to be the floor of (lB-1)/2 which is the same as for the odd lB-1.

    Args:
        G (igraph.Graph): The network the algorithm is to be applied to.
        lB (int): The diameter of the boxes used to cover the network.

    Returns:
        centres (list): A list of nodes assigned to be centres under the MEMB algorithm.
    """

    number_EM_calcs = 0

    # If the diameter lB is less than or equal to 2, then the maximum radius is 0 and so every node is in its own box.
    if lB == 1 or lB == 2:
        return list([v["label"] for v in G.vs()]), 0

    # Each box can have diameter of up to lB, so the maximum radius is rB = (lB-1)/2.
    rB = int((lB - 1) / 2)

    # Initialise empty lists for the covered and centre nodes.
    centres = []
    covered = []

    # The labels of the uncovered nodes are 0, ..., n-1 where n is the order of the network.
    uncovered = [i for i in range(G.vcount())]

    # Initialise an empty dictionary for the excluded mass of each node.
    excluded_mass = {}

    # For each node, calculate the initial excluded mass by finding the number of neighbours within a radius rB.
    for node in G.vs():
        excluded_mass[node] = G.neighborhood_size(node, order=rB)
        number_EM_calcs += 1

    # Rank the non-centre nodes in order from highest excluded mass to lowest.
    ordered_list = sorted(excluded_mass.items(), key=lambda item: item[1], reverse=True)

    # On the initial iteration, set maiden to True.
    maiden = True

    # The first value of p is that in the 1st position in the ordered list.
    p_value = ordered_list[0]

    # Iterate while there are still uncovered nodes in the network.
    while len(uncovered) > 0:

        # If it is not the first iteration, then find the values for the next iteration.
        if not maiden:
            # Find the next p using the method described above.
            ordered_list, p_value, number_EM_calcs = find_next_centre(G, covered, rB, ordered_list, number_EM_calcs)
            # Reorder the list according to decreasing excluded mass.
            # Note at this stage, that some of the nodes have their excluded mass from different stages of the iterative process.
            # This does not affect the outcome.
            ordered_list = sorted(ordered_list, key=lambda x: x[1], reverse=True).copy()
        else:
            # If this is the first iteration then reset the maiden variable.
            maiden = False

        # Take the label of the node to be the next p.
        p = p_value[0].index
        # Add p to the list of centres.
        centres.append(p)
        # Remove p from the list of non-centres.
        ordered_list.remove(p_value)

        # Find the neighbours of the new centre p within a radius rB.
        neighbours = G.neighborhood(p, order=rB)
        # Cover each of the nodes in this radius, and remove them from the list of uncovered nodes.
        for neighbour in neighbours:
            if neighbour in uncovered:
                covered.append(neighbour)
                uncovered.remove(neighbour)

    # Return the list of centres found.
    return centres, number_EM_calcs


def find_next_centre(G, covered, rB, ordered_list, number_EM_calcs):
    """
    Find the node with the next highest excluded mass to be the next centre.

    Args:
        G (igraph.Graph): The network to be analysed.
        covered (list): A list of indices for the covered nodes in the network at the current stage.
        rB (int): The radius of boxes to be found.
        ordered_list (list): A list of tuples containing the nodes of the network and their most recently updated excluded mass, in decreasing order of excluded mass.

    Returns:
        ordered_list (list): An updated version of the ordered list.
        p (tuple): The new centre value and its excluded mass.
    """

    # Create a copy of the ordered list in its current state.
    working_list = ordered_list.copy()

    # Start from the first node in the list.
    index = 0

    # Initialise an empty variable to store the maximum excluded mass.
    max_excluded_mass = 0

    # The working list stores all options for the next value of p.
    # Iterate whilst there are still multiple options for the next centre p.
    while len(working_list) > 1:

        # Take the next available node in the list.
        node = working_list[index]

        # Update the excluded mass for this node with the new covered nodes.
        new_excluded_mass = update_excluded_mass(G, covered, node[0], rB)
        number_EM_calcs += 1

        # If this is the new maximum then update the values.
        if new_excluded_mass > max_excluded_mass:
            max_excluded_mass = new_excluded_mass

            # Remove all nodes from the list of potential centres which have an excluded mass less than the current maximum
            working_list = list(filter(lambda x: x[1] > max_excluded_mass, working_list)).copy()

            # Add the current node back to the list if it has been removed.
            if node[1] == new_excluded_mass:
                working_list.append(node)

            # Update the excluded mass for the node in the ordered list.
            i = ordered_list.index(node)
            ordered_list[i] = (node[0], new_excluded_mass)

            # Update the excluded mass for the node in the working list.
            j = working_list.index(node)
            working_list[j] = (node[0], new_excluded_mass)

            # Look for the next node in the ordered list.
            index = j + 1

        # If the node is not the new maximum, i.e. it is less than some other value, then remove it from the working list.
        else:
            # Find the node in the list and remove it.
            i = ordered_list.index(node)
            ordered_list[i] = (node[0], new_excluded_mass)
            working_list.pop(index)

    # The final remaining node is the new centre p.
    p = working_list[0]

    # Return the new centre and the ordered list of nodes.
    return ordered_list, p, number_EM_calcs


def update_excluded_mass(G, covered, node, rB):
    """
    Update the excluded mass for the given node with the current list of covered nodes.

    Args:
        G (igraph.Graph): The network to be analysed.
        covered (list): A list of the nodes which are currently covered in the network.
        node (igraph.Vertex): The node whose excluded mass is to be updated.
        rB (int): The radius of the boxes for the box covering.

    Returns:
        excluded_mass (int): The updated excluded mass for the given node.
    """
    # Find all neighbours of the node within radius rB.
    neighbourhood = G.neighborhood(node, order=rB)
    # Find the number of those which are not yet covered.
    excluded_mass = len(set(neighbourhood) - set(covered))
    # Return the updated excluded mass.
    return excluded_mass


def distance_based_MEMB_all_lB(G, lB_min=3):
    """
    Calculates the centres for each lB from lB_min to the diameter of the network by first calculating the shortest path distance between every pair of nodes in the network.

    Args:
        G (igraph.Graph): The network to be analysed.
        lB_min (int) (opt): The minimum diameter of the boxes for the box covering.

    Returns:
        lB (list): A list of all box diameters lB.
        NB (list): A list of all number of boxes NB.
    """

    # Calculate the matrix of shortest paths between each pair of nodes in the network, and convert it to a numpy array.
    distance_matrix = G.distances()
    distance_np_array = np.array(distance_matrix)

    # Calculate the diameter and order of the network.
    diam = G.diameter()
    N = G.vcount()

    # Initialise an empty array for the number of boxes in the box covering for each lB.
    NB = []

    # Find the nearest odd number to the diameter of the network.
    nearest_odd = int(np.ceil(diam) // 2 * 2 + 1)

    # Generate a list of odd box diameters from the minimum value to two more than the diameter.
    lB = [i for i in range(lB_min, nearest_odd + 2, 2)]

    # Iterate through each possible lB.
    for l in lB:
        # Initialise an empty list to store the centres.
        centres = []

        # Find the maximum radius of the boxes.
        rB = int((l - 1) / 2)

        # Create a local copy of the array of shortest path distances.
        lB_dist = distance_np_array.copy()

        # The following lines of code convert the distance matrix into a matrix with ones if the nodes are within a radius of rB, and zeroes otherwise.
        # Remove any pairs of nodes which have a shortest path distance longer than the radius.
        lB_dist[lB_dist > rB] = 0
        # Reset the remaining values to 1.
        lB_dist[lB_dist > 0] = 1
        # Fill the diagonal with 1s.
        np.fill_diagonal(lB_dist, 1)

        # Initialise an array for the uncovered nodes. A 1 in the ith position means that the ith node is uncovered.
        # Begin with all nodes uncovered.
        uncovered = np.array([[1] * N])

        # Whilst there are still uncovered nodes in the network.
        while sum(sum(uncovered)) > 0:
            # Find the excluded masses by multiplying the uncovered array with the normalised matrix of distances.
            # The result is an array where the ith element is the excluded mass of the ith node.
            excluded_masses = np.matmul(uncovered, lB_dist)[0]

            # Assign p to be the node with the greatest excluded mass.
            p = np.argmax(excluded_masses)
            max_excluded_mass = excluded_masses[p]

            # Find the new nodes which are now covered and were not covered in the previous iteration.
            new_covered = np.array([x and y for x, y in zip(lB_dist[p], uncovered[0])])
            # Find the new array of uncovered nodes.
            uncovered = uncovered - new_covered
            # Add p to the list of centres.
            centres.append(p)

        # Add the number of boxes NB to the list.
        NB.append(len(centres))

    # Return the list of diameters lB and number of boxes NB.
    return lB, NB


def distance_based_MEMB(G, lB):
    """
    Calculates the centres for each lB from lB_min to the diameter of the network by first calculating the shortest path distance between every pair of nodes in the network.

    Args:
        G (igraph.Graph): The network to be analysed.
        lB_min (int) (opt): The minimum diameter of the boxes for the box covering.

    Returns:
        centres (list): A list of centres found by the algorithm.
    """

    # Calculate the matrix of shortest paths between each pair of nodes in the network, and convert it to a numpy array.
    distance_matrix = G.distances()
    lB_dist = np.array(distance_matrix)

    # Initialise an empty list to store the centres.
    centres = []

    # Find the maximum radius of the boxes.
    rB = int((lB - 1) / 2)

    # The following lines of code convert the distance matrix into a matrix with ones if the nodes are within a radius of rB, and zeroes otherwise.
    # Remove any pairs of nodes which have a shortest path distance longer than the radius.
    lB_dist[lB_dist > rB] = 0
    # Reset the remaining values to 1.
    lB_dist[lB_dist > 0] = 1
    # Fill the diagonal with 1s.
    np.fill_diagonal(lB_dist, 1)

    # Initialise an array for the uncovered nodes. A 1 in the ith position means that the ith node is uncovered.
    # Begin with all nodes uncovered.
    uncovered = np.array([[1] * G.vcount()])

    # Whilst there are still uncovered nodes in the network.
    while sum(sum(uncovered)) > 0:
        # Find the excluded masses by multiplying the uncovered array with the normalised matrix of distances.
        # The result is an array where the ith element is the excluded mass of the ith node.
        excluded_masses = np.matmul(uncovered, lB_dist)[0]

        # Assign p to be the node with the greatest excluded mass.
        p = np.argmax(excluded_masses)
        max_excluded_mass = excluded_masses[p]

        # Find the new nodes which are now covered and were not covered in the previous iteration.
        new_covered = np.array([x and y for x, y in zip(lB_dist[p], uncovered[0])])
        # Find the new array of uncovered nodes.
        uncovered = uncovered - new_covered
        # Add p to the list of centres.
        centres.append(str(p))

    # Return the list of diameters lB and number of boxes NB.
    return centres

