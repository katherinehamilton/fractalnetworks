"""Implements the Random Sequential box covering algorithm"""

# Mathematics modules
import random


def random_sequential(G, lB):
    """
    Implements the random sequential box covering algorithm.

    Args:
        G (igraph.Graph) : The network to be analysed.
        lB (int)         : The diameter of the boxes.

    Returns:
        (tuple) : A tuple containing two lists, specifically:
                    a list of centre nodes found by the algorithm;
                    and a list of lists, where each sublist is a box under the box-covering algorithm.
    """
    # Find the radius of the boxes
    rB = int((lB - 1) / 2)

    # Initialise lists to store the centre and non-centre nodes
    centres = []
    non_centres = [i for i in range(G.vcount())]

    # Initialise lists to store the covered and uncovered nodes.
    covered = []
    uncovered = [i for i in range(G.vcount())]

    # Initialise an empty list to store the boxes
    boxes = []

    # Iterate while there are still uncovered nodes in the network.
    while len(covered) < G.vcount():
        # Choose a random centre.
        p = random.choice(non_centres)
        # Find the uncovered nodes within a radius of rB from this centre
        newly_covered = set(G.neighborhood([p], order=rB)[0]).intersection(set(uncovered))

        # If there are any new nodes which are now covered, then add the new centre and the new box.
        if len(newly_covered) > 0:
            boxes.append(list(newly_covered))
            centres.append(p)

            # Update the list of covered and uncovered nodes
            covered.extend(newly_covered)
            uncovered = list(set(uncovered).difference(set(newly_covered)))

        # Remove the node p from the list of non-centres.
        non_centres.remove(p)

    # Return the list of centres and the boxes.
    return centres, boxes
