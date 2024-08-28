"""This module contains functions to generate instances of known fractal network models"""

# Song-Havlin-Makse Model and Hub Attraction Dynamical Growth Model
from .HADGmodel import *

# (u,v)-Flowers
from .uvflowermodel import *

# Nested Barabasi-Albert Model
from .nestedBAmodel import *


def generate_barabasi_albert_network(m, N, save=False):
    """
    Generates a Barabasi-Albert network with parameters m and N.

    Args:
        m (int) : The number of edges added to each newly added node.
        N (int) : The number of nodes in the network.
        save (bool) (opt) : If True, the file is saved to the network-files folder. Default is False.

    Returns:
        (networkx.Graph) : The generated Barabasi-Albert network.
    """
    # Initialise a cycle graph of length w = u + v
    G = nx.barabasi_albert_graph(N, m)

    # Save the file if save is True.
    if save:
        # Save the file in the format SHM-model-m-p-generationn-examplei.gml
        # For p != 0 the method is non-deterministic and so multiple examples need to be saved.
        count = 1
        # Initialise a Boolean variable to False, where False means that it is yet to be saved,
        # and True means it has been saved.
        saved = False

        # Iterate while the file is yet to be saved.
        while not saved:
            filename = "BA-model-" + str(m) + "-" + str(N) + "-example" + str(count) + ".gml"
            filepath = "network-files/models/barabasi-albert-model/" + filename
            # If the count-th example already exists, increment the count by 1 and try to save again.
            if not os.path.isfile(filepath):
                nx.write_gml(G, filepath)
                saved = True
            else:
                count += 1

    return G
