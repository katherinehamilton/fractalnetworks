"""Functions to generate (u,v)-flowers"""

# Network analysis modules
import networkx as nx


def generate_uv_flower(u, v, n, save=False):
    """
    Generates an n-th generation (u,v)-flower.

    Args:
        u (int)                      : Value of u, i.e. path length of one of the parallel paths.
        v (int)                      : Value of v, i.e. path length of one of the parallel paths.
        n (int)                      : Number of generations.
        save (:obj:`bool`, optional) : If True, the file is saved to the network-files folder. Default is False.

    Returns:
        (networkx.Graph) : The generated (u,v)-flower.
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
        G (networkx.Graph) : The (u,v)-flower network in its current (t-1)-th generation.
        u (int)            : Value of u, i.e. path length of one of the parallel paths.
        v (int)            : Value of v, i.e. path length of one of the parallel paths.

    Returns:
        (networkx.Graph) : The (u,v)-flower network in the t-th generation.

    """
    # Find a list of all the nodes and edges in the network at the (t-1)-th generation.
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
        v (int) : Value of v, i.e. path length of the parallel paths.
        n (int) : Number of generations.

    Returns:
        (networkx.Graph) : The generated (u,v)-flower.
    """
    # Returns the (u,v)-flower found by the fractal generator, but with u hardcoded as 1.
    return generate_uv_flower(1, v, n)


def add_new_path(l, N, edge):
    """
    Adds a new parallel path to the network, a step in the (u, v)-flower generation process.

    Args:
        l (int)      : The length of the path to be added to the network.
        N (int)      : A counter which stores the next unused integer to label nodes.
        edge (tuple) : The edge from the network being replaced with parallel paths.

    Returns:
        (tuple) : A tuple containing a networkx.Graph and int, specifically:
                    a path graph of length l with vertices labelled correctly;
                    and a counter which stores the next unused integer to label nodes.
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

    # Return the path graph to be merged with the (u, v)-flower graph,
    #   and the counter for the next unused integer for node labels.
    return Hl, N
