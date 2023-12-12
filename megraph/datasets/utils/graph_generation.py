# modified based on https://github.com/lukecavabarrett/pna/blob/master/multitask_benchmark/datasets_generation/graph_generation.py

import math
import random
from enum import Enum

import matplotlib.pyplot as plt  # only required to plot
import networkx as nx
import numpy as np
from scipy import spatial

"""
    Generates random graphs of different types of a given size.
    Some of the graph are created using the NetworkX library, for more info see
    https://networkx.github.io/documentation/networkx-1.10/reference/generators.html
"""


class GraphType(Enum):
    RANDOM = 0
    ERDOS_RENYI = 1
    BARABASI_ALBERT = 2
    GRID = 3
    CAVEMAN = 5
    TREE = 6
    LADDER = 7
    LINE = 8
    STAR = 9
    CATERPILLAR = 10
    LOBSTER = 11


# probabilities of each type in case of random type
MIXTURE = [
    (GraphType.ERDOS_RENYI, 0.2),
    (GraphType.BARABASI_ALBERT, 0.2),
    (GraphType.GRID, 0.05),
    (GraphType.CAVEMAN, 0.05),
    (GraphType.TREE, 0.15),
    (GraphType.LADDER, 0.05),
    (GraphType.LINE, 0.05),
    (GraphType.STAR, 0.05),
    (GraphType.CATERPILLAR, 0.1),
    (GraphType.LOBSTER, 0.1),
]


def erdos_renyi(N, degree=None, seed=None):
    """Creates an Erdős-Rényi or binomial graph of size N with degree/N probability of edge creation"""
    p = random.random() if degree is None else degree / N
    return nx.fast_gnp_random_graph(N, p, seed, directed=False)


def barabasi_albert(N, degree=None, seed=None):
    """Creates a random graph according to the Barabási-Albert preferential attachment model
    of size N and where nodes are atteched with degree edges"""
    if degree is None:
        # degree = int(random.random() * (N - 1)) + 1
        degree = np.random.randint(1, N)
    return nx.barabasi_albert_graph(N, degree, seed)


def grid_graph(n, m):
    """Creates a grid graph of size n x m"""
    total = n * m
    g = nx.empty_graph(total)
    edges = []
    j = 0
    for i in range(total):
        j = j + 1
        if j == m:
            j = 0
        if j != 0:
            edges.append((i, i + 1))
        if i + m < total:
            edges.append((i, i + m))
    g.add_edges_from(edges)
    return g


def grid(N):
    """Creates a m x k 2d grid graph with N = m*k and m and k as close as possible"""
    m = 1
    for i in range(1, int(math.sqrt(N)) + 1):
        if N % i == 0:
            m = i
    # return nx.grid_2d_graph(m, N // m)
    # change to manual edges
    return grid_graph(m, N // m)


def caveman(N):
    """Creates a caveman graph of m cliques of size k, with m and k as close as possible"""
    m = 1
    for i in range(1, int(math.sqrt(N)) + 1):
        if N % i == 0:
            m = i
    return nx.caveman_graph(m, N // m)


def tree(N, seed=None):
    """Creates a tree of size N with a power law degree distribution"""
    return nx.random_powerlaw_tree(N, seed=seed, tries=10000)


def ladder(N):
    """Creates a ladder graph of N nodes: two rows of N/2 nodes, with each pair connected by a single edge.
    In case N is odd another node is attached to the first one."""
    G = nx.ladder_graph(N // 2)
    if N % 2 != 0:
        G.add_node(N - 1)
        G.add_edge(0, N - 1)
    return G


def line(N):
    """Creates a graph composed of N nodes in a line"""
    return nx.path_graph(N)


def star(N):
    """Creates a graph composed by one center node connected N-1 outer nodes"""
    return nx.star_graph(N - 1)


def caterpillar(N, seed=None):
    """Creates a random caterpillar graph with a backbone of size b (drawn from U[1, N)), and N − b
    pendent vertices uniformly connected to the backbone."""
    if seed is not None:
        np.random.seed(seed)
    B = np.random.randint(low=1, high=N)
    G = nx.empty_graph(N)
    for i in range(1, B):
        G.add_edge(i - 1, i)
    for i in range(B, N):
        G.add_edge(i, np.random.randint(B))
    return G


def lobster(N, seed=None):
    """Creates a random Lobster graph with a backbone of size b (drawn from U[1, N)), and p (drawn
    from U[1, N - b ]) pendent vertices uniformly connected to the backbone, and additional
    N - b - p pendent vertices uniformly connected to the previous pendent vertices"""
    if seed is not None:
        np.random.seed(seed)
    B = np.random.randint(low=1, high=N)
    F = np.random.randint(low=B + 1, high=N + 1)
    G = nx.empty_graph(N)
    for i in range(1, B):
        G.add_edge(i - 1, i)
    for i in range(B, F):
        G.add_edge(i, np.random.randint(B))
    for i in range(F, N):
        G.add_edge(i, np.random.randint(low=B, high=F))
    return G


def randomize(A):
    """Adds some randomness by toggling some edges without changing the expected number of edges of the graph"""
    BASE_P = 0.9

    # e is the number of edges, r the number of missing edges
    N = A.shape[0]
    e = np.sum(A) / 2
    r = N * (N - 1) / 2 - e

    # ep chance of an existing edge to remain, rp chance of another edge to appear
    if e <= r:
        ep = BASE_P
        rp = (1 - BASE_P) * e / r
    else:
        ep = BASE_P + (1 - BASE_P) * (e - r) / e
        rp = 1 - BASE_P

    array = np.random.uniform(size=(N, N), low=0.0, high=0.5)
    array = array + array.transpose()
    remaining = np.multiply(np.where(array < ep, 1, 0), A)
    appearing = np.multiply(
        np.multiply(np.where(array < rp, 1, 0), 1 - A), 1 - np.eye(N)
    )
    ans = np.add(remaining, appearing)

    # assert (np.all(np.multiply(ans, np.eye(N)) == np.zeros((N, N))))
    # assert (np.all(ans >= 0))
    # assert (np.all(ans <= 1))
    # assert (np.all(ans == ans.transpose()))
    return ans


def generate_graph(N, type=GraphType.RANDOM, seed=None, degree=None):
    """
    Generates random graphs of different types of a given size. Note:
     - graph are undirected and without weights on edges
     - node values are sampled independently from U[0,1]
    :param N:       number of nodes
    :param type:    type chosen between the categories specified in GraphType enum
    :param seed:    random seed
    :param degree:  average degree of a node, only used in some graph types
    :return:        adj_matrix: N*N numpy matrix
                    node_values: numpy array of size N
    """
    random.seed(seed)
    np.random.seed(seed)

    # sample which random type to use
    if type == GraphType.RANDOM:
        type = np.random.choice(
            [t for (t, _) in MIXTURE], 1, p=[pr for (_, pr) in MIXTURE]
        )[0]

    # generate the graph structure depending on the type
    if type == GraphType.ERDOS_RENYI:
        # if degree == None:
        #     degree = random.random() * N
        G = erdos_renyi(N, degree, seed)
    elif type == GraphType.BARABASI_ALBERT:
        # if degree == None:
        #     degree = int(random.random() * (N - 1)) + 1
        G = barabasi_albert(N, degree, seed)
    elif type == GraphType.GRID:
        G = grid(N)
    elif type == GraphType.CAVEMAN:
        G = caveman(N)
    elif type == GraphType.TREE:
        G = tree(N, seed)
    elif type == GraphType.LADDER:
        G = ladder(N)
    elif type == GraphType.LINE:
        G = line(N)
    elif type == GraphType.STAR:
        G = star(N)
    elif type == GraphType.CATERPILLAR:
        G = caterpillar(N, seed)
    elif type == GraphType.LOBSTER:
        G = lobster(N, seed)
    else:
        print("Type not defined")
        return

    # generate adjacency matrix and nodes values
    nodes = list(G)
    random.shuffle(nodes)
    adj_matrix = nx.to_numpy_array(G, nodes)
    node_values = np.random.uniform(low=0, high=1, size=N)

    # randomization
    adj_matrix = randomize(adj_matrix)

    # draw the graph created
    # nx.draw(G, pos=nx.spring_layout(G))
    # plt.draw()

    return adj_matrix, node_values, type


# generate_graph_geo from https://github.com/deepmind/graph_nets
DISTANCE_WEIGHT_NAME = "distance"


# default theta change from 1000.0 to 200.0
def generate_graph_geo(num_nodes, dimensions=2, theta=200.0, rate=1.0):
    """Creates a connected graph.

    The graphs are geographic threshold graphs, but with added edges via a
    minimum spanning tree algorithm, to ensure all nodes are connected.

    Args:
      num_nodes: number of nodes per graph.
      dimensions: (optional) An `int` number of dimensions for the positions.
        Default= 2.
      theta: (optional) A `float` threshold parameters for the geographic
        threshold graph's threshold. Large values (1000+) make mostly trees. Try
        20-60 for good non-trees. Default=1000.0.
      rate: (optional) A rate parameter for the node weight exponential sampling
        distribution. Default= 1.0.

    Returns:
      The graph.
    """
    # Create geographic threshold graph.
    pos_array = np.random.uniform(size=(num_nodes, dimensions))
    pos = dict(enumerate(pos_array))
    weight = dict(enumerate(np.random.exponential(rate, size=num_nodes)))
    geo_graph = nx.geographical_threshold_graph(
        num_nodes, theta, pos=pos, weight=weight
    )

    # Create minimum spanning tree across geo_graph's nodes.
    distances = spatial.distance.squareform(spatial.distance.pdist(pos_array))
    i_, j_ = np.meshgrid(range(num_nodes), range(num_nodes), indexing="ij")
    weighted_edges = list(zip(i_.ravel(), j_.ravel(), distances.ravel()))
    mst_graph = nx.Graph()
    mst_graph.add_weighted_edges_from(weighted_edges, weight=DISTANCE_WEIGHT_NAME)
    mst_graph = nx.minimum_spanning_tree(mst_graph, weight=DISTANCE_WEIGHT_NAME)
    # Put geo_graph's node attributes into the mst_graph.
    for i in mst_graph.nodes():
        mst_graph.nodes[i].update(geo_graph.nodes[i])

    # Compose the graphs.
    combined_graph = nx.compose_all((mst_graph, geo_graph.copy()))
    # Put all distance weights into edge attributes.
    for i, j in combined_graph.edges():
        combined_graph.get_edge_data(i, j).setdefault(
            DISTANCE_WEIGHT_NAME, distances[i, j]
        )
    return combined_graph, mst_graph, geo_graph


def generate_graph_sbm(num_nodes, min_block=5, max_block=15):
    connected = False
    while not connected:
        # generate random blocks
        remains = num_nodes
        lower_bound = min_block
        block_sizes = []
        while True:
            a = random.randint(lower_bound, max_block)
            if remains - a < lower_bound:
                break
            block_sizes.append(a)
            remains -= a
        block_sizes.append(remains)
        assert np.sum(block_sizes) == num_nodes
        # generate random prob
        num_blocks = len(block_sizes)
        intra_block_probs = np.random.rand(num_blocks) * 0.2 + 0.3
        inter_block_probs = np.random.rand(num_blocks, num_blocks) * 0.005 + 0.0005
        inter_block_probs = (inter_block_probs + inter_block_probs.T) / 2
        eye = np.eye(num_blocks)
        block_probs = intra_block_probs * eye + inter_block_probs * (1 - eye)
        # generate graph
        graph = nx.stochastic_block_model(block_sizes, block_probs, seed=2022)
        connected = nx.is_connected(graph)
    return graph
