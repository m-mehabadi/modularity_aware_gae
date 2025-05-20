import numpy as np
import scipy.sparse as sp
import tensorflow as tf


flags = tf.app.flags
FLAGS = flags.FLAGS


"""
Disclaimer: functions defined from lines 18 to 54 in this file come from
the tkipf/gae original repository on Graph Autoencoders. Moreover, the
mask_test_edges function is borrowed from philipjackson's mask_test_edges 
pull request on this same repository.
"""


def sparse_to_tuple(sparse_mx):

    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()

    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape

    return coords, values, shape


def preprocess_graph(adj):

    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])

    degree_mat_inv_sqrt = sp.diags(np.power(np.array(adj_.sum(1)), -0.5).flatten())

    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt)

    return sparse_to_tuple(adj_normalized)


def construct_feed_dict(adj_normalized, adj_normalized_layer2, adj_orig, features, deg_matrix, placeholders):

    # Construct feed dictionary
    feed_dict = dict()
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['adj']: adj_normalized})
    feed_dict.update({placeholders['adj_layer2']: adj_normalized_layer2})
    feed_dict.update({placeholders['adj_orig']: adj_orig})

    if not FLAGS.simple:
        feed_dict.update({placeholders['degree_matrix']: deg_matrix})

    return feed_dict


def mask_test_edges(adj, test_percent, val_percent):
    """ Randomly removes some edges from original graph for validation/test sets

    :param adj: scipy.sparse.csr_matrix
        Original adjacency matrix
    :param test_percent: float
        Percentage of edges in test set
    :param val_percent: float
        Percentage of edges in validation set
    :return: tuple containing:
        - train adjacency matrix
        - validation edges
        - validation edges (negative)
        - test edges
        - test edges (negative)
    """

    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()

    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0

    # Convert to COO format for easier edge manipulation
    adj = adj.tocoo()
    edges = np.vstack((adj.row, adj.col)).T
    num_test = int(np.floor(edges.shape[0] * test_percent / 100.))
    num_val = int(np.floor(edges.shape[0] * val_percent / 100.))

    # Store edges in list and remove symmetric edges
    edge_tuples = list(map(tuple, edges))
    edges_all = edge_tuples + [(j, i) for (i, j) in edge_tuples]
    edges_all = list(set(edges_all))
    np.random.shuffle(edges_all)

    # Split edge set for training and testing
    test_edges = edges_all[:num_test]
    val_edges = edges_all[num_test:num_test+num_val]
    train_edges = edges_all[num_test+num_val:]
    
    # Create training adjacency matrix
    data = np.ones(len(train_edges))
    adj_train = sp.csr_matrix((data, zip(*train_edges)), shape=adj.shape)
    adj_train = adj_train + adj_train.T

    # Create validation/test sets of negative edges (non-edges)
    test_edges_false = []
    val_edges_false = []
    
    # Function to sample negative edges
    def sample_neg_edges(num_needed, existing_edges):
        neg_edges = set()
        while len(neg_edges) < num_needed:
            i = np.random.randint(0, adj.shape[0])
            j = np.random.randint(0, adj.shape[0])
            if i != j and (i, j) not in existing_edges and (j, i) not in existing_edges:
                neg_edges.add((i, j))
        return list(neg_edges)

    # Generate negative test edges
    test_edges_false = sample_neg_edges(num_test, set(edges_all))
    
    # Generate negative validation edges
    val_edges_false = sample_neg_edges(num_val, set(edges_all + test_edges_false))

    assert len(test_edges_false) == num_test
    assert len(val_edges_false) == num_val

    return adj_train, val_edges, val_edges_false, test_edges, test_edges_false


def preprocess_degree(adj, is_simple):

    """
    Preprocessing degree-based term for modularity loss
    :param adj: sparse adjacency matrix of the graph
    :param is_simple: "simple" boolean flag for modularity
    :return: degree-based term matrices
    """

    if is_simple:
        deg_matrix = None
        deg_matrix_init = None

    else:
        if FLAGS.verbose:
            print("Preprocessing on degree matrices")
        deg = np.sum(adj, 1)
        deg_matrix = (1.0 / np.sum(adj)) * deg.dot(np.transpose(deg))
        #deg_matrix = deg_matrix - np.diag(np.diag(deg_matrix))
        deg_matrix_init = sp.csr_matrix(deg_matrix)
        deg_matrix = sparse_to_tuple(deg_matrix_init)
        if FLAGS.verbose:
            print("Done! \n")

    return deg_matrix, deg_matrix_init


def introductory_message():

    """
    An introductory message to display when launching experiments
    """

    print("\n \n \n \n[MODULARITY-AWARE GRAPH AUTOENCODERS]\n \n \n \n")

    print("EXPERIMENTAL SETTING \n")

    print("- Graph dataset:", FLAGS.dataset)
    print("- Mode name:", FLAGS.model)
    print("- Number of models to train:", FLAGS.nb_run)
    print("- Number of training iterations for each model:", FLAGS.iterations)
    print("- Learning rate:", FLAGS.learning_rate)
    print("- Dropout rate:", FLAGS.dropout)
    print("- Use of node features in the input layer:", FLAGS.features)

    if FLAGS.model in ("gcn_ae", "gcn_vae"):
        print("- Dimension of the GCN hidden layer:", FLAGS.hidden)
    print("- Dimension of the output layer:", FLAGS.dimension)
    print("- lambda:", FLAGS.lamb)
    print("- beta:", FLAGS.beta)
    print("- gamma:", FLAGS.gamma)
    print("- s:", FLAGS.s_reg)

    if FLAGS.fastgae:
        print("- FastGAE: yes, with", FLAGS.measure, "sampling\n     - alpha:",
        FLAGS.alpha, "\n     - n_s:", FLAGS.nb_node_samples, "\n     - replacement:", FLAGS.replace, "\n")
    else:
        print("- FastGAE: no \n")

    print("Final embedding vectors will be evaluated on:")
    if FLAGS.task == 'task_1':
        print('- Task 1, i.e., pure community detection')
    else:
        print('- Task 2, i.e., joint community detection and link prediction')
    print("\n \n \n \n")
