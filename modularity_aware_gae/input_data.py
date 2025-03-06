import networkx as nx
import numpy as np
import pickle as pkl
import scipy.sparse as sp
import sys


"""
Disclaimer: the parse_index_file function from this file, as well as the
cora/citeseer/pubmed parts of the loading functions, come from the
tkipf/gae original repository on Graph Autoencoders
"""


def parse_index_file(filename):

    index = []
    for line in open(filename):
        index.append(int(line.strip()))

    return index


def load_data(dataset, dataset_path=None):
    """
    Load datasets
    :param dataset: name of the input graph dataset (optional)
    :param dataset_path: path to graph data file (required if dataset is None)
    :return: n*n sparse adjacency matrix and n*f node features matrix
    """
    if dataset is None:
        if dataset_path is None:
            raise ValueError('Error: dataset_path must be provided when dataset is None')
        # Load custom dataset
        adj = nx.adjacency_matrix(nx.read_edgelist(dataset_path,
                                                   nodetype = str,
                                                   data = (('weight', int),),
                                                   delimiter = ' '))
        features = sp.identity(adj.shape[0])
        return adj, features

    # Use default path for predefined datasets
    base_path = "../data"

    if dataset == 'cora-large':
        adj = nx.adjacency_matrix(nx.read_edgelist(f"{base_path}/coralarge", delimiter = ' '))
        features = sp.identity(adj.shape[0])

    elif dataset == 'sbm':
        adj = nx.adjacency_matrix(nx.read_edgelist(f"{base_path}/sbm.txt"))
        features = sp.identity(adj.shape[0])

    elif dataset == 'blogs':
        adj = nx.adjacency_matrix(nx.read_edgelist(f"{base_path}/blogs",
                                                   nodetype = int,
                                                   data = (('weight', int),),
                                                   delimiter = ' '))
        features = sp.identity(adj.shape[0])
    
    elif dataset == 'IBM':
        print("Loading IBM AML dataset - 123")
        adj = nx.adjacency_matrix(nx.read_edgelist(f"{base_path}/IBM_AML/IBM-graph.edgelist",
                                                   nodetype = str,
                                                   data = (('weight', int),),
                                                   delimiter = ' '))
        features = sp.identity(adj.shape[0])

    elif dataset in ('cora', 'citeseer', 'pubmed'):
        # Load the data: x, tx, allx, graph
        names = ['x', 'tx', 'allx', 'graph']
        objects = []
        for i in range(len(names)):
            with open(f"{base_path}/ind.{dataset}.{names[i]}", 'rb') as f:
                if sys.version_info > (3, 0):
                    objects.append(pkl.load(f, encoding = 'latin1'))
                else:
                    objects.append(pkl.load(f))
        x, tx, allx, graph = tuple(objects)
        test_idx_reorder = parse_index_file(f"{base_path}/ind.{dataset}.test.index")
        test_idx_range = np.sort(test_idx_reorder)
        if dataset == 'citeseer':
            # Fix citeseer dataset (there are some isolated nodes in the graph)
            # Find isolated nodes, add them as zero-vecs into the right position
            test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
            tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
            tx_extended[test_idx_range - min(test_idx_range), :] = tx
            tx = tx_extended
        features = sp.vstack((allx, tx)).tolil()
        features[test_idx_reorder, :] = features[test_idx_range, :]
        graph = nx.from_dict_of_lists(graph)
        adj = nx.adjacency_matrix(graph)

    else:
        raise ValueError('Error: undefined dataset!')

    return adj, features


def load_labels(dataset, labelset_path=None):
    """
    Load node-level labels
    :param dataset: name of the input graph dataset (optional)
    :param labelset_path: path to labels file (required if dataset is None)
    :return: n-dim array of node labels, used for community detection
    """
    if dataset is None:
        if labelset_path is None:
            raise ValueError('Error: labelset_path must be provided when dataset is None')
        # Load custom labels
        labels = np.loadtxt(labelset_path, delimiter = ' ', dtype = str)
        return labels

    # Use default path for predefined datasets
    base_path = "../data"

    if dataset == 'cora-large':
        labels = np.loadtxt(f"{base_path}/coralarge-cluster", delimiter = ' ', dtype = str)

    elif dataset == 'sbm':
        labels = np.repeat(range(100), 1000)

    elif dataset == 'blogs':
        labels = np.loadtxt(f"{base_path}/blogs-cluster", delimiter = ' ', dtype = str)
    
    elif dataset == 'IBM':
        print("Loading IBM AML dataset - 456")
        labels = np.loadtxt(f"{base_path}/IBM_AML/IBM-labels.csv", delimiter = ' ', dtype = str)

    elif dataset in ('cora', 'citeseer', 'pubmed'):
        names = ['ty', 'ally']
        objects = []
        for i in range(len(names)):
            with open(f"{base_path}/ind.{dataset}.{names[i]}", 'rb') as f:
                if sys.version_info > (3, 0):
                    objects.append(pkl.load(f, encoding = 'latin1'))
                else:
                    objects.append(pkl.load(f))
        ty, ally = tuple(objects)
        test_idx_reorder = parse_index_file(f"{base_path}/ind.{dataset}.test.index")
        test_idx_range = np.sort(test_idx_reorder)
        if dataset == 'citeseer':
            # Fix citeseer dataset (there are some isolated nodes in the graph)
            # Find isolated nodes, add them as zero-vecs into the right position
            test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
            ty_extended = np.zeros((len(test_idx_range_full), ty.shape[1]))
            ty_extended[test_idx_range - min(test_idx_range), :] = ty
            ty = ty_extended
        labels = sp.vstack((ally, ty)).tolil()
        labels[test_idx_reorder, :] = labels[test_idx_range, :]
        # One-hot to integers
        labels = np.argmax(labels.toarray(), axis = 1)

    else:
        raise ValueError('Error: undefined dataset!')

    return labels