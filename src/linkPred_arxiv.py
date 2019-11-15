#! usr<F2>bin/python

import os
import pickle
import pandas as pd
import numpy as np
import networkx as nx
from utils import project_root, ARXIV_EDGE
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO,
                                    datefmt="%Y-%m-%d %H:%M:%S")

file_edgelist = os.path.join(project_root(), "tests", "data", ARXIV_EDGE)

def create_graph(edge_csv):
    g  =nx.read_edgelist(edge_csv, delimiter=',', create_using=nx.DiGraph(), encoding='utf-8-sig')
    for edge in g.edges():
        g[edge[0]][edge[1]]['weight'] = 1
    g = g.to_undirected()
    return g

def remove_edges(g, min_edges, portion=0.5):
    
    all_edges = g.edges()
    num_rm_edges = int((g.number_of_edges()-len(min_edges))*portion)
    np.random.shuffle(all_edges)


def min_spanning_edges(g):
    return list(nx.minimum_spanning_edges(g))

def sample_positive():
    pass

def sample_negtive():
    pass


g = create_graph(file_edgelist)
print(nx.is_connected(g))
