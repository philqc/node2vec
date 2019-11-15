#! usr<F2>bin/python

import os
import pickle
import pandas as pd
import networkx as nx
from utils import project_root, ARXIV_EDGE
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO,
                                    datefmt="%Y-%m-%d %H:%M:%S")

file_edgelist = os.path.join(project_root(), "tests", "data", ARXIV_EDGE)

def create_graph(edge_csv):
    g =nx.read_edgelist(edge_csv, delimiter=',', create_using=nx.Graph())
    return g

def sample_positive():
    pass

def sample_negtive():
    pass

g = create_graph(file_edgelist)
print(nx.info(g))
