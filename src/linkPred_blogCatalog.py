#! usr<F2>bin/python

import os
import pickle
import pandas as pd
import networkx as nx
from utils import project_root, BLOGCATALOG_FEATURES, BLOGCATALOG_EDGE
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO,
                                    datefmt="%Y-%m-%d %H:%M:%S")

file_feature =  os.path.join(project_root(), "tests", "data", BLOGCATALOG_FEATURES)
file_edgelist = os.path.join(project_root(), "tests", "data", BLOGCATALOG_EDGE)

def create_graph(edge_csv):
    g =nx.read_edgelist(edge_csv, delimiter=',')
    return g



            
g = create_graph(file_edgelist)
print(nx.info(g))
