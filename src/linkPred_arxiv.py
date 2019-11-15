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
    
    removed_edges = []
    for edge_uv in all_edges:
        edge_vu = (edge_uv[1], edge_uv[2])
        if (edge_uv not in all_edges) and (edge_vu not in all_edges):
            removed_edges.append((edge_uv[0], edge_uv[1], 1))
            g.remove_edge(edge_uv[0], edge_uv[1])
        if len(removed_edges) == num_rm_edges:
            return g,  removed_edges

def min_spanning_edges(g):
    return list(nx.minimum_spanning_edges(g))

def sample_positive():
    pass

def sample_negtive():
    pass

def main():
    #create graph
    g = create_graph(file_edgelist)
    print(nx.info(g))

    #reduce the graph

    #convert reduced graph to csv

    #create positive and negative samples

    #run node2vec to get embedding

    #create feature and label matrix with custom binary operator

    #evaluation through AUC score
    

if __name__ == "__main__":
    main()


