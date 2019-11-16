#! usr<F2>bin/python

import os
import pickle
import pandas as pd
import numpy as np
import networkx as nx
from utils import project_root, ARXIV_EDGE, ARXIV_REDUCED_EDGE
import logging
import csv

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO,
                                    datefmt="%Y-%m-%d %H:%M:%S")

file_edgelist = os.path.join(project_root(), "tests", "data", ARXIV_EDGE)
file_reduced_edgelist = os.path.join(project_root(), "tests", "data", ARXIV_REDUCED_EDGE)


def create_graph(edge_csv):
    g  =nx.read_edgelist(edge_csv, delimiter=',', create_using=nx.DiGraph(), encoding='utf-8-sig')
    for edge in g.edges():
        g[edge[0]][edge[1]]['weight'] = 1
    g = g.to_undirected()
    return g

def remove_edges(g, min_edges, portion=0.5):
    print(nx.is_connected(g))
    #print(min_edges)
    all_edges = list(g.edges())
    G = nx.Graph()
    G.add_edges_from(min_edges)
    print(nx.info(G))
    num_all_edges = len(all_edges)
    num_rm_edges = int((g.number_of_edges()-len(min_edges))*portion)
    np.random.shuffle(all_edges)
    logging.info("Total Edges, Number of Remove Edges: %s , %s " %(num_all_edges, num_rm_edges))
    count = 0
    removed_edges = []
    remain_edges = all_edges
    for edge_uv in all_edges:
        #print(edge_uv)
        edge_vu = (edge_uv[1], edge_uv[0], {'weight':1})
        edge_uv = (edge_uv[0], edge_uv[1], {'weight':1})
        if (edge_uv not in min_edges) and (edge_vu not in min_edges):
            removed_edges.append((edge_uv[0], edge_uv[1], 1))
            remain_edges.remove((edge_uv[0], edge_uv[1]))
            count+=1
            if count%1000 == 0:
                print('%s edges have been removed.'%(count))
            if count  == num_rm_edges:
                return remain_edges, removed_edges

def min_spanning_edges(g):
    return list(nx.minimum_spanning_edges(g))

def sample_positive():
    pass

def sample_negtive(g, num_sample):
    n_samples = []
    start_nodes = g.nodes(data=False)
    end_nodes = g.nodes(data=False)
    print(start_nodes)
    return n_samples
:
def main():
    #create graph
    g = create_graph(file_edgelist)
    
    #if graph is not connected, there will be error in node2vec
    if nx.is_connected(g):
        print("Graph is connected!!")
    else:
        print("Graph is not connected, Choose the largest connected component.")
        Gcc = sorted(nx.connected_components(g), key=len, reverse=True)
        g = g.subgraph(Gcc[0])
        

    print("Graph is connected: %s"%(nx.is_connected(g)))
    print(nx.info(g))

    """
    #reduce the graph
    re_edges, rm_edges = remove_edges(g, min_spanning_edges(g))
    logging.info("Remain Edges, Remove Edges(positive samples): %s , %s " %(len(re_edges), len(rm_edges)))
    
    #convert reduced graph to csv
    logging.info("Create New EdgeList for Arxiv_Reduced.")
    start_nodes = [x[0] for x in re_edges]
    end_nodes = [x[1] for x in re_edges]

    print(len(set(start_nodes)|set(end_nodes)))
    
    df = pd.DataFrame({'start':start_nodes, 'end':end_nodes})
    df.to_csv(file_reduced_edgelist, header=False, index=False)
    """
    #create positive and negative samples
    positive_samples = 5000
    negative_sample = sample_negtive(g, 5000)
    all_sample = positive_samples + negative_samples
    #create feature and label matrix with custom binary operator
    
    #evaluation through AUC score
    

if __name__ == "__main__":
    main()


