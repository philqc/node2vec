#! usr/bin/python

import os
import random
import pickle
import pandas as pd
import numpy as np
import networkx as nx
from utils import project_root, ARXIV_EDGE, ARXIV_REDUCED_EDGE, ARXIV_FEATURES
import logging
import csv
import learn_features
from sklearn import metrics, model_selection, pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO,
                                    datefmt="%Y-%m-%d %H:%M:%S")

file_edgelist = os.path.join(project_root(), "tests", "data", ARXIV_EDGE)
file_reduced_edgelist = os.path.join(project_root(), "tests", "data", ARXIV_REDUCED_EDGE)
features_pkl =  os.path.join(project_root(), "tests", "data", ARXIV_FEATURES)

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

def sample_negtive(g, num_sample):
    n_samples = []
    start_nodes = list(g.nodes(data=False))
    end_nodes = list(g.nodes(data=False))
    all_edges = g.edges()
    count = 0
    while True:
        u = random.choice(start_nodes)
        v = random.choice(end_nodes)
        if u!=v and ((u,v) not in all_edges):
            n_samples.append((u,v,0))
            count+=1
            if count%1000 == 0:
                print('%s negative samples  have been generated.'%(count))
            if count  == num_sample:
                return n_samples 


def create_features(samples, dic_emb, binary_operator='l2'):
    f = np.zeros((len(samples), 128))
    l = np.zeros(len(samples))
    idx = 0
    for s in samples:
        f1 = dic_emb[s[0]]
        f2 = dic_emb[s[1]]
        f_l2 = f1 * f2
        f[idx] = f_l2
        l[idx] = s[2]
        idx+=1

    return f, l

def evaluation(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
    scaler = StandardScaler()
    lin_clf = LogisticRegression(C=1)
    clf = pipeline.make_pipeline(scaler, lin_clf)
    clf.fit(X_train, y_train)
    auc = metrics.scorer.roc_auc_scorer(clf, X_test, y_test)

    """
    if clf.classes_[0] == 1: # only needs probabilities of positive class
        auc = roc_auc_score(y_test, y_score[:, 0])
    else:
        auc = roc_auc_score(y_test, y_score[:, 1])
    """
    print("Link Prediction AUC SCORE:%s"%(auc))
    

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
    
    learn_features.main()
    
    #rm_edges = [('84424','47999',1),('84424','66200',1), ('84424','47999',1),('84424','66200',1)]
    #create positive and negative samples
    positive_samples = rm_edges
    negative_samples = sample_negtive(g, len(rm_edges))
    all_sample = positive_samples + negative_samples
    #create feature and label matrix with custom binary operator
    with open(features_pkl, 'rb') as f:
        n2v_dic = pickle.load(f)
    #evaluation through AUC score
    features, labels = create_features(all_sample, n2v_dic)
 
    evaluation(features, labels)

if __name__ == "__main__":
    main()


