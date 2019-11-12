#! usr/bin/python

import os
import pickle
import logging
import numpy as np
import pandas as pd
from utils import project_root, BLOGCATALOG_LABELS, BLOGCATALOG_FEATURES
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO,
                            datefmt="%Y-%m-%d %H:%M:%S")
file_feature =  os.path.join(project_root(), "tests", "data", BLOGCATALOG_FEATURES)
file_label = os.path.join(project_root(), "tests", "data", BLOGCATALOG_LABELS)
NODE = 'node'
GROUP = 'group'

def sparse2array_inv_binarize(y):
    mlb = MultiLabelBinarizer(range(y.shape[1]))
    mlb.fit(y)
    y_ = mlb.inverse_transform(y)
    return y_

def compute_metrics(y_test, preds):    
    mlb = MultiLabelBinarizer(range(y_test.shape[1]))
    mlb.fit(preds)
    preds = mlb.transform(preds)
    #convert y_test from sparse back and forth to get binarized version.
    #There's probably a better way to do this
    y_test = sparse2array_inv_binarize(y_test)
    y_test = mlb.transform(y_test)                             
    microF1 = f1_score(y_test, preds, average='micro')
    macroF1 = f1_score(y_test, preds, average='macro')
 
    return microF1, macroF1

#ture features into numpy matrix 
def create_features(features_pkl):
    with open(features_pkl, 'rb') as f:
            n2v_dic = pickle.load(f)
    nodes = n2v_dic.keys()
    num_nodes, num_features = len(nodes), len(n2v_dic['1'])
    logging.info("Create Feature Matrix in Numpy: %s nodes, %s features" %(num_nodes, num_features))
    features = np.zeros((num_nodes, num_features))
    for node in nodes:
        idx = int(node)-1
        features[idx] = n2v_dic[node]
    logging.info("Feature Matrix Created.")
    return features

def create_labels(label_csv):
    label_df = pd.read_csv(label_csv, header=None, names=[NODE, GROUP])
    num_groups = len(label_df[GROUP].unique())
    num_nodes = len(label_df[NODE].unique())
    logging.info("Create Label Matrix in Numpy: %s nodes, %s groups" %(num_nodes, num_groups))
    labels = np.zeros((num_nodes, num_groups))
    for index, row in label_df.iterrows():
        node = row[NODE]
        group = row[GROUP]
        labels[node-1, group-1] = 1
    logging.info("Label Matrix Created.")
    return labels

def kFold_average(features, labels, clf, f1='macro', k=10):
    test_result = []
    kf = KFold(n_splits=k, shuffle=True)
    for train_index, test_index in kf.split(features):
        # print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = features[train_index], features[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        clf = clf.fit(X_train, y_train)
        #code (from https://github.com/apoorvavinod/node2vec/blob/master/src/Classifier.py)
        y_test_ = sparse2array_inv_binarize(y_test)
        num_predictions = [len(item) for item in y_test_]
        probabilities = clf.predict_proba(X_test)
        sorted_indices_probs = probabilities.argsort()
        y_pred = [sorted_indices[-num:].tolist() for (sorted_indices,num) in zip(sorted_indices_probs,num_predictions)]
        mi, ma = compute_metrics(y_test, y_pred)
        logging.info("Macro F1: %s"%ma)
        logging.info("Micro F1: %s"%mi)
        test_result.append(ma)
    return sum(test_result)/len(test_result) 


def main():
    features = create_features(file_feature)
    labels = create_labels(file_label)
    clf = OneVsRestClassifier(LogisticRegression(multi_class='ovr',solver='lbfgs'))
    k = 2
    kfold_avg = kFold_average(features, labels, clf, k=k)
    logging.info("%s Fold CV Macro F1 Score: %s " %(k, kfold_avg))


if __name__ == "__main__":
    main()



