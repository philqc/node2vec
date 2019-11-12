import pandas as pd
import numpy as np
from utils import project_root, prob_distribution_from_dict, BLOGCATALOG_EDGE
import os
from typing import Dict, List, Tuple
import pdb
import json
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO,
                    datefmt="%Y-%m-%d %H:%M:%S")


EDGE_CSV = os.path.join(project_root(), "tests", "data", BLOGCATALOG_EDGE)
PATH_PROBS = os.path.join(project_root(), "tests", "prob_relations_BLOGCATALOG.json")

PARAMETERS = {
    "q": 0.25,
    "p": 0.25
}

COL1 = 'col1'
COL2 = 'col2'

#col1 equals to 'user', col2 equals to 'page'
def load_edges(edge_csv: str) -> pd.DataFrame:
    edges = pd.read_csv(edge_csv, header=None, names=[COL1, COL2])
    # make sure every node number is presented string for skipgram model
    edges[COL1] = edges[COL1].astype(str)
    edges[COL2] = edges[COL2].astype(str)
    return edges


def list_all_nodes(df: pd.DataFrame) -> List[str]:
    users = df[COL1].unique()
    pages = df[COL2].unique()
    return list(set(users).union(set(pages)))

def list_pages_nodes(df: pd.DataFrame) -> List[str]:
    users = df[COL1].unique()
    pages = df[COL2].unique()
    return list(set(users).union(set(pages)))


def get_neighbors_neighbors(df_start: pd.DataFrame, df_neighbors: pd.DataFrame, p: float, q: float) -> Dict:
    dct = {}
    logging.info("get_neighbors_neighbors: %s ids to compute" % len(df_start))
    for i, (previous, possible_starts) in enumerate(df_start.items()):
        if i % 100 == 0 and i > 0:
            logging.info("Precomputed %s nodes" % i)
        dct[previous] = {}
        for start in possible_starts:
            # Probability to get back to itself
            dct[previous][start] = {previous: 1 / p}
            for neighbor in df_neighbors[start]:
                # Second neighbors
                if neighbor != previous:
                    if neighbor in possible_starts:
                        # Previous node and start share the same neighbor !
                        # TODO: I think this will never be the case with Relations.csv?
                        dct[previous][start][neighbor] = 1
                    else:
                        # there is a distance of 2 between previous and neighbor
                        dct[previous][start][neighbor] = 1 / q
            # Transform to probability distribution
            dct[previous][start] = prob_distribution_from_dict(dct[previous][start])

    return dct


def get_transition_probabilites(df: pd.DataFrame, save_dict: bool, drop_page_ids: bool, p: float = PARAMETERS["p"],
                                q: float = PARAMETERS["q"]) -> Tuple[Dict[str, Dict[str, Dict[str, float]]], List[str]]:

    # Calculate this here cuz we modify dataframe
    all_nodes = list_all_nodes(df)
   
    df_users = df.groupby(COL1)[COL2].apply(list)
    # Recompute df_pages as well
    df_pages = df.groupby(COL2)[COL1].apply(list)
    

    df_total = (df_users.append(df_pages)).groupby(level=0).apply(sum)

    
    # Modify df now removing columns with page ids to drop
    logging.info("df_users.shape = %s; df_pages.shape = %s" % (df_users.shape, df_pages.shape))

    logging.info("Getting All Nodes' neighbors and its neighbors' neighbors")
    user_neighbors = get_neighbors_neighbors(df_total,  df_total, p, q)

    if save_dict:
        with open(PATH_PROBS, 'w', encoding='utf-8') as f_out:
            json.dump(user_neighbors, f_out)

    return user_neighbors, all_nodes


