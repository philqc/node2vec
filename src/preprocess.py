import pandas as pd
from src.utils import project_root, prob_distribution_from_dict, RELATIONS
import os
from typing import Dict, List, Tuple
import pdb
import json
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO,
                    datefmt="%Y-%m-%d %H:%M:%S")

TEST_CSV = os.path.join(project_root(), "tests", "data", RELATIONS)
PATH_PROBS = os.path.join(project_root(), "tests", "prob_relations.json")
LIKE_ID = "like_id"
USER_ID = "userid"

PARAMETERS = {
    "q": 0.5,
    "p": 2
}


def load_csv(path_csv: str) -> pd.DataFrame:
    df = pd.read_csv(path_csv)
    # make sure every id is a string for skipgram model
    df[USER_ID] = df[USER_ID].astype(str)
    df[LIKE_ID] = df[LIKE_ID].astype(str)
    return df


def list_all_nodes(df: pd.DataFrame) -> List[str]:
    users = df[USER_ID].unique()
    pages = df[LIKE_ID].unique()
    return list(users) + list(pages)


def list_user_nodes(df: pd.DataFrame) -> List[str]:
    users = df[USER_ID].unique()
    return list(users)


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

    if drop_page_ids:
        df_pages = df.groupby(LIKE_ID)[USER_ID].apply(list)
        len_bef = len(df)
        logging.info("Dropping pages with only 1 like -> Shape before = {}".format(df.shape))
        pages_to_drop = []
        for page_id, user_liked in df_pages.iteritems():
            if len(user_liked) == 1:
                pages_to_drop.append(page_id)

        df = df[~df[LIKE_ID].isin(pages_to_drop)]
        logging.info("Dropped a total of {} page_ids".format(len_bef - df.shape[0]))

    # Calculate this here cuz we modify dataframe
    all_nodes = list_all_nodes(df)

    df_users = df.groupby(USER_ID)[LIKE_ID].apply(list)
    # Recompute df_pages as well
    df_pages = df.groupby(LIKE_ID)[USER_ID].apply(list)

    # Modify df now removing columns with page ids to drop
    logging.info("df_users.shape = %s; df_pages.shape = %s" % (df_users.shape, df_pages.shape))

    logging.info("Getting Users' neighbors and its neighbors' neighbors")
    user_neighbors = get_neighbors_neighbors(df_users, df_pages, p, q)
    logging.info("Getting Pages' neighbors and its neighbors' neighbors")
    pages_neighbors = get_neighbors_neighbors(df_pages, df_users, p, q)

    # This is all neighbors now
    user_neighbors.update(pages_neighbors)

    if save_dict:
        with open(PATH_PROBS, 'w', encoding='utf-8') as f_out:
            json.dump(user_neighbors, f_out)

    return user_neighbors, all_nodes


def main():
    df = load_csv(TEST_CSV)
    get_transition_probabilites(df, True, True)
    #pdb.set_trace()


if __name__ == "__main__":
    main()
