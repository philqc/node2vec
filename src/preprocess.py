import pandas as pd
from src.utils import prob_distribution_from_dict
from typing import Dict, List, Tuple
from src.config import logging, LIKE_ID, USER_ID


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


def list_pages_nodes(df: pd.DataFrame) -> List[str]:
    return list(df[LIKE_ID].unique())


def get_neighbors_neighbors(df_start: pd.DataFrame, df_neighbors: pd.DataFrame, p: float, q: float) -> Dict:
    dct = {}  # type: ignore
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
                        dct[previous][start][neighbor] = 1
                    else:
                        # there is a distance of 2 between previous and neighbor
                        dct[previous][start][neighbor] = 1 / q
            # Transform to probability distribution
            dct[previous][start] = prob_distribution_from_dict(dct[previous][start])

    return dct


def get_transition_probabilites(
        df: pd.DataFrame, drop_page_ids: bool, min_like: int,
        p: float = 1., q: float = 1.
) -> Tuple[Dict[str, Dict[str, Dict[str, float]]], List[str]]:
    if drop_page_ids:
        if min_like <= 1:
            logging.warning("drop_page_ids is set to true but min_like (%s) <= 1 --> no pages will be removed"
                            % min_like)
        else:
            df_pages = df.groupby(LIKE_ID)[USER_ID].apply(list)
            len_bef = len(df)
            logging.info("Dropping pages with less than {} likes -> Shape before = {}".format(min_like, df.shape))
            pages_to_drop = []
            for page_id, user_liked in df_pages.iteritems():
                if len(user_liked) < min_like:
                    pages_to_drop.append(page_id)

            df = df[~df[LIKE_ID].isin(pages_to_drop)]
            logging.info("Dropped a total of {} page_ids".format(len_bef - df.shape[0]))
    if len(df) == 0:
        raise RuntimeError("No more pages left (either min_like (%s) is too big or input data is invalid" % min_like)

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

    return user_neighbors, all_nodes
