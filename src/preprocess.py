import pandas as pd
from src.utils import project_root, prob_distribution_from_dict
import os
from typing import Dict, List
import pdb
import json
from pprint import pprint

TEST_CSV = os.path.join(project_root(), "tests", "data", "Relation", "Relation.csv")
PATH_PROBS = os.path.join(project_root(), "tests", "prob_relations.json")
LIKE_ID = "like_id"
USER_ID = "userid"

PARAMETERS = {
    "q": 0.5,
    "p": 2
}


def load_csv(path_csv: str) -> pd.DataFrame:
    df = pd.read_csv(path_csv)
    return df


def list_all_nodes(df: pd.DataFrame) -> List[str]:
    users = df[USER_ID].unique()
    pages = df[LIKE_ID].unique()
    return list(users) + list(pages)


def get_neighbors_neighbors(df_start, df_neighbors):
    dct = {}
    for previous, possible_starts in df_start.items():
        dct[previous] = {}
        for start in possible_starts:
            # Probability to get back to itself
            dct[previous][start] = {previous: 1 / PARAMETERS["p"]}
            for neighbor in df_neighbors[start]:
                # Second neighbors
                if neighbor != previous:
                    if neighbor in possible_starts:
                        # Previous node and start share the same neighbor !
                        # TODO: I think this will never be the case with Relations.csv?
                        dct[previous][start][neighbor] = 1
                    else:
                        # there is a distance of 2 between previous and neighbor
                        dct[previous][start][neighbor] = 1 / PARAMETERS["q"]
            # Transform to probability distribution
            dct[previous][start] = prob_distribution_from_dict(dct[previous][start])

    return dct


def get_transition_probabilites(df: pd.DataFrame, save_dict: bool) -> Dict[str, Dict[str, Dict[str, float]]]:
    df_users = df.groupby(USER_ID)[LIKE_ID].apply(list)
    df_pages = df.groupby(LIKE_ID)[USER_ID].apply(list)

    # Users'neighbors and neighbors'neighbors
    user_neighbors = get_neighbors_neighbors(df_users, df_pages)
    # Now get pages'neighbors and neighbors'neighbors
    pages_neighbors = get_neighbors_neighbors(df_pages, df_users)

    # This all neighbors now
    user_neighbors.update(pages_neighbors)

    if save_dict:
        with open(PATH_PROBS, 'w', encoding='utf-8') as f_out:
            json.dump(user_neighbors, f_out)

    return user_neighbors


def main():
    df = load_csv(TEST_CSV)
    all_nodes = list_all_nodes(df)
    # get_transition_probabilites(df, True)
    #pdb.set_trace()


if __name__ == "__main__":
    main()
