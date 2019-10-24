import numpy as np
from typing import Dict, List
from pprint import pprint
import pdb
import json
import random
from src.preprocess import *


def random_walk(matrix_prob: Dict, previous_node: str, length: int):
    """ TODO: Find out how to start the random walk since we
    need information about the previous node to know the probability distribution
    """
    try:
        # Actually using the start node as the previous node and randomly sampling a a start node
        # TODO: Find out how they did in paper
        possible_starts = matrix_prob[previous_node].keys()
        start_node = random.sample(possible_starts, 1)[0]

        walk = [previous_node, start_node]
        for i in range(length):
            # probability distribution
            p_dist = matrix_prob[walk[-2]][walk[-1]]
            # draw a sample
            sample = np.random.choice(list(p_dist.keys()), p=list(p_dist.values()))

            walk.append(sample)
    except KeyError as err:
        raise KeyError(err)

    # remove previous node because it is not sampled according to prob.distribution
    return walk[1:]


def learn_features(matrix_prob: Dict, list_nodes: List[str], dim_features: int = 128,
                   walks_per_node: int = 10, walk_length: int = 80, context_size: int = 10):
    walks = []
    for i in range(walks_per_node):
        for node in list_nodes:
            walks.append(random_walk(matrix_prob, node, walk_length))

    # pprint(walks)
    optimize(walks, context_size, dim_features)


def optimize(walks: List[List[str]], context_size: int, dim_features: int):
    # TODO
    pass


def main():

    df = load_csv(TEST_CSV)
    matrix_prob = get_transition_probabilites(df, False)
    list_nodes = list_all_nodes(df)
    # with open(PATH_PROBS, 'r', encoding='utf-8') as f_in:
    #    prob_transition = json.load(f_in)

    learn_features(matrix_prob, list_nodes, walks_per_node=3, walk_length=5)
    #random_walk(prob_transition, "page1", 10)
    # pdb.set_trace()


if __name__ == "__main__":
    main()
