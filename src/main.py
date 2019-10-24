import pandas as pd
import numpy as np
import pdb
import json
import random
from src.preprocess import PATH_PROBS


def random_walk(matrix_prob, previous_node: str, length: int):
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

    # print('Random walk =', walk)
    return walk


def main():
    with open(PATH_PROBS, 'r', encoding='utf-8') as f_in:
        prob_transition = json.load(f_in)

    random_walk(prob_transition, "page1", 10)
    # pdb.set_trace()


if __name__ == "__main__":
    main()