from typing import Dict
from gensim.test.utils import get_tmpfile
from gensim.models.callbacks import CallbackAny2Vec
import pandas as pd
import os
import numpy as np

from src.config import RelationsData


def prob_distribution_from_dict(dct: Dict):
    total = sum(dct.values())
    dct = {k: v / total for k, v in dct.items()}
    return dct


def modify_path_if_exists(path, extension: str):
    """
    If path exists, iterates until we find a path
    that doesn't exist to store model/file
    :param path:
    :param extension: ['json', 'pkl', 'txt', etc.]
    """
    extension = '.' + extension
    if os.path.exists(path + extension):
        i = 0
        while os.path.exists(path + '_' + str(i) + extension):
            i += 1
        path += '_' + str(i) + extension
    else:
        path += extension
    return path


# Class for a memory-friendly iterator over the dataset
class MySentences(object):
    def __init__(self, filename):
        self.filename = filename

    def __iter__(self):
        for line in open(self.filename, encoding='utf-8'):
            yield line.split()


class EpochSaver(CallbackAny2Vec):
    """Callback to save model after each epoch."""

    def __init__(self, path_prefix):
        self.path_prefix = path_prefix
        self.epoch = 0

    def on_epoch_end(self, model):
        output_path = get_tmpfile('{}_epoch{}.model'.format(self.path_prefix, self.epoch))
        model.save(output_path)
        self.epoch += 1


def create_fake_test_csv():
    """
    For testing purpose/benchmarking speed
    Create dummy csv file with arbitrary number of users/pages
    to benchmark performance
    """
    user_ids = np.random.randint(0, 10 ** 4, size=10 ** 4)
    page_ids = np.random.randint(0, 10 ** 3, size=10 ** 4)
    page_ids = np.array(["pageid_" + str(_id) for _id in page_ids])
    data = np.stack((user_ids, page_ids), axis=1)
    pd.DataFrame(
        data, columns=[RelationsData.USER_ID, RelationsData.LIKE_ID]
    ).to_csv('../tests/data/Relation/Fake_Big_Relation.csv')


if __name__ == "__main__":
    create_fake_test_csv()
