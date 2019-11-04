from pathlib import Path
from typing import Dict
from gensim.test.utils import get_tmpfile
from gensim.models.callbacks import CallbackAny2Vec
import os

RELATIONS = os.path.join("Relation", "Relation.csv")


def project_root() -> Path:
    """Returns project root folder."""
    return Path(__file__).parent.parent


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
