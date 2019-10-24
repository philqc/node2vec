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


class EpochSaver(CallbackAny2Vec):
    """Callback to save model after each epoch."""

    def __init__(self, path_prefix):
        self.path_prefix = path_prefix
        self.epoch = 0

    def on_epoch_end(self, model):
        output_path = get_tmpfile('{}_epoch{}.model'.format(self.path_prefix, self.epoch))
        model.save(output_path)
        self.epoch += 1
