from pathlib import Path
from typing import Dict
from gensim.test.utils import get_tmpfile
from gensim.models.callbacks import CallbackAny2Vec
import os
import numpy as np

RELATIONS = os.path.join("Relation", "Relation.csv")
BLOGCATALOG_EDGE = os.path.join("BlogCatalog-dataset/data", "edges.csv")
BLOGCATALOG_NODE = os.path.join("BlogCatalog-dataset/data", "nodes.csv")
BLOGCATALOG_FEATURES = os.path.join("./", "features_node2vec.pkl")
BLOGCATALOG_LABELS = os.path.join("BlogCatalog-dataset/data", "group-edges.csv")

def alias_setup(probs):
    '''
    probs： 某个概率分布
    返回: Alias数组与Prob数组
    '''
    K       = len(probs)
    q       = np.zeros(K) # 对应Prob数组
    J       = np.zeros(K, dtype=np.int) # 对应Alias数组
    # Sort the data into the outcomes with probabilities
    # that are larger and smaller than 1/K.
    smaller = [] # 存储比1小的列
    larger  = [] # 存储比1大的列
    for kk, prob in enumerate(probs):
        q[kk] = K*prob # 概率
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)
 
    # Loop though and create little binary mixtures that
    # appropriately allocate the larger outcomes over the
    # overall uniform mixture.
    
    # 通过拼凑，将各个类别都凑为1
    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()
 
        J[small] = large # 填充Alias数组
        q[large] = q[large] - (1.0 - q[small]) # 将大的分到小的上
 
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)
 
    return J, q
 
def alias_draw(J, q):
    '''
    输入: Prob数组和Alias数组
    输出: 一次采样结果
    '''
    K  = len(J)
    # Draw from the overall uniform mixture.
    kk = int(np.floor(np.random.rand()*K)) # 随机取一列
 
    # Draw from the binary mixture, either keeping the
    # small one, or choosing the associated larger one.
    if np.random.rand() < q[kk]: # 比较
        return kk
    else:
        return J[kk]
 

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
