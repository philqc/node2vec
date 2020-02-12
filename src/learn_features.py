import pickle
import argparse
import multiprocessing
import random
import gensim
import os
from typing import Dict, List
import tqdm

from src.config import logging, RelationsData, BlogCatalogData
from src.utils import MySentences
from src.data.base import DataLoader
from src.data.relations import RelationsDataLoader
from src.data.blogcatalog import BlogCatalogDataLoader
from src.data.weighted_dict import WeightedDict


ID = "id"
TRAIN = "train"
RESUME = "resume"
ALL = "all"
PREPROCESS = "preprocess"


def random_walk(matrix_prob: Dict[str, Dict[str, WeightedDict]], previous_node: str, length: int):
    try:
        # Actually using the start node as the previous node and randomly sampling a start node
        possible_starts = matrix_prob[previous_node].keys()
        start_node = random.sample(possible_starts, 1)[0]

        walk = [previous_node, start_node]
        for _ in range(length - 2):
            # draw a sample
            walk.append(matrix_prob[walk[-2]][walk[-1]].sample())

    except KeyError as err:
        raise KeyError(err)

    return walk


def sample_walks(path_save: str, matrix_prob: Dict, all_nodes: List[str],
                 walks_per_node: int = 10, walk_length: int = 80):
    with open(path_save, 'w', encoding='utf-8') as f_txt:
        for _ in tqdm.tqdm(range(walks_per_node), desc="Random walk"):
            for node in all_nodes:
                f_txt.write(" ".join(random_walk(matrix_prob, node, walk_length)) + '\n')


def optimize(path_sentences: str, like_nodes: List[str], mode: str, path_save: str,
             epochs: int = 10, context_size: int = 10, dim_features: int = 128, path_model: str = None):
    """
    :param path_sentences: Input of .txt file to sentences (one sentence per line)
    :param epochs: number of epochs to run model
    :param path_save: where to save the embeddings
    :param like_nodes: List of all like/item ids
    :param context_size: Also called window size
    :param dim_features:
    :param mode: {'train' or 'resume'} resume to resume training
    :param path_model: path model if we are resuming training
    :return:
    """
    cores = multiprocessing.cpu_count()

    # save model each epoch
    # epoch_logger = EpochSaver('word2vec')

    n_negative_samples = 10
    # minimum term frequency (to define the vocabulary)
    min_count = 2

    # a memory-friendly iterator
    sentences = MySentences(path_sentences)

    if mode in [TRAIN, ALL]:
        logging.info('Starting Training of Word2Vec Model')
        model = gensim.models.Word2Vec(sentences, min_count=min_count, sg=1, size=dim_features,
                                       iter=epochs, workers=cores, negative=n_negative_samples,
                                       window=context_size)
    elif mode == RESUME:
        logging.info('Resuming Training of Word2Vec Model')
        model = gensim.models.Word2Vec.load(path_model)
        # Start at the learning rate that we previously stopped
        model.train(sentences, total_examples=model.corpus_count, epochs=epochs,
                    start_alpha=model.min_alpha_yet_reached)
    else:
        raise ValueError('Specify valid value for mode (%s)' % mode)

    write_embeddings_to_file(model, like_nodes, path_save)


def write_embeddings_to_file(model: gensim.models.Word2Vec, like_nodes: List[str], path_save: str) -> None:
    logging.info('Writting embeddings to file %s' % path_save)
    embeddings = {}
    for v in list(model.wv.vocab):
        # we only keep likes' nodes embeddings
        if v in like_nodes:
            vec = model.wv.__getitem__(v)
            embeddings[str(v)] = vec

    with open(path_save, "wb") as f_out:
        pickle.dump(embeddings, f_out)


def preparing_samples(
        dataloader: DataLoader, p: float, q: float, walk_length: int,
        walks_per_node: int, context_size: int, path_save_sentences: str
):
    logging.info("Precomputing transition probabilities...")
    matrix_prob, list_nodes = dataloader.get_transition_probabilites(p, q)

    if context_size >= walk_length:
        raise ValueError("Context size can't be greater or equal to walk length !")

    logging.info("Sampling walks to create our dataset")
    sample_walks(path_save_sentences, matrix_prob, list_nodes, walks_per_node, walk_length)
    return dataloader.list_like_nodes()


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--type",
        type=str,
        help='Either "Relation" or "BlogCatalog" dataset',
        default="Relation"
    )
    parser.add_argument(
        "--save",
        help="Path of the folder to save the user features (default is same folder as data)",
        default=None,
    )
    parser.add_argument(
        "--walks_per_node",
        help="Number of random samples starting from each node",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--walk_length",
        help="Length of the random walk",
        type=int,
        default=80,
    )
    parser.add_argument(
        "--dim_features",
        help="Dimension of embedding",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--context_size",
        help="Context size for skip-gram model (windows size)",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--mode",
        help="{preprocess, train, resume, all}",
        type=str,
        default=ALL,
    )
    parser.add_argument(
        "--epochs",
        help="Number of epochs to run the model",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--p",
        help="Parameter p of node2vec model",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--q",
        help="Parameter q of node2vec model",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--min_like",
        help="An item needs min_like to be in the dataset",
        type=int,
        default=2,
    )

    args = parser.parse_args()

    return args


def main():
    args = parse()

    if args.type.lower() == "relation" or args.type.lower() == "relations":
        dataloader = RelationsDataLoader(RelationsData.CSV_FILE, min_like=args.min_like)
        folder = RelationsData.FOLDER
    elif args.type.lower() == "blogcatalog":
        dataloader = BlogCatalogDataLoader(BlogCatalogData.EDGE_CSV, min_like=args.min_like)
        folder = BlogCatalogData.FOLDER
    else:
        raise NotImplementedError("Other datatypes are not yet impleented")

    if args.save is None:
        args.save = folder

    str_save = f"_p_{args.p}_q_{args.q}_minLike_{args.min_like}"
    file_sampled_walks = "sampled_walks" + str_save + ".txt"
    # Save sample sentences (random walks) to a .txt file to be memory efficient
    path_sentences = os.path.join(args.save, file_sampled_walks)

    # add number of epochs for name file of embeddings
    str_save += f"_dim_{args.dim_features}_window_{args.context_size}_epochs_{args.epochs}"

    file_embeddings = "features_node2vec" + str_save + ".pkl"

    args.save = os.path.join(args.save, file_embeddings)

    if args.mode in [PREPROCESS, ALL]:
        like_nodes = preparing_samples(
            dataloader, args.p, args.q, args.walk_length,
            args.walks_per_node, args.context_size, path_sentences
        )
    else:
        like_nodes = dataloader.list_like_nodes()

    if args.mode in [ALL, TRAIN, RESUME]:
        logging.info("Starting training of skip-gram model")
        optimize(path_sentences, like_nodes, args.mode, args.save, args.epochs, args.context_size, args.dim_features)


if __name__ == "__main__":
    main()
