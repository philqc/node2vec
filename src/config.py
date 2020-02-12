import os
from pathlib import Path
import logging

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S"
)


def project_root() -> Path:
    """Returns project root folder."""
    return Path(__file__).parent.parent


class RelationsData:
    """Bipartite graph from DataScience Projet"""
    FOLDER = os.path.join(project_root(), "tests", "data", "Relation")
    CSV_FILE = os.path.join(FOLDER, "Relation.csv")
    # Columns
    LIKE_ID = "like_id"
    USER_ID = "userid"


class BlogCatalogData:
    """ BlogCatalog Dataset """
    FOLDER = os.path.join(project_root(), "tests", "data", "BlogCatalog-dataset")
    EDGE_CSV = os.path.join(FOLDER, "data", "edges.csv")
    NODE_CSV = os.path.join(FOLDER, "data", "nodes.csv")
    LABELS_FILE = os.path.join(FOLDER, "data", "group-edges.csv")
    # Node2vec features file
    FEATURES_FILE = os.path.join(FOLDER, "features_node2vec_blogcatalog.pkl")
