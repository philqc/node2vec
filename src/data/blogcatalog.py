import pandas as pd
import os

from src.data.base import DataLoader


class BlogCatalogDataLoader(DataLoader):
    COL1 = 'col1'
    COL2 = 'col2'

    def __init__(
            self,
            path_edge_csv: str,
            min_like: int = 1
    ):
        if not os.path.exists(path_edge_csv):
            raise ValueError(f"path_csv provided doesn't exist = {path_edge_csv}")

        df = pd.read_csv(path_edge_csv, header=None, names=[self.COL1, self.COL2])
        super().__init__(df, self.COL1, self.COL2, min_like)
