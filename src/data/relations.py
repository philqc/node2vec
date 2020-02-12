import pandas as pd
import os
from typing import Optional
from src.config import logging
from src.data.base import DataLoader


class RelationsDataLoader(DataLoader):
    def __init__(
            self,
            path_csv: str,
            col_user_id: Optional[str] = None,
            col_like_id: Optional[str] = None,
            min_like: int = 1
    ):
        if not os.path.exists(path_csv):
            raise ValueError(f"path_csv provided doesn't exist = {path_csv}")

        df = pd.read_csv(path_csv)

        if col_user_id is None:
            user_id = df.columns.tolist()[1]
            logging.info(f"col_user_id is not provided so we use 2nd column of .csv file = {user_id} as users")
        else:
            user_id = col_user_id
        if col_like_id is None:
            like_id = df.columns.tolist()[2]
            logging.info(f"col_like_id is not provided so we use 3rd column of .csv file = {like_id} as likes")
        else:
            like_id = col_like_id

        super().__init__(df, user_id, like_id, min_like)
