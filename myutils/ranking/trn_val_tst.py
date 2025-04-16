import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from config.constants import (
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
)
from ..msr.python_splitters import python_stratified_split
from . import negative_sampling


class Module:
    def __init__(
        self, 
        data: pd.DataFrame,
        n_users: int, 
        n_items: int,
        col_user: str=DEFAULT_USER_COL, 
        col_item: str=DEFAULT_ITEM_COL,
        ):
        self.data = data
        self.n_users = n_users
        self.n_items = n_items
        self.col_user = col_user
        self.col_item = col_item
        self.dataloader = negative_sampling.NegativeSamplingDataLoader(data, col_user, col_item)

    def get(
        self, 
        filter_by: str = "user",
        trn_val_tst_ratio: list = [0.7, 0.1, 0.2],
        neg_per_pos: list = [4, 1, 10],
        batch_size: list = [32, 128, 32],
        seed: int = 42,
    ):
        trn, val, tst = python_stratified_split(
            data=self.data,
            filter_by=filter_by,
            ratio=trn_val_tst_ratio,
            col_user=self.col_user,
            col_item=self.col_item,
            seed=seed,
        )

        pos_per_user = self._histories(trn)

        loaders = []

        zip_obj = zip([trn, val, tst], neg_per_pos, batch_size)

        for split_, ratio_, batch_ in zip_obj:
            loader = self.dataloader.get(split_, ratio_, batch_)
            loaders.append(loader)

        return loaders, pos_per_user

    def _histories(self, data):
        pos_per_user_dict = (
            data
            .sort_values(
                by=[DEFAULT_USER_COL, DEFAULT_ITEM_COL], 
                ascending=[True, True]
                )
            .groupby(DEFAULT_USER_COL)[DEFAULT_ITEM_COL]
            .apply(set)
            .to_dict()
            )

        pos_per_user_tensor = [
            torch.tensor(list(items), dtype=torch.long)
            for user, items in sorted(pos_per_user_dict.items())
        ]

        pos_per_user = pad_sequence(
            pos_per_user_tensor, 
            batch_first=True, 
            padding_value=self.n_items,
        )

        return pos_per_user