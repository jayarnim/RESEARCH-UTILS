import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from ..config.constants import (
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
        trn_val_tst_ratio: list = [0.8, 0.1, 0.1],
        neg_per_pos: list = [4, 4, 99],
        batch_size: list = [128, 128, 32],
        seed: int = 42,
    ):
        loo = (
            self.data
            .groupby(self.col_user)
            .sample(n=1, random_state=seed)
            .sort_values(by=self.col_user)
            .reset_index(drop=True)
        )
        
        loo_loader = self.dataloader.get(
            data=loo,
            neg_per_pos=99,
            batch_size=100,
        )

        remain = (
            self.data[~self.data[[self.col_user, self.col_item]]
            .apply(tuple, axis=1)
            .isin(set(loo[[self.col_user, self.col_item]]
            .apply(tuple, axis=1)))]
            .reset_index(drop=True)
        )

        trn, val, tst = python_stratified_split(
            data=remain,
            filter_by=filter_by,
            ratio=trn_val_tst_ratio,
            col_user=self.col_user,
            col_item=self.col_item,
            seed=seed,
        )

        trn_pos_per_user = self._histories(trn)

        loaders = []

        zip_obj = zip([trn, val, tst], neg_per_pos, batch_size)

        for split_, ratio_, batch_ in zip_obj:
            loader = self.dataloader.get(split_, ratio_, batch_)
            loaders.append(loader)

        loaders.append(loo_loader)

        return loaders, trn_pos_per_user

    def _histories(self, data):
        all_users = sorted(
            data[DEFAULT_USER_COL]
            .unique()
        )
        
        pos_per_user_dict = {
            user: set(data[data[DEFAULT_USER_COL] == user][DEFAULT_ITEM_COL])
            for user in all_users
        }

        pos_per_user_tensor = [
            torch.tensor(
                list(pos_per_user_dict.get(user, [])), 
                dtype=torch.long
            )
            for user in all_users
        ]

        pos_per_user = pad_sequence(
            pos_per_user_tensor, 
            batch_first=True, 
            padding_value=self.n_items,
        )

        return pos_per_user
