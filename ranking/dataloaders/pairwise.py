import random
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from config.constants import (
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
)

class PairwiseDataset(Dataset):
    def __init__(
        self, 
        data: pd.DataFrame, 
        neg_items_per_user: dict,
        neg_per_pos: int=10,
        col_user: str=DEFAULT_USER_COL,
        col_item: str=DEFAULT_ITEM_COL,
        ):
        self.user_item_pairs = list(zip(data[col_user], data[col_item]))
        self.neg_items_per_user = neg_items_per_user
        self.neg_per_pos = neg_per_pos
        self.col_user = col_user
        self.col_item = col_item

    def __len__(self):
        return len(self.user_item_pairs)

    def __getitem__(self, idx):
        return self._pairwise(idx)

    def _pairwise(self, idx):
        user, pos = self.user_item_pairs[idx]
        user_list = [user] * self.neg_per_pos
        pos_list = [pos] * self.neg_per_pos
        neg_list = random.sample(
            population=self.neg_items_per_user[user],
            k=self.neg_per_pos
            )

        return user_list, pos_list, neg_list


class PairwiseDataLoader:
    def __init__(
        self,
        origin: pd.DataFrame,
        col_user: str=DEFAULT_USER_COL,
        col_item: str=DEFAULT_ITEM_COL,
    ):
        self.col_user = col_user
        self.col_item = col_item
        self.neg_items_per_user = self._generate_negative_sample_pool(
            data=origin, 
            col_user=col_user, 
            col_item=col_item,
            )

    def get(
        self, 
        data: pd.DataFrame,
        neg_per_pos: int=10,
        batch_size: int=32,
    ):
        dataset = PairwiseDataset(
            data=data, 
            neg_items_per_user=self.neg_items_per_user,
            neg_per_pos=neg_per_pos,
            col_user=self.col_user, 
            col_item=self.col_item, 
            )
        loader = DataLoader(
            dataset=dataset, 
            batch_size=batch_size, 
            shuffle=True,
            collate_fn=self._collate,
            )
        return loader

    def _generate_negative_sample_pool(
        self,
        data: pd.DataFrame, 
        col_user: str=DEFAULT_USER_COL,
        col_item: str=DEFAULT_ITEM_COL,
    ):
        all_users = sorted(data[col_user].unique())
        all_items = sorted(data[col_item].unique())
        pos_items_per_user = (
            data.groupby(col_user)[col_item]
            .apply(set)
            .to_dict()
        )
        neg_items_per_user = {
            user: list(set(all_items) - pos_items_per_user[user])
            for user in all_users
        }
        return neg_items_per_user

    def _collate(self, batch):
        user_list, pos_list, neg_list = zip(*batch)  # unzip
        user_batch = torch.cat([torch.tensor(u) for u in user_list], dim=0)
        pos_batch = torch.cat([torch.tensor(p) for p in pos_list], dim=0)
        neg_batch = torch.cat([torch.tensor(n) for n in neg_list], dim=0)
        return user_batch, pos_batch, neg_batch