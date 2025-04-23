import random
import itertools
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from ..config.constants import (
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
)


class NegativeSamplingDataset(Dataset):
    def __init__(
        self, 
        data: pd.DataFrame, 
        neg_items_per_user: dict,
        neg_per_pos: int=10,
        col_user: str=DEFAULT_USER_COL,
        col_item: str=DEFAULT_ITEM_COL,
    ):
        # global attr
        self.user_item_pairs = list(zip(data[col_user], data[col_item]))
        self.neg_items_per_user = neg_items_per_user
        self.neg_per_pos = neg_per_pos
        self.col_user = col_user
        self.col_item = col_item

    def __len__(self):
        return len(self.user_item_pairs)

    def __getitem__(self, idx):
        user, pos = self.user_item_pairs[idx]
        
        # Negative Sampling
        neg_list = random.sample(
            population=self.neg_items_per_user[user],
            k=self.neg_per_pos
            )

        user_list = [user] * (1 + self.neg_per_pos)
        item_list = [pos] + neg_list
        label_list = [1] + [0] * self.neg_per_pos

        return user_list, item_list, label_list


class NegativeSamplingDataLoader:
    def __init__(
        self,
        origin: pd.DataFrame,
        col_user: str=DEFAULT_USER_COL,
        col_item: str=DEFAULT_ITEM_COL,
    ):
        # global attr
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
        dataset = NegativeSamplingDataset(
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
        
        pos_per_user = {
            user: set(data[data[DEFAULT_USER_COL] == user][DEFAULT_ITEM_COL])
            for user in all_users
        }

        neg_items_per_user = {
            user: list(set(all_items) - pos_per_user[user])
            for user in all_users
        }

        return neg_items_per_user

    def _collate(self, batch):
        user_list, item_list, label_list = zip(*batch)
        
        user_batch = torch.tensor(
            list(itertools.chain.from_iterable(user_list)), 
            dtype=torch.long
        )
        
        item_batch = torch.tensor(
            list(itertools.chain.from_iterable(item_list)), 
            dtype=torch.long
        )
        
        label_batch = torch.tensor(
            list(itertools.chain.from_iterable(label_list)),
            dtype=torch.float32
        )
        
        return user_batch, item_batch, label_batch