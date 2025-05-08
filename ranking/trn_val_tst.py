from typing import Optional
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from sklearn.feature_extraction.text import TfidfTransformer
from ..config.constants import (
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
)
from ..msr.python_splitters import python_stratified_split
from .curriculum_dataloader import CurriculumDataLoader as dataloader


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
        self.dataloader = dataloader(data, col_user, col_item)

    def get(
        self, 
        filter_by: str = "user",
        trn_val_tst_ratio: list=[0.8, 0.1, 0.1],
        neg_per_pos: list=[4, 4, 100, 100],
        batch_size: list=[32, 32, 1, 1],
        n_phases: int=4,
        max_hist: Optional[int]=None,
        seed: int=42,
    ):
        loo = (
            self.data
            .groupby(self.col_user)
            .sample(n=1, random_state=seed)
            .sort_values(by=self.col_user)
            .reset_index(drop=True)
        )

        remain = (
            self.data[~self.data[[self.col_user, self.col_item]]
            .apply(tuple, axis=1)
            .isin(set(loo[[self.col_user, self.col_item]]
            .apply(tuple, axis=1)))]
            .reset_index(drop=True)
        )

        split_list = python_stratified_split(
            data=remain,
            filter_by=filter_by,
            ratio=trn_val_tst_ratio,
            col_user=self.col_user,
            col_item=self.col_item,
            seed=seed,
        )

        split_list.append(loo)

        loaders = []
        zip_obj = zip(split_list, neg_per_pos, batch_size)
        for split_, neg_, batch_ in zip_obj:
            loader = self.dataloader.get(
                data=split_, 
                neg_per_pos=neg_, 
                batch_size=batch_, 
                n_phases=n_phases,
            )
            loaders.append(loader)

        histories = self._histories(
            data=split_list[0],
            max_hist=max_hist,
        )

        return loaders, histories

    def _histories(
        self, 
        data: pd.DataFrame, 
        max_hist: Optional[int]=None,
    ):
        tfidf = self._tfidf(data) if max_hist is not None else None

        all_users = sorted(data[self.col_user].unique())
        pos_per_user_ids = []

        for user in all_users:
            items = data[data[self.col_user] == user][self.col_item].unique()
            item_ids = torch.tensor(items, dtype=torch.long)

            # TF-IDF 기반 정렬 및 top-k 선택
            if max_hist is not None and len(items) > max_hist:
                scores = torch.tensor(
                    [tfidf.get((user, item), 0.0) for item in items],
                    dtype=torch.float32
                )
                topk_vals, topk_indices = torch.topk(scores, k=max_hist)
                item_ids = item_ids[topk_indices]

            pos_per_user_ids.append(item_ids)

        pos_per_user_padding = pad_sequence(
            pos_per_user_ids,
            batch_first=True,
            padding_value=self.n_items
        )

        return pos_per_user_padding

    def _tfidf(
        self, 
        data: pd.DataFrame,
    ):
        user_item_matrix = (
            data.groupby([self.col_user, self.col_item])
            .size()
            .unstack(fill_value=0)
            .astype(float)
        )

        tfidf = TfidfTransformer(norm=None)
        tfidf_matrix = tfidf.fit_transform(user_item_matrix)

        tfidf_df = pd.DataFrame(
            tfidf_matrix.toarray(),
            index=user_item_matrix.index,
            columns=user_item_matrix.columns,
        )

        tfidf_dict = {
            (user, item): tfidf_df.loc[user, item]
            for user in tfidf_df.index
            for item in tfidf_df.columns
            if tfidf_df.loc[user, item] > 0
        }

        return tfidf_dict
