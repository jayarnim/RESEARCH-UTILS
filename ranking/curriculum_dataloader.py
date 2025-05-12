from typing import Optional
from itertools import chain
import pandas as pd
from ..config.constants import (
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
)
from .negative_sampling_dataloader import NegativeSamplingDataLoader as dataloader


class DataLoaderCombination:
    def __init__(self, dataloader_list):
        self.dataloader_list = dataloader_list

    def __iter__(self):
        return chain(*self.dataloader_list)

    def __len__(self):
        return sum(len(dl) for dl in self.dataloader_list)


class CurriculumDataLoader:
    def __init__(
        self,
        origin: pd.DataFrame,
        col_user: str=DEFAULT_USER_COL,
        col_item: str=DEFAULT_ITEM_COL,
        n_phases: Optional[int]=None,
    ):
        self.origin = origin
        self.col_user = col_user
        self.col_item = col_item
        self.n_phases = n_phases
        self.dataloader = dataloader(origin, col_user, col_item)

    def get(
        self,
        data: pd.DataFrame,
        neg_per_pos: int=10,
        batch_size: int=32,
    ):

        phase_user_list = self._split_users_by_histlen(self.n_phases)

        phase_data_list = [
            self._filter_by_user(data, phase_user)
            for phase_user in phase_user_list
        ]

        phase_loader_list = [
            self.dataloader.get(phase_data, neg_per_pos, batch_size)
            for phase_data in phase_data_list
        ]

        combined_iter = DataLoaderCombination(phase_loader_list)

        return combined_iter


    def _split_users_by_histlen(
        self,
        n_phases: int=4,
    ):
        user2histlen = (
            self.origin
            .groupby(self.col_user)[self.col_item]
            .count()
            .to_dict()
        )

        sorted_users = sorted(
            user2histlen, 
            key=user2histlen.get,
        )

        n_total = len(sorted_users)

        phase_user_list = []

        for i in range(self.n_phases):
            start = (i * n_total) // self.n_phases
            end = ((i + 1) * n_total) // self.n_phases
            phase_user = set(sorted_users[start:end])
            phase_user_list.append(phase_user)

        return phase_user_list

    def _filter_by_user(
        self,
        data: pd.DataFrame,
        phase_users: set,
    ):
        CONDITION = data[self.col_user].isin(phase_users)
        return data[CONDITION].reset_index(drop=True)
