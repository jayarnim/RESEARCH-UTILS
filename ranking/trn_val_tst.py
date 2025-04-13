import pandas as pd
from config.constants import (
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
)
from msr.python_splitters import python_stratified_split
from .dataloaders import (
    leave_one_out,
    pointwise,
    pairwise,
)


class NegativeSamplingDataloader:
    def __init__(self, data, col_user, col_item):
        self.data = data
        self.col_user = col_user
        self.col_item = col_item

        self.leave_one_out_ = leave_one_out.LeaveOneOutDataLoader(data, col_user, col_item)
        self.pointwise_ = pointwise.PointwiseDataLoader(data, col_user, col_item)
        self.pairwise_ = pairwise.PairwiseDataLoader(data, col_user, col_item)

    def get(
        self, 
        filter_by: str = "user",
        trn_val_tst_ratio: list = [0.7, 0.1, 0.2],
        neg_per_pos: list = [4, 1, 10],
        how_to_learn: list = ['pairwise', 'pairwise', 'pointwise'],
        batch_size: list = [128, 128, 32],
        seed: int = 42,
    ):
        splits = python_stratified_split(
            data=self.data,
            filter_by=filter_by,
            ratio=trn_val_tst_ratio,
            col_user=self.col_user,
            col_item=self.col_item,
            seed=seed,
        )

        loaders = []

        zip_obj = zip(how_to_learn, splits, neg_per_pos, batch_size)

        for learn_, split_, ratio_, batch_ in zip_obj:
            if learn_ == 'leave_one_out':
                loader = self.leave_one_out_.get(split_, ratio_, batch_)
                loaders.append(loader)

            elif learn_ == 'pointwise':
                loader = self.pointwise_.get(split_, ratio_, batch_)
                loaders.append(loader)

            else:
                loader = self.pairwise_.get(split_, ratio_, batch_)
                loaders.append(loader)

        return loaders
