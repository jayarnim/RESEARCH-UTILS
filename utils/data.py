import pandas as pd
from sklearn.preprocessing import LabelEncoder
from config.constants import (
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
)


def description(
    data: pd.DataFrame, 
    col_user: str=DEFAULT_USER_COL, 
    col_item: str=DEFAULT_ITEM_COL
    ):
    N_USERS = data[col_user].nunique()
    N_ITEMS = data[col_item].nunique()
    TOTAL_INTERACTION = len(data)
    DENSITY = data.shape[0] / (N_USERS * N_ITEMS)

    print(
        f"number of user: {N_USERS}",
        f"number of item: {N_ITEMS}",
        f"total interaction: {TOTAL_INTERACTION}",
        f"mean interaction of user: {TOTAL_INTERACTION // N_USERS}",
        f"mean interaction of item: {TOTAL_INTERACTION // N_ITEMS}",
        f"interaction density: {DENSITY * 100:.4f} %",
        sep="\n",
    )


def filtering(
    data: pd.DataFrame, 
    col_user: str=DEFAULT_USER_COL, 
    col_item: str=DEFAULT_ITEM_COL,
    n_user: int=0,
    n_item: int=0,
    ):
    USER_COUNTS = data[col_user].value_counts()
    VALID_USER = USER_COUNTS[USER_COUNTS >= n_user].index

    ITEM_COUNTS = data[col_item].value_counts()
    VALID_ITEM = ITEM_COUNTS[ITEM_COUNTS >= n_item].index

    return VALID_USER, VALID_ITEM


def user_interaction_quantile(
    data: pd.DataFrame, 
    low: float=0.25,
    high: float=0.75,
    col_user: str=DEFAULT_USER_COL, 
):
    user_counts = data[col_user].value_counts()

    low_threshold = user_counts.quantile(low)
    high_threshold = user_counts.quantile(high)

    user_low = user_counts[user_counts==low_threshold].index[0]
    user_high = user_counts[user_counts==high_threshold].index[0]

    return user_low, user_high


def label_encoding(
    data: pd.DataFrame, 
    col_user: str=DEFAULT_USER_COL, 
    col_item: str=DEFAULT_ITEM_COL
    ):
    user_encoder = LabelEncoder()
    data[col_user] = user_encoder.fit_transform(data[col_user])
    user_label = dict(zip(user_encoder.classes_, user_encoder.transform(user_encoder.classes_)))
    
    item_encoder = LabelEncoder()
    data[col_item] = item_encoder.fit_transform(data[col_item])
    item_label = dict(zip(item_encoder.classes_, item_encoder.transform(item_encoder.classes_)))

    return data, user_label, item_label