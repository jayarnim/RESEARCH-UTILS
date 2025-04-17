import numpy as np
import torch.nn as nn
import pandas as pd
from ..config.constants import (
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
    DEFAULT_LABEL_COL,
    DEFAULT_PREDICTION_COL,
    DEFAULT_K,
)
from ..msr.python_evaluation import (
    hit_ratio_at_k,
    map_at_k, 
    ndcg_at_k, 
    precision_at_k, 
    recall_at_k
)


def rel_top_k(
    rating_true: pd.DataFrame,
    rating_pred: pd.DataFrame,
    col_user: str=DEFAULT_USER_COL,
    col_item: str=DEFAULT_ITEM_COL,
    col_rating: str=DEFAULT_LABEL_COL,
    col_prediction: str=DEFAULT_PREDICTION_COL,
    k: int=DEFAULT_K,
):
    kwargs = locals().copy()

    hr_ = hit_ratio_at_k(**kwargs)
    precision_ = precision_at_k(**kwargs)
    recall_ = recall_at_k(**kwargs)
    map_ = map_at_k(**kwargs)
    ndcg_ = ndcg_at_k(**kwargs)

    return hr_, precision_, recall_, map_, ndcg_


def aggdiv_top_k(
    rating_pred: pd.DataFrame,
    col_user: str=DEFAULT_USER_COL,
    col_item: str=DEFAULT_ITEM_COL,
    k: int=DEFAULT_K,
):
    aggdiv = rating_pred[col_item].nunique()
    norm = rating_pred[col_user].nunique() * k
    return aggdiv / norm


def novelty_at_k(
    origin: pd.DataFrame,
    rating_pred: pd.DataFrame,
    col_user: str=DEFAULT_USER_COL,
    col_item: str=DEFAULT_ITEM_COL,
):
    # P(i)
    item_counts = origin[col_item].value_counts()
    total_interactions = len(origin)
    item_probs = item_counts / total_interactions

    # -log2(P(i))
    novelty_scores = []
    for _, row in rating_pred.iterrows():
        item = row[col_item]
        pop = item_probs.get(item, 1e-6)
        novelty_scores.append(-np.log2(pop))

    # mean novelty
    rating_pred["novelty"] = novelty_scores
    user_novelty = rating_pred.groupby(col_user)["novelty"].mean()
    mean_novelty = user_novelty.mean()

    return mean_novelty


def serendipity_top_k(
    origin: pd.DataFrame,
    rating_true: pd.DataFrame,
    rating_pred: pd.DataFrame,
    col_user: str=DEFAULT_USER_COL,
    col_item: str=DEFAULT_ITEM_COL,
    k: int=DEFAULT_K,
):
    popular_items = (
        origin[col_item]
        .value_counts()
        .nlargest(k)
        .index
        .tolist()
    )
    popular_items_set = set(popular_items)

    true_dict = (
        rating_true.groupby(col_user)[col_item]
        .apply(set)
        .to_dict()
    )

    rec_dict = (
        rating_pred.groupby(col_user)[col_item]
        .apply(list)
        .to_dict()
    )

    total_score = 0.0
    user_count = 0

    for user, rec_items in rec_dict.items():
        rel_sum = 0

        for item in rec_items:
            if item not in popular_items_set:
                if item in true_dict.get(user, set()):
                    rel_sum += 1

        if len(rec_items) > 0:
            serendipity = rel_sum / len(rec_items)
            total_score += serendipity
            user_count += 1

    return total_score / user_count if user_count > 0 else 0.0


def personalization_at_k(
    rating_pred: pd.DataFrame,
    col_user: str=DEFAULT_USER_COL,
    col_item: str=DEFAULT_ITEM_COL,
    k: int=DEFAULT_K,
):
    user_item_dict = rating_pred.groupby(col_user)[col_item].apply(set)
    users = user_item_dict.index
    n_users = len(users)
    
    total_overlap = 0.0
    count = 0

    user_list = users.tolist()

    for i in range(n_users):
        for j in range(i + 1, n_users):
            u, v = user_list[i], user_list[j]
            Lu = user_item_dict[u]
            Lv = user_item_dict[v]
            overlap = len(Lu & Lv)
            total_overlap += overlap
            count += 1

    if count == 0:
        return 0.0
    else:
        avg_overlap = total_overlap / count
        personalization = 1 - (avg_overlap / k)
        return personalization


def eval_top_k(
    model_name: str,
    origin: pd.DataFrame,
    rating_true: pd.DataFrame,
    rating_pred: pd.DataFrame,
    col_user: str=DEFAULT_USER_COL,
    col_item: str=DEFAULT_ITEM_COL,
    col_rating: str=DEFAULT_LABEL_COL,
    col_prediction: str=DEFAULT_PREDICTION_COL,
    k: int=DEFAULT_K,
):
    rating_true = (
        rating_true[rating_true[col_rating]==1]
        .drop_duplicates(subset=[col_user, col_item])
        .sort_values(by=col_user, ascending=True)
        )

    rating_pred = (
        rating_pred
        .drop_duplicates(subset=[col_user, col_item])
        .sort_values(by=[col_user, col_prediction], ascending=[True, False])
        .groupby(col_user)
        .head(k)
        )

    hr_, precision_, recall_, map_, ndcg_ = rel_top_k(
        rating_true,
        rating_pred,
        col_user,
        col_item,
        col_rating,
        col_prediction,
        k,
    )

    aggdiv_ = aggdiv_top_k(
        rating_pred,
        col_user,
        col_item,
        k,
    )

    novelty_ = novelty_at_k(
        origin,
        rating_pred,
        col_user,
        col_item,
    )

    serendipity_ = serendipity_top_k(
        origin,
        rating_true,
        rating_pred,
        col_user,
        col_item,
        k,
        )

    per_ = personalization_at_k(
        rating_pred,
        col_user,
        col_item,
        k,
    )

    result = dict(
        model=model_name,
        hit_ratio=hr_, 
        precision=precision_, 
        recall=recall_, 
        map=map_, 
        ndcg=ndcg_,
        aggdiv=aggdiv_,
        novelty=novelty_,
        serendipity=serendipity_,
        personalization=per_,
    )

    print(
        f"HR@{k}: {hr_:f}",
        f"PRECISION@{k}: {precision_:f}",
        f"RECALL@{k}: {recall_:f}", 
        f"MAP@{k}: {map_:f}",
        f"NDCG@{k}: {ndcg_:f}",
        f"AGGDIV@{k}: {aggdiv_:f}",
        f"MEAN NOVELTY@{k}: {novelty_:f}",
        f"MEAN SERENDIPITY@{k}: {serendipity_:f}",
        f"PERSONALIZATION@{k}: {per_:f}", 
        sep='\n'
    )

    return result
