import numpy as np
import torch
import torch.nn.functional as F


class TaskLossFN:
    def __init__(self, type):
        self.type = type
        self._assert_arg_error()

    def compute(self, pos_score, neg_score):
        if self.type=='bpr':
            task_loss = self._bpr(pos_score, neg_score)
        else:
            task_loss = self._climf(pos_score, neg_score)
        return task_loss

    def _bpr(self, pos_score, neg_score):
        diff = pos_score.unsqueeze(1) - neg_score
        bpr_loss = -torch.log(torch.sigmoid(diff)).mean()
        return bpr_loss

    def _climf(self, score_i, score_j):
        term1 = F.logsigmoid(score_i)                       # log(sigmoid(score_i))
        diff = score_j - score_i.unsqueeze(1)               # [B, N]
        term2 = F.logsigmoid(-diff).sum(dim=1)              # log(1 - sigmoid(diff))
        return - (term1 + term2).mean()

    def _assert_arg_error(self):
        CONDITION = (self.type in ["bpr", "climf"])
        ERROR_MESSAGE = "argument for parameter 'type' must be either 'bpr' or 'climf'."
        assert CONDITION, ERROR_MESSAGE
