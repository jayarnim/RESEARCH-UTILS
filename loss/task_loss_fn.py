from typing import Literal
import torch
import torch.nn.functional as F


class Module:
    def __init__(
        self, 
        fn_type: Literal['bce', 'bpr', 'climf']='bce',
    ):
        self.fn_type = fn_type

    def compute(self, logit, target):
        if self.fn_type=='bce':
            return F.binary_cross_entropy_with_logits(logit, target)
        elif self.fn_type=='bpr':
            return self._bpr(logit, target)
        elif self.fn_type=='climf':
            return self._climf(logit, target)
        else:
            raise ValueError("Invalid Type of Loss Function")

    def _bpr(self, logit, target):
        mask_pos = (target==1)
        mask_neg = (target==0)
        logit_pos = logit[mask_pos]                                              # [B]
        logit_neg = logit[mask_neg].view(logit_pos.size(0), -1)                  # [B, N]

        diff = logit_pos.unsqueeze(1) - logit_neg
        return -torch.log(torch.sigmoid(diff)).mean()

    def _climf(self, logit, target):
        mask_pos = (target==1)
        mask_neg = (target==0)
        logit_pos = logit[mask_pos]                                 # [B]
        logit_neg = logit[mask_neg].view(logit_pos.size(0), -1)     # [B, N]

        term1 = F.logsigmoid(logit_pos)                             # log(sigmoid(score_i))
        diff = logit_neg - logit_pos.unsqueeze(1)                   # [B, N]
        term2 = F.logsigmoid(-diff).sum(dim=1)                      # log(1 - sigmoid(diff))
        return -(term1 + term2).mean()