import torch
import numpy as np
from sklearn.metrics import roc_auc_score

def multi_binary_auc(logits, gt):
    if isinstance(logits, torch.Tensor):
        logits = logits.cpu().detach().numpy()
    if isinstance(gt, torch.Tensor):
        gt = gt.cpu().detach().numpy()
    gt_binarized = np.eye(len(np.unique(gt)))[gt]
    auc_list = []
    for i in range(gt_binarized.shape[1]):
        logit_i = logits[:, i]
        gt_i = gt_binarized[:, i]
        auc = roc_auc_score(gt_i, logit_i)
        auc_list.append(auc)
    return np.mean(auc_list)