import torch
import torch.nn.functional as F

'''
modify than _bck: use patch ori_score instead of using selection_score
use softmax select patch, then use selected patch's ori logits for classification

method list:
topj
delta_softmax
delta_diff
topj * delta_softmax
topj * delta_diff
bottomk_irrelevant
'''

def index_topj_classifier(logits, topj, **kwargs):
    """
    mean of topk logits for each class
    logits: N x C logit for each patch
    topj: tuple of the top number of patches to use for pooling
    """
    # Sums logits across topj patches for each class, to get class prediction for each topj
    maxj = min(max(topj), logits.size(0)) # Ensures j is smaller than number of patches. Unlikely for number of patches to be < 10, but just in case
    _, indices = logits.topk(maxj, 0, True, True) # maxj x C
    return indices

def index_delta_softmax_classifier(logits, topj, **kwargs):
    '''
    delta - most discriminative patch
    softmax for delta implementation
    '''
    maxj = min(max(topj), logits.size(0))
    softmax_logits = F.softmax(logits, dim=1)
    _, indices = softmax_logits.topk(maxj, 0, True, True)
    return indices

def index_delta_diff_classifier(logits, topj, **kwargs):
    '''
    delta - most discriminative patch
    diff for delta implementation
    diff = top1 - top2
    '''
    maxj = min(max(topj), logits.size(0))
    # every row, abs(top1 - top2)
    top1_logits_row = torch.topk(logits, 1, dim=1)[0][:, 0]
    top2_logits_row = torch.topk(logits, 2, dim=1)[0][:, 1]
    diff_logits = torch.abs(top1_logits_row - top2_logits_row)
    diff_logits = torch.stack([diff_logits] * logits.size(1), dim=1)
    _, indices = diff_logits.topk(maxj, 0, True, True)
    return indices

def index_bottomk_irrel_classifier(logits, topj, n_classes, bottomk=None, detection=False, **kwargs):
    '''
    least irrelevant
    least like background/normal classes
    logits should have more bg classes (> coords_list)
    logits: tumorA, tumorB, normal0, normal1, normal2, ...
    '''
    assert n_classes is not None, "coords_list should be provided"
    assert logits.size(1) > n_classes, "logits should have more bg classes"

    maxj = min(max(topj), logits.size(0))
    if bottomk is None:
        bottomk = maxj
    if detection:
        fg_logits = logits[:, 0].unsqueeze(1)
        bg_logits = logits[:, 1:]
        top1_bg_logits_row = torch.topk(bg_logits, 1, dim=1)[0][:, 0]
    else:
        fg_logits = logits[:, :n_classes]
        bg_logits = logits[:, n_classes:]

    # merge bg_logits to one class
    bg_logits = bg_logits.sum(dim=1)
    
    # get least irrelevant patches
    if bottomk > bg_logits.size(0):
        print("heyhey small", bottomk, bg_logits.size(0))
        bottomk = bg_logits.size(0)
    _, bg_indices = bg_logits.topk(bottomk, 0, False, True) # return smallest
    fg_logits = fg_logits[bg_indices]
    if detection:
        fg_logits = torch.cat([fg_logits, top1_bg_logits_row[bg_indices].unsqueeze(1)], dim=1)
    _, fg_indices = fg_logits.topk(maxj, 0, True, True)
    indices = bg_indices[fg_indices]
    return indices

