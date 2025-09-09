import os
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

def topj_pooling(logits, topj, return_indices=False, **kwargs):
    """
    mean of topk logits for each class
    logits: N x C logit for each patch
    topj: tuple of the top number of patches to use for pooling
    """
    # Sums logits across topj patches for each class, to get class prediction for each topj
    maxj = min(max(topj), logits.size(0)) # Ensures j is smaller than number of patches. Unlikely for number of patches to be < 10, but just in case
    values, indices = logits.topk(maxj, 0, True, True) # maxj x C
    preds = {j : values[:min(j, maxj)].mean(dim=0, keepdim=True) for j in topj} # dict of 1 x C logit scores
    pooled_logits = {key: val for key,val in preds.items()}    
    preds = {key: val.argmax(dim=1) for key,val in preds.items()} # dict of predicted class indices
    if return_indices:
        return preds, pooled_logits, indices
    return preds, pooled_logits


def delta_softmax_classifier_pooling(logits, topj, return_indices=False, **kwargs):
    '''
    delta - most discriminative patch
    softmax for delta implementation
    '''
    maxj = min(max(topj), logits.size(0))
    softmax_logits = F.softmax(logits, dim=1)
    # mix_logits = softmax_logits * logits
    # values, indices = softmax_logits.topk(maxj, 0, True, True)
    _, indices = softmax_logits.topk(maxj, 0, True, True)
    values = torch.stack([logits[indices[:, i], i] for i in range(logits.size(1))], dim=1)
    preds = {j : values[:min(j, maxj)].mean(dim=0, keepdim=True) for j in topj}
    pooled_logits = {key: val for key,val in preds.items()}
    preds = {key: val.argmax(dim=1) for key,val in preds.items()}

    if return_indices:
        return preds, pooled_logits, indices
    else:
        return preds, pooled_logits


def delta_diff_classifier_pooling(logits, topj, return_indices=False, **kwargs):
    '''
    delta - most discriminative patch
    diff for delta implementation
    diff = top1 - top2
    '''
    maxj = min(max(topj), logits.size(0))
    # diff_logits = torch.abs(logits[:, 0] - logits[:, 1])
    # every row, abs(top1 - top2)
    top1_logits_row = torch.topk(logits, 1, dim=1)[0][:, 0]
    top2_logits_row = torch.topk(logits, 2, dim=1)[0][:, 1]
    diff_logits = torch.abs(top1_logits_row - top2_logits_row)
    diff_logits = torch.stack([diff_logits] * logits.size(1), dim=1)
    _, indices = diff_logits.topk(maxj, 0, True, True)
    values = logits[indices[:, 0]]
    # mix_logits = logits * diff_logits
    # values, indices = mix_logits.topk(maxj, 0, True, True)
    preds = {j : values[:min(j, maxj)].mean(dim=0, keepdim=True) for j in topj}
    pooled_logits = {key: val for key,val in preds.items()}
    preds = {key: val.argmax(dim=1) for key,val in preds.items()}

    if return_indices:
        return preds, pooled_logits, indices
    else:
        return preds, pooled_logits


def topj_delta_diff_classifier_pooling(logits, topj, return_indices=False, **kwargs):
    '''
    ori_logits * delta_logits
    delta_logits = top1 - top2
    '''
    maxj = min(max(topj), logits.size(0))
    # every row, abs(top1 - top2)
    top1_logits_row = torch.topk(logits, 1, dim=1)[0][:, 0]
    top2_logits_row = torch.topk(logits, 2, dim=1)[0][:, 1]
    diff_logits = torch.abs(top1_logits_row - top2_logits_row)
    diff_logits = torch.stack([diff_logits] * logits.size(1), dim=1)
    mix_logits = logits * diff_logits
    _, indices = mix_logits.topk(maxj, 0, True, True)
    values = torch.stack([logits[indices[:, i], i] for i in range(logits.size(1))], dim=1)
    preds = {j : values[:min(j, maxj)].mean(dim=0, keepdim=True) for j in topj}
    pooled_logits = {key: val for key,val in preds.items()}
    preds = {key: val.argmax(dim=1) for key,val in preds.items()}

    if return_indices:
        return preds, pooled_logits, indices
    else:
        return preds, pooled_logits
    

def topj_delta_softmax_classifier_pooling(logits, topj, return_indices=False, **kwargs):
    '''
    ori_logits * delta_logits
    delta_logits = softmax(logits)
    '''
    maxj = min(max(topj), logits.size(0))
    softmax_logits = F.softmax(logits, dim=1)
    mix_logits = softmax_logits * logits
    _, indices = mix_logits.topk(maxj, 0, True, True)
    values = torch.stack([logits[indices[:, i], i] for i in range(logits.size(1))], dim=1)
    preds = {j : values[:min(j, maxj)].mean(dim=0, keepdim=True) for j in topj}
    pooled_logits = {key: val for key,val in preds.items()}
    preds = {key: val.argmax(dim=1) for key,val in preds.items()}

    if return_indices:
        return preds, pooled_logits, indices
    else:
        return preds, pooled_logits


def bottomk_irrel_classifier_pooling(logits, topj, return_indices=False, coords_list=None, bottomk=None, detection=False, **kwargs):
    '''
    least irrelevant
    least like background/normal classes
    logits should have more bg classes (> coords_list)
    logits: tumorA, tumorB, normal0, normal1, normal2, ...
    '''
    assert coords_list is not None, "coords_list should be provided"
    if type(coords_list) == int:
        assert logits.size(1) > coords_list, "logits should have more bg classes"
        coords_list = list(range(coords_list))
    elif type(coords_list) == list:
        assert logits.size(1) > len(coords_list), "logits should have more bg classes"
    else:
        raise ValueError("coords_list should be int or list")

    maxj = min(max(topj), logits.size(0))
    if bottomk is None:
        bottomk = maxj
    if detection:
        fg_logits = logits[:, 0].unsqueeze(1)
        bg_logits = logits[:, 1:]
        top1_bg_logits_row = torch.topk(bg_logits, 1, dim=1)[0][:, 0]
    else:
        fg_logits = logits[:, :len(coords_list)]
        bg_logits = logits[:, len(coords_list):]

    # merge bg_logits to one class
    bg_logits = bg_logits.sum(dim=1)
    
    # get least irrelevant patches
    bg_values, bg_indices = bg_logits.topk(bottomk, 0, False, True) # return smallest
    fg_logits = fg_logits[bg_indices]
    if detection:
        fg_logits = torch.cat([fg_logits, top1_bg_logits_row[bg_indices].unsqueeze(1)], dim=1)
    fg_values, fg_indices = fg_logits.topk(maxj, 0, True, True)
    # get relative indice to original logits
    indices = bg_indices[fg_indices]
    preds = {j : fg_values[:min(j, maxj)].mean(dim=0, keepdim=True) for j in topj}
    pooled_logits = {key: val for key,val in preds.items()}
    preds = {key: val.argmax(dim=1) for key,val in preds.items()}
    if return_indices:
        return preds, pooled_logits, indices
    else:
        return preds, pooled_logits


def bottomk_irrel_delta_softmax_classifier_pooling(logits, topj, return_indices=False, coords_list=None, bottomk=None, detection=False, **kwargs):
    '''
    delta - least irrelevant patch
    first choose least irrelevant patches, then calculate softmax for delta implementation
    '''
    assert coords_list is not None, "coords_list should be provided"
    assert logits.size(1) > len(coords_list), "logits should have more bg classes"

    maxj = min(max(topj), logits.size(0))
    if bottomk is None:
        bottomk = maxj
    if detection:
        fg_logits = logits[:, 0].unsqueeze(1)
        bg_logits = logits[:, 1:]
        top1_bg_logits_row = torch.topk(bg_logits, 1, dim=1)[0][:, 0]
    else:
        fg_logits = logits[:, :len(coords_list)]
        bg_logits = logits[:, len(coords_list):]

    # merge bg_logits to one class
    bg_logits = bg_logits.sum(dim=1)

    # get least irrelevant patches
    bg_values, bg_indices = bg_logits.topk(bottomk, 0, False, True) # return smallest
    fg_logits = fg_logits[bg_indices]
    if detection:
        fg_logits = torch.cat([fg_logits, top1_bg_logits_row[bg_indices].unsqueeze(1)], dim=1)
    fg_logits_softmax = F.softmax(fg_logits, dim=1)
    _, fg_indices = fg_logits_softmax.topk(maxj, 0, True, True)
    fg_values = torch.stack([fg_logits[fg_indices[:, i], i] for i in range(fg_logits.size(1))], dim=1)
    indices = bg_indices[fg_indices]
    preds = {j : fg_values[:min(j, maxj)].mean(dim=0, keepdim=True) for j in topj}
    pooled_logits = {key: val for key,val in preds.items()}
    preds = {key: val.argmax(dim=1) for key,val in preds.items()}
    if return_indices:
        return preds, pooled_logits, indices
    else:
        return preds, pooled_logits
    

def bottomk_irrel_delta_diff_classifier_pooling(logits, topj, return_indices=False, coords_list=None, bottomk=None, detection=False, **kwargs):
    '''
    delta - least irrelevant patch
    first choose least irrelevant patches
    then calculate diff for delta implementation
    '''
    # choose least irrelevant patches, logits should have more bg classes (> coords_list)
    # also choose most discriminative patches
    assert coords_list is not None, "coords_list should be provided"
    assert logits.size(1) > len(coords_list), "logits should have more bg classes"

    maxj = min(max(topj), logits.size(0))
    if bottomk is None:
        bottomk = maxj
    if detection:
        fg_logits = logits[:, 0].unsqueeze(1)
        bg_logits = logits[:, 1:]
        top1_bg_logits_row = torch.topk(bg_logits, 1, dim=1)[0][:, 0]
    else:
        fg_logits = logits[:, :len(coords_list)]
        bg_logits = logits[:, len(coords_list):]

    # merge bg_logits to one class
    bg_logits = bg_logits.sum(dim=1)

    # get least irrelevant patches
    bg_values, bg_indices = bg_logits.topk(bottomk, 0, False, True) # return smallest
    fg_logits = fg_logits[bg_indices]
    if detection:
        fg_logits = torch.cat([fg_logits, top1_bg_logits_row[bg_indices].unsqueeze(1)], dim=1)
    # every row, abs(top1 - top2)
    top1_logits_row = torch.topk(fg_logits, 1, dim=1)[0][:, 0]
    top2_logits_row = torch.topk(fg_logits, 2, dim=1)[0][:, 1]
    diff_logits = torch.abs(top1_logits_row - top2_logits_row)
    diff_logits = torch.stack([diff_logits] * fg_logits.size(1), dim=1)
    # mix_logits = fg_logits * diff_logits
    # _, fg_indices = mix_logits.topk(maxj, 0, True, True)
    _, fg_indices = diff_logits.topk(maxj, 0, True, True)
    fg_values = torch.stack([fg_logits[fg_indices[:, i], i] for i in range(fg_logits.size(1))], dim=1)
    indices = bg_indices[fg_indices]
    preds = {j : fg_values[:min(j, maxj)].mean(dim=0, keepdim=True) for j in topj}
    pooled_logits = {key: val for key,val in preds.items()}
    preds = {key: val.argmax(dim=1) for key,val in preds.items()}
    if return_indices:
        return preds, pooled_logits, indices
    else:
        return preds, pooled_logits


def topj_bottomk_irrel_delta_softmax_classifier_pooling(logits, topj, return_indices=False, coords_list=None, bottomk=None, detection=False, **kwargs):
    '''
    first get least irrelevant patch logits,
    then softmax_logits * ori_logits
    '''
    assert coords_list is not None, "coords_list should be provided"
    assert logits.size(1) > len(coords_list), "logits should have more bg classes"

    maxj = min(max(topj), logits.size(0))
    if bottomk is None:
        bottomk = maxj
    if detection:
        fg_logits = logits[:, 0].unsqueeze(1)
        bg_logits = logits[:, 1:]
        top1_bg_logits_row = torch.topk(bg_logits, 1, dim=1)[0][:, 0]
    else:
        fg_logits = logits[:, :len(coords_list)]
        bg_logits = logits[:, len(coords_list):]

    # merge bg_logits to one class
    bg_logits = bg_logits.sum(dim=1)

    # get least irrelevant patches
    bg_values, bg_indices = bg_logits.topk(bottomk, 0, False, True) # return smallest
    fg_logits = fg_logits[bg_indices]
    if detection:
        fg_logits = torch.cat([fg_logits, top1_bg_logits_row[bg_indices].unsqueeze(1)], dim=1)
    fg_logits_softmax = F.softmax(fg_logits, dim=1)
    mix_logits = fg_logits_softmax * fg_logits
    _, fg_indices = mix_logits.topk(maxj, 0, True, True)
    fg_values = torch.stack([fg_logits[fg_indices[:, i], i] for i in range(fg_logits.size(1))], dim=1)
    indices = bg_indices[fg_indices]
    preds = {j : fg_values[:min(j, maxj)].mean(dim=0, keepdim=True) for j in topj}
    pooled_logits = {key: val for key,val in preds.items()}
    preds = {key: val.argmax(dim=1) for key,val in preds.items()}
    if return_indices:
        return preds, pooled_logits, indices
    else:
        return preds, pooled_logits

def topj_bottomk_irrel_delta_diff_classifier_pooling(logits, topj, return_indices=False, coords_list=None, bottomk=None, detection=False, **kwargs):
    '''
    first get least irrelevant patch logits,
    then diff_logits = top1 - top2
    then ori_logits * diff_logits
    '''
    assert coords_list is not None, "coords_list should be provided"
    assert logits.size(1) > len(coords_list), "logits should have more bg classes"

    maxj = min(max(topj), logits.size(0))
    if bottomk is None:
        bottomk = maxj
    if detection:
        fg_logits = logits[:, 0].unsqueeze(1)
        bg_logits = logits[:, 1:]
        top1_bg_logits_row = torch.topk(bg_logits, 1, dim=1)[0][:, 0]
    else:
        fg_logits = logits[:, :len(coords_list)]
        bg_logits = logits[:, len(coords_list):]

    # merge bg_logits to one class
    bg_logits = bg_logits.sum(dim=1)

    # get least irrelevant patches
    bg_values, bg_indices = bg_logits.topk(bottomk, 0, False, True) # return smallest
    fg_logits = fg_logits[bg_indices]
    if detection:
        fg_logits = torch.cat([fg_logits, top1_bg_logits_row[bg_indices].unsqueeze(1)], dim=1)
    # diff_logits = torch.abs(fg_logits[:, 0] - fg_logits[:, 1])
    # diff_logits = torch.stack([diff_logits, diff_logits], dim=1)
    # every row, abs(top1 - top2)
    top1_logits_row = torch.topk(fg_logits, 1, dim=1)[0][:, 0]
    top2_logits_row = torch.topk(fg_logits, 2, dim=1)[0][:, 1]
    diff_logits = torch.abs(top1_logits_row - top2_logits_row)
    diff_logits = torch.stack([diff_logits] * fg_logits.size(1), dim=1)
    mix_logits = fg_logits * diff_logits
    _, fg_indices = mix_logits.topk(maxj, 0, True, True)
    fg_values = torch.stack([fg_logits[fg_indices[:, i], i] for i in range(fg_logits.size(1))], dim=1)
    indices = bg_indices[fg_indices]
    preds = {j : fg_values[:min(j, maxj)].mean(dim=0, keepdim=True) for j in topj}
    pooled_logits = {key: val for key,val in preds.items()}
    preds = {key: val.argmax(dim=1) for key,val in preds.items()}
    if return_indices:
        return preds, pooled_logits, indices
    else:
        return preds, pooled_logits
    

# def topj_pooling_return_idx(logits, topj):
#     """
#     logits: N x C logit for each patch
#     topj: tuple of the top number of patches to use for pooling
#     """
#     # Sums logits across topj patches for each class, to get class prediction for each topj
#     topj_storage = topj[0]
#     if topj[0] < 1:    #percentage
#         topj[0] = int(logits.size(0) * topj[0])
#         # print("using percentage, selecting: ", topj[0])
#     maxj = min(max(topj), logits.size(0)) # Ensures j is smaller than number of patches. Unlikely for number of patches to be < 10, but just in case
#     # topj[0] = topj_storage
#     values, indices = logits.topk(maxj, 0, True, True) # maxj x C
#     if topj_storage < 1:
#         preds = {topj_storage : values[:min(j, maxj)].mean(dim=0, keepdim=True) for j in topj}
#     else:
#         preds = {j : values[:min(j, maxj)].mean(dim=0, keepdim=True) for j in topj} # dict of 1 x C logit scores
#     pooled_logits = {key: val for key,val in preds.items()}    
#     preds = {key: val.argmax(dim=1) for key,val in preds.items()} # dict of predicted class indices
#     topj[0] = topj_storage
#     return preds, pooled_logits, indices


# def topj_pooling_return_value(logits, topj):
#     maxj = min(max(topj), logits.size(0)) # Ensures j is smaller than number of patches. Unlikely for number of patches to be < 10, but just in case
#     values, _ = logits.topk(maxj, 0, True, True) # maxj x C
#     return values

# def bottomk_irrelavant_classifier_pooling_4detection(logits, topj, return_indices=False, coords_list=None, bottomk=None, **kwargs):
#     # choose least irrelevant patches, logits should have more bg classes (> coords_list)
#     # tumor, normal0, normal1, normal2, normal3, ...
#     # also choose most discriminative patches
#     assert coords_list is not None, "coords_list should be provided"
#     assert logits.size(1) > len(coords_list), "logits should have more bg classes"

#     maxj = min(max(topj), logits.size(0))
#     if bottomk is None:
#         bottomk = maxj
#     # fg_logits = logits[:, :len(coords_list)]
#     fg_logits = logits[:, 0].unsqueeze(1)
#     bg_logits = logits[:, 1:]
#     top1_bg_logits_row = torch.topk(bg_logits, 1, dim=1)[0][:, 0]

#     # merge bg_logits to one class
#     bg_logits = bg_logits.sum(dim=1)

#     # get least irrelevant patches
#     bg_values, bg_indices = bg_logits.topk(bottomk, 0, False, True) # return smallest
#     fg_logits = fg_logits[bg_indices]
#     # fg_logits = torch.cat([fg_logits, bg_logits[bg_indices].unsqueeze(1)], dim=1)
#     fg_logits = torch.cat([fg_logits, top1_bg_logits_row[bg_indices].unsqueeze(1)], dim=1)
#     fg_values, fg_indices = fg_logits.topk(maxj, 0, True, True)
#     indices = bg_indices[fg_indices]
#     preds = {j : fg_values[:min(j, maxj)].mean(dim=0, keepdim=True) for j in topj}
#     pooled_logits = {key: val for key,val in preds.items()}
#     preds = {key: val.argmax(dim=1) for key,val in preds.items()}
#     if return_indices:
#         return preds, pooled_logits, indices
#     else:
#         return preds, pooled_logits
    