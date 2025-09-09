import argparse
import pdb
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import math
from pathlib import Path

# internal imports
from utils.file_utils import save_pkl, load_pkl
from utils.utils import *
from utils.core_utils import train
from datasets.dataset_generic import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset
from models.model_conch import conch_coca
# from models.conch.downstream.zeroshot_path import zero_shot_classifier, run_mizero
from models.conch.downstream.utils import AverageMeter, merge_dict
from models.conch.open_clip_custom import tokenize, get_tokenizer

# pytorch imports
import torch
from torch.utils.data import DataLoader, sampler, Dataset
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
import numpy as np
import json
from sklearn.metrics import (balanced_accuracy_score, cohen_kappa_score, 
                             classification_report, roc_auc_score)
from tqdm import tqdm
from openslide import OpenSlide
import cv2
from PIL import Image


@torch.no_grad()
def zero_shot_classifier(model, classnames, templates, tokenizer=None, device=None):
    """
    classnames: list of lists of classnames (one list of classnames per class)
    templates: list of templates 
    """
    if tokenizer is None:
        tokenizer = get_tokenizer()

    zeroshot_weights = []
    for classnames_for_class in classnames:
        embeddings_for_class = []
        for classname in classnames_for_class:
            texts = [template.replace('CLASSNAME', classname) for template in templates]
            token_ids = tokenize(tokenizer, texts) # Tokenize with custom tokenizer
            token_ids = token_ids.to(device)
            classname_embeddings = model.encode_text(token_ids)
            # classname_embeddings: [num_templates, embedding_dim]
            embeddings_for_class.append(F.normalize(classname_embeddings, dim=-1))

        # class_embedding: [num_classnames, num_templates, embedding_dim]
        class_embedding = torch.stack(embeddings_for_class, dim=0)
        # over all templates and classnames
        class_embedding = class_embedding.mean(dim=(0, 1))
        class_embedding /= class_embedding.norm()

        # class_embedding: [embedding_dim]
        zeroshot_weights.append(class_embedding)

    # zeroshot_weights: [embedding_dim, num_classes]
    zeroshot_weights = torch.stack(zeroshot_weights, dim=1)
    return zeroshot_weights


def topj_pooling(logits, topj):
    """
    logits: N x C logit for each patch
    topj: tuple of the top number of patches to use for pooling
    """
    # Sums logits across topj patches for each class, to get class prediction for each topj
    maxj = min(max(topj), logits.size(0)) # Ensures j is smaller than number of patches. Unlikely for number of patches to be < 10, but just in case
    values, _ = logits.topk(maxj, 0, True, True) # maxj x C
    preds = {j : values[:min(j, maxj)].mean(dim=0, keepdim=True) for j in topj} # dict of 1 x C logit scores
    pooled_logits = {key: val for key,val in preds.items()}    
    preds = {key: val.argmax(dim=1) for key,val in preds.items()} # dict of predicted class indices
    return preds, pooled_logits


def topj_pooling_return_idx(logits, topj):
    """
    logits: N x C logit for each patch
    topj: tuple of the top number of patches to use for pooling
    """
    # Sums logits across topj patches for each class, to get class prediction for each topj
    maxj = min(max(topj), logits.size(0)) # Ensures j is smaller than number of patches. Unlikely for number of patches to be < 10, but just in case
    values, indices = logits.topk(maxj, 0, True, True) # maxj x C
    # print(values.shape, indices.shape)
    # print(values)
    # print(indices)
    preds = {j : values[:min(j, maxj)].mean(dim=0, keepdim=True) for j in topj} # dict of 1 x C logit scores
    pooled_logits = {key: val for key,val in preds.items()}    
    preds = {key: val.argmax(dim=1) for key,val in preds.items()} # dict of predicted class indices
    return preds, pooled_logits, indices


@torch.no_grad()
def run_mizero_simple_4visual(model, classifier, dataloader, device, topj = (1,5,10,50,100), 
        dump_results = False, dump_patch_level = False, 
        metrics=['acc', 'bacc', 'weighted_kappa', 'kappa', 'roc_auc', 'weighted_f1']):                                            
    
    # modified from run_mizero_simple for visualization
    dict_keys = list(topj)
    meters = {j: AverageMeter() for j in dict_keys}

    logits_all, targets_all, patch_logits_all, coords_all, preds_all, top_coords_all = {}, [], [], [], {}, {}
    for idx, data in enumerate(tqdm(dataloader)): # batch size is always 1
        image_features = data[0].to(device).squeeze(0)
        target = data[1].to(device)
        coords = data[2].squeeze().numpy()
        full_path = data[3][0]  # loader returns a list of length 1
        patient_idx = full_path.split("/")[-1].replace(".h5", "")

        logits = image_features @ classifier
        logits = logits[:, :2]
        # logits = F.softmax(logits, dim=1) #ANCHOR - softmax before
        preds, pooled_logits, indices = topj_pooling_return_idx(logits, topj = topj)
        # pooled_logits = {key: F.softmax(val, dim=1) for key,val in pooled_logits.items()}   #ANCHOR - softmax after
        top_coords_all[patient_idx] = {}
        top_coords_all[patient_idx]['luad'] = coords[indices[:,0].cpu().numpy()]
        top_coords_all[patient_idx]['lusc'] = coords[indices[:,1].cpu().numpy()]
        # return top_coords_all
        results = {key: (val == target).float().item() for key, val in preds.items()}
        
        preds_all = merge_dict(preds_all, preds, value_fn = lambda x: x.item())
        logits_all = merge_dict(logits_all, pooled_logits, value_fn = lambda x: x.cpu().numpy())
        targets_all.append(target.cpu().numpy())

        for j in topj:
            meters[j].update(results[j], n=1) # Update AverageMeters with new results

    # Save raw preds & targets
    targets_all = np.concatenate(targets_all, axis=0)
    logits_all = {key: np.concatenate(logits_all[key], axis=0) for key in dict_keys}
    logits_scale = model.logit_scale.exp().item() if model else 56.3477
    probs_all = {key: F.softmax(torch.from_numpy(logits_all[key]) * logits_scale, dim=1).numpy() for key in dict_keys}
    # Compute metrics
    preds_all = {key: np.array(preds_all[key]) for key in dict_keys}
    baccs = {key: balanced_accuracy_score(targets_all, val) for key, val in preds_all.items()}
    cls_rep = {key: classification_report(targets_all, val, output_dict=True, zero_division=0) for key, val in preds_all.items()}
    kappas = {key: cohen_kappa_score(targets_all, val) for key, val in preds_all.items()}
    weighted_kappas = {key: cohen_kappa_score(targets_all, val, weights='quadratic') for key, val in preds_all.items()}
    roc_aucs = {}
    for key, probs in probs_all.items():
        n_classes = probs.shape[1]
        if n_classes == 2:
            class_probs = probs[:,1]
            roc_kwargs = {}
        else:
            class_probs = probs
            roc_kwargs = {'multi_class': 'ovo', 'average': 'macro'}        
        try:
            roc_auc = roc_auc_score(targets_all, class_probs, **roc_kwargs)
        except ValueError:
            roc_auc = np.nan
        roc_aucs[key] = roc_auc

    # Get final accuracy across all images
    accs = {j: meters[j].avg for j in topj}

    dump = {}
    results = {'acc': accs, 
            'bacc': baccs, 
            'report': cls_rep, 
            'kappa': kappas,
            'weighted_kappa': weighted_kappas, # quadratic weights
            'roc_auc': roc_aucs,
            'weighted_f1': {key: cls_rep[key]['weighted avg']['f1-score'] for key in dict_keys}}
    results = {k: results[k] for k in metrics}
    if dump_results:
        # dump slide level predictions
        dump['logits'] = logits_all
        dump['targets'] = targets_all
        dump['preds'] = preds_all
        # if hasattr(model, "logit_scale"):
        #     dump['temp_scale'] = model.logit_scale.exp().item()
        
    return results, dump, top_coords_all


nsclc_zeroshot_path = "/home/txiang/pathology/CLAM/models/classifier_weights/nsclc_luad_lusc.pt"
zeroshot_weights = torch.load(nsclc_zeroshot_path)


dataset = Generic_MIL_Dataset(csv_path='dataset_csv/nsclc.csv',
                                    data_dir=os.path.join('data/nsclc', 'merge_features_conch'),
                                    shuffle=False,
                                    seed=1,
                                    print_info=True,
                                    label_dict= {'LUAD':0, 'LUSC':1},
                                    patient_strat=False,
                                    ignore=[])
dataset.load_from_h5(True)
dataset.load_full_path(True)

feats, lbl, coords, full_path = dataset[0]
loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

print("running mizero")
results, dump, top_coords = run_mizero_simple_4visual(None, zeroshot_weights, loader, device, 
                                                      metrics=['roc_auc', 'acc', 'bacc', 'weighted_f1', 'kappa'], 
                                                      dump_results=True, topj = [10])
print(results)

raw_dir = "/home/txiang/pathology/CLAM/data/nsclc_raw"
visual_target_dir = "visual_results/conch_zs_topk"
os.makedirs(visual_target_dir, exist_ok=True)

def topk_visual_summary(results, dump, top_coords, exp="debug"):
    result_dir = os.path.join(visual_target_dir, exp)
    os.makedirs(result_dir, exist_ok=True)
    with open(os.path.join(result_dir, "results.json"), "w") as f:
        json.dump(results, f)

    for idx, patient in enumerate(tqdm(top_coords.items())):
        # if idx < 3:
        #     continue
        lbl = dump['targets'][idx]
        pred = dump['preds'][10][idx]
        lbl = "LUAD" if lbl == 0 else "LUSC"
        pred = "LUAD" if pred == 0 else "LUSC"
        patient_idx, coords = patient
        patient_slide = os.path.join(raw_dir, lbl, f"{patient_idx}.svs")
        patient_target_visual_dir = os.path.join(visual_target_dir, exp, lbl, patient_idx)
        os.makedirs(patient_target_visual_dir, exist_ok=True)
        # print(lbl, pred)
        # break
        
        if lbl == pred:
            Path(os.path.join(patient_target_visual_dir, "_correct.txt")).touch()
        else:
            Path(os.path.join(patient_target_visual_dir, "_wrong.txt")).touch()

        # basic - wsi thumbnail
        slide = OpenSlide(patient_slide)
        ori_size = slide.dimensions
        thumbnail = slide.get_thumbnail((800, 800))
        # thumbnail.save(os.path.join(patient_target_visual_dir, "thumbnail.png"))

        patch_size = 256
        # visualize topk patches
        for top_ind, coord in enumerate(coords['luad']):
            x, y = coord
            x, y = int(x), int(y)
            patch = slide.read_region((x, y), 0, (patch_size, patch_size))
            patch.save(os.path.join(patient_target_visual_dir, f"luad_top_{top_ind}_{x}_{y}.png"))
        
        for top_ind, coord in enumerate(coords['lusc']):
            x, y = coord
            x, y = int(x), int(y)
            patch = slide.read_region((x, y), 0, (patch_size, patch_size))
            patch.save(os.path.join(patient_target_visual_dir, f"lusc_top_{top_ind}_{x}_{y}.png"))
        
        # visualize topk patches on thumbnail
        thumbnail = slide.get_thumbnail((2000, 2000))
        thumbnail = thumbnail.convert("RGB")
        thumbnail = np.array(thumbnail)
        # print(ori_size, thumbnail.shape)
        for top_ind, coord in enumerate(coords['luad']):
            x, y = coord
            x, y = int(x), int(y)
            relative_x = x / ori_size[0] * thumbnail.shape[1]
            relative_x1 = (x+patch_size) / ori_size[0] * thumbnail.shape[1] 
            relative_y = y / ori_size[1] * thumbnail.shape[0]
            relative_y1 = (y+patch_size) / ori_size[1] * thumbnail.shape[0]
            relative_x, relative_y, relative_x1, relative_y1 = int(relative_x), int(relative_y), int(relative_x1), int(relative_y1)
            cv2.rectangle(thumbnail, (relative_x, relative_y), (relative_x1, relative_y1), (255, 0, 0), 4)
            # print(relative_x, relative_y, relative_x1, relative_y1)
        
        for top_ind, coord in enumerate(coords['lusc']):
            x, y = coord
            x, y = int(x), int(y)
            relative_x = x / ori_size[0] * thumbnail.shape[1]
            relative_x1 = (x+patch_size) / ori_size[0] * thumbnail.shape[1] 
            relative_y = y / ori_size[1] * thumbnail.shape[0]
            relative_y1 = (y+patch_size) / ori_size[1] * thumbnail.shape[0]
            relative_x, relative_y, relative_x1, relative_y1 = int(relative_x), int(relative_y), int(relative_x1), int(relative_y1)
            cv2.rectangle(thumbnail, (relative_x, relative_y), (relative_x1, relative_y1), (0, 0, 255), 4)
            # print(relative_x, relative_y, relative_x1, relative_y1)
            # break
        thumbnail = Image.fromarray(thumbnail)
        thumbnail.save(os.path.join(patient_target_visual_dir, "thumbnail_overlay.png"))
        # thumbnail.show()
        # print(thumbnail.size)
        # print(ori_size)
        # print(coords)
        # break
        # if idx > 5:
        #     break

print('start visualization')
topk_visual_summary(results, dump, top_coords, exp="no_softmax")

