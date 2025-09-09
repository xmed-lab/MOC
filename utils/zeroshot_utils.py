
import os
import random
from tqdm import tqdm
import numpy as np
import json
from sklearn.metrics import (balanced_accuracy_score, cohen_kappa_score, 
                             classification_report, roc_auc_score)

import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F

from models.conch.downstream.utils import AverageMeter, merge_dict
from models.conch.open_clip_custom import tokenize, get_tokenizer

from transformers import CLIPProcessor, CLIPModel

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


@torch.no_grad()
def zero_shot_classifier_plip(classnames, templates, device=None):
    """
    classnames: list of lists of classnames (one list of classnames per class)
    templates: list of templates 
    """
    model = CLIPModel.from_pretrained("vinid/plip")
    model.to(device)
    processor = CLIPProcessor.from_pretrained("vinid/plip")

    zeroshot_weights = []
    for classnames_for_class in classnames:
        embeddings_for_class = []
        for classname in classnames_for_class:
            texts = [template.replace('CLASSNAME', classname) for template in templates]
            inputs = processor(texts, return_tensors="pt", padding=True)
            inputs.to(device)
            classname_embeddings = model.get_text_features(**inputs)
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


@torch.no_grad()
def prompt2weight(model, prompt, tokenizer=None, device=None):
    if tokenizer is None:
        tokenizer = get_tokenizer()
        
    token_ids = tokenize(tokenizer, [prompt]) # Tokenize with custom tokenizer
    token_ids = token_ids.to(device)
    prompt_embedding = model.encode_text(token_ids)
    prompt_embedding = F.normalize(prompt_embedding, dim=-1)
    return prompt_embedding


@torch.no_grad()
def cls2weightWtemplate(model, classname, templates, tokenizer=None, device=None):
    if tokenizer is None:
        tokenizer = get_tokenizer()

    embeddings_for_class = []
    for template in templates:
        text = template.replace('CLASSNAME', classname)
        token_ids = tokenize(tokenizer, [text])
        token_ids = token_ids.to(device)
        classname_embedding = model.encode_text(token_ids)
        # classname_embedding: [num_templates, embedding_dim]
        embeddings_for_class.append(F.normalize(classname_embedding, dim=-1))
    
    # class_embedding: [num_templates, embedding_dim]
    class_embedding = torch.stack(embeddings_for_class, dim=0)
    # over all templates and classnames
    class_embedding = class_embedding.mean(dim=0)
    class_embedding /= class_embedding.norm()
    return class_embedding


def promptfile2weight(conch_model, device, prompt_file, label_map):
    idx_to_class = {v: k for k, v in label_map.items()}
    with open(prompt_file, 'r') as f:
        prompts = json.load(f)['0']
    classnames = prompts['classnames']
    templates = prompts['templates']
    n_classes = len(classnames)
    classnames_text = [classnames[str(idx_to_class[idx])] for idx in range(n_classes)]
    zeroshot_weights = zero_shot_classifier(conch_model, classnames_text, templates, device=device)
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


@torch.no_grad()
def run_mizero_simple(model, classifier, dataloader, device, topj = (1,5,10,50,100), 
        dump_results = False, dump_patch_level = False, 
        metrics=['acc', 'bacc', 'weighted_kappa', 'kappa', 'roc_auc', 'weighted_f1']):                                            
        
    dict_keys = list(topj)
    meters = {j: AverageMeter() for j in dict_keys}

    logits_all, targets_all, patch_logits_all, coords_all, preds_all = {}, [], [], [], {}
    for idx, data in enumerate(dataloader): # batch size is always 1
        image_features = data[0].to(device).squeeze(0)
        target = data[1].to(device)
        logits = image_features @ classifier
        preds, pooled_logits = topj_pooling(logits, topj = topj)
        results = {key: (val == target).float().item() for key, val in preds.items()}
        
        preds_all = merge_dict(preds_all, preds, value_fn = lambda x: x.item())
        logits_all = merge_dict(logits_all, pooled_logits, value_fn = lambda x: x.cpu().numpy())
        targets_all.append(target.cpu().numpy())

        for j in topj:
            meters[j].update(results[j], n=1) # Update AverageMeters with new results

    # Save raw preds & targets
    targets_all = np.concatenate(targets_all, axis=0)
    logits_all = {key: np.concatenate(logits_all[key], axis=0) for key in dict_keys}
    probs_all = {key: F.softmax(torch.from_numpy(logits_all[key]) * 56.3477, dim=1).numpy() for key in dict_keys}
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
        
    return results, dump


@torch.no_grad()
def run_mizero_simple_4visual(model, classifier, dataloader, device, topj = (1,5,10,50,100), 
        dump_results = False, dump_patch_level = False, 
        metrics=['acc', 'bacc', 'weighted_kappa', 'kappa', 'roc_auc', 'weighted_f1'], 
        coords_list=['luad', 'lusc'], pooling_policy_func=None, **kwargs):                                            
    
    assert pooling_policy_func is not None, "pooling_policy_func must be provided"
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
        
        preds, pooled_logits, indices = pooling_policy_func(logits, topj = topj, return_indices=True, coords_list=coords_list, **kwargs)
        top_coords_all[patient_idx] = {}
        for i, coords_key in enumerate(coords_list):
            top_coords_all[patient_idx][coords_key] = coords[indices[:,i].cpu().numpy()]
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
        
    return results, dump, top_coords_all
