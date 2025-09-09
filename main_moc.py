import argparse
import os

# internal imports
from utils.utils import *
from utils.core_utils import train
from datasets.dataset_generic import Generic_MIL_Dataset
from models.model_conch import conch_coca
from utils.zeroshot_utils import zero_shot_classifier
from utils.patch_selection_classifier_index import (index_topj_classifier,
    index_delta_diff_classifier, index_delta_softmax_classifier, index_bottomk_irrel_classifier)
from utils.patch_selection_classifier import (topj_pooling, 
    delta_softmax_classifier_pooling, delta_diff_classifier_pooling, bottomk_irrel_classifier_pooling)

# pytorch imports
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
import numpy as np
import json
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from tqdm import tqdm
import pandas as pd
from glob import glob

def get_args():
    parser = argparse.ArgumentParser(description='Configurations for WSI Training')
    parser.add_argument('--fold', type=int, default=0, help='fold number')
    parser.add_argument('--shot', type=int, default=1, help='split number')
    parser.add_argument('--topj', type=int, default=10, help='topj for classifier selection')
    parser.add_argument('--topk', type=int, default=10, help='topk for final pooling')
    parser.add_argument('--result_dir', type=str, default='results/moc_train', help='result directory')
    parser.add_argument('--dataset', type=str, default='nsclc', choices=['nsclc', 'rcc'], help='dataset name')
    parser.add_argument('--pretrain', type=str, default='conch', choices=['conch'], help='pretrain model')
    parser.add_argument('--disable_tqdm', action='store_true', help='disable tqdm for better log')
    parser.add_argument('--discard_classifiers', nargs='+', default=[], help='topk, delta_softmax, delta_diff, bottomk')
    parser.add_argument('--load_weight', type=bool, default=True, help='load stored classifier weight')
    parser.add_argument('--check_zeroshot', type=bool, default=True, help='get zero-shot results')
    parser.add_argument('--ablation_study', type=str, default='none', choices=['none', 'avg', 'sum', 'max'], help='ablation study')
    parser.add_argument('--summary', action='store_true', help='summary results, no training')
    parser.add_argument('--summary_dir', type=str, default='')
    args = parser.parse_args()
    return args
args = get_args()

if not os.path.exists(args.result_dir):
    os.makedirs(args.result_dir)

# >>>>>>>>>>>>>>>>>>>> result summary >>>>>>>>>>>>>>>>>>>>>>>>
if args.summary:
    print("start summary")
    for shot in [1, 2, 4, 8]:
        summary_dir = args.summary_dir + f"/{shot}_shot"
        summary_file = os.path.join(args.summary_dir, f"summary_{shot}.csv")
        try:
            # if exist, remove first
            if os.path.exists(summary_file):
                os.remove(summary_file)
            folds = [0,1,2,3,4]
            test_aucs = []
            zs_test_aucs = []
            test_accs = []
            zs_test_accs = []
            for fold in folds:
                with open(os.path.join(summary_dir, f"best_results_shot_{shot}_fold_{fold}.json")) as f:
                    results = json.load(f)
                zs_test_auc = results['zero_shot_test']['auc']
                zs_test_acc = results['zero_shot_test']['acc']
                test_auc = results['test_at_best_val']
                test_acc = results['test_acc_at_best_val']
                zs_test_accs.append(zs_test_acc)
                test_accs.append(test_acc)
                test_aucs.append(test_auc)
                zs_test_aucs.append(zs_test_auc)
            folds.append("mean")
            test_aucs.append(np.mean(test_aucs))
            zs_test_aucs.append(np.mean(zs_test_aucs))
            test_accs.append(np.mean(test_accs))
            zs_test_accs.append(np.mean(zs_test_accs))
            pd.DataFrame({"fold": folds, "test_auc": test_aucs, "zs_test_auc": zs_test_aucs, "test_acc": test_accs, "zs_test_acc": zs_test_accs}).to_csv(summary_file, index=False)
        except:
            # check probably no zeroshot
            try:
                if os.path.exists(summary_file):
                    os.remove(summary_file)
                folds = [0,1,2,3,4]
                test_aucs = []
                test_accs = []
                for fold in folds:
                    with open(os.path.join(summary_dir, f"best_results_shot_{shot}_fold_{fold}.json")) as f:
                        results = json.load(f)
                    test_auc = results['test_at_best_val']
                    test_acc = results['test_acc_at_best_val']
                    test_accs.append(test_acc)
                    test_aucs.append(test_auc)
                folds.append("mean")
                test_aucs.append(np.mean(test_aucs))
                test_accs.append(np.mean(test_accs))
                pd.DataFrame({"fold": folds, "test_auc": test_aucs, "test_acc": test_accs}).to_csv(summary_file, index=False)
            except:
                # check probably doing ablation study
                try:
                    if os.path.exists(summary_file):
                        os.remove(summary_file)
                    folds = [0,1,2,3,4]
                    accs = []
                    aucs = []
                    for fold in folds:
                        # print(os.path.join(summary_dir, f"*_shot_{shot}_fold_{fold}.json"))
                        # print(glob(os.path.join(summary_dir, f"*_shot_{shot}_fold_{fold}.json")))
                        with open(glob(os.path.join(summary_dir, f"*_shot_{shot}_fold_{fold}.json"))[0]) as f:
                            results = json.load(f)
                        acc = results['acc']
                        auc = results['auc']
                        accs.append(acc)
                        aucs.append(auc)
                    folds.append("mean")
                    accs.append(np.mean(accs))
                    aucs.append(np.mean(aucs))
                    pd.DataFrame({"fold": folds, "auc": aucs, "acc": accs}).to_csv(summary_file, index=False)
                except:
                    print(f"shot {shot} summary failed")
    print("end summary")
    exit()

# <<<<<<<<<<<<<<<<<<<< result summary <<<<<<<<<<<<<<<<<<<<<<<<

# >>>>>>>>>>>>>>>>>>> data preparation >>>>>>>>>>>>>>>>>>>>>>>>
# prepare data and zeroshot classifiers
device = "cuda" if torch.cuda.is_available() else "cpu"
if args.pretrain == 'conch':
    checkpoint_path = "models/conch_checkpoint.bin"
    conch_model = conch_coca(checkpoint_path=checkpoint_path)
    conch_model.eval()
    conch_model.to(device)
prompt_file_tissue = 'models/prompts/prompts_patch_level_5cls.json'
label_map_tissue = {'Tumor': 0, 'Stroma': 1, 'Inflammation': 2, 'Vascular': 3, 'Necrosis': 4}
idx_to_class_tissue = {v:k for k,v in label_map_tissue.items()}
with open(prompt_file_tissue) as f:
    prompts_tissue = json.load(f)['0']
classnames_tissue = prompts_tissue['classnames']
templates_tissue = prompts_tissue['templates']
n_classes_tissue = len(classnames_tissue)
classnames_text_tissue = [classnames_tissue[str(idx_to_class_tissue[idx])] for idx in range(n_classes_tissue)]
if args.pretrain == 'conch':
    zeroshot_weights_tissue_conch_path = "models/classifier_weights/weights_tissue_conch.pt"
    if os.path.exists("models/classifier_weights") is False:
        os.makedirs("models/classifier_weights")
    if os.path.exists(zeroshot_weights_tissue_conch_path) and args.load_weight:
        zeroshot_weights_tissue = torch.load(zeroshot_weights_tissue_conch_path)
    else:
        zeroshot_weights_tissue = zero_shot_classifier(conch_model, classnames_text_tissue, templates_tissue, device=device)
        torch.save(zeroshot_weights_tissue, zeroshot_weights_tissue_conch_path)
else:
    raise NotImplementedError
print("zershot weights_tissue shape: ", zeroshot_weights_tissue.shape)

if args.dataset == 'nsclc':
    args.n_classes = 2
    prompt_file = "models/prompts/nsclc_prompts_all_per_class_worse.json"
    label_map = {'LUAD': 0, 'LUSC': 1}
    prompt_file_ext = 'models/prompts/nsclc_prompts_w4normal.json'
    label_map_ext = {'LUAD': 0, 'LUSC': 1, 'Stroma': 2, "Inflammation": 3, "Vascular": 4, "Necrosis": 5}

    idx_to_class = {v:k for k,v in label_map.items()}
    with open(prompt_file) as f:
        prompts = json.load(f)['0']
    classnames = prompts['classnames']
    templates = prompts['templates']
    n_classes = len(classnames)
    classnames_text = [classnames[str(idx_to_class[idx])] for idx in range(n_classes)]
    conch_temperature = 56.3477
    if args.pretrain == 'conch':
        zeroshot_weights_nsclc_conch_path = "models/classifier_weights/weights_nsclc_conch.pt"
        if os.path.exists(zeroshot_weights_nsclc_conch_path) and args.load_weight:
            zeroshot_weights = torch.load(zeroshot_weights_nsclc_conch_path)
        else:
            zeroshot_weights = zero_shot_classifier(conch_model, classnames_text, templates, device=device)
            torch.save(zeroshot_weights, zeroshot_weights_nsclc_conch_path)
    else:
        raise NotImplementedError

    idx_to_class_ext = {v:k for k,v in label_map_ext.items()}
    with open(prompt_file_ext) as f:
        prompts_ext = json.load(f)['0']
    classnames_ext = prompts_ext['classnames']
    classnames_text_ext = [classnames_ext[str(idx_to_class_ext[idx])] for idx in range(len(classnames_ext))]
    if args.pretrain == 'conch':
        zeroshot_weights_nsclc_ext_conch_path = "models/classifier_weights/weights_nsclc_ext_conch.pt"
        if os.path.exists(zeroshot_weights_nsclc_ext_conch_path) and args.load_weight:
            zeroshot_weights_ext = torch.load(zeroshot_weights_nsclc_ext_conch_path)
        else:
            zeroshot_weights_ext = zero_shot_classifier(conch_model, classnames_text_ext, templates, device=device)
            torch.save(zeroshot_weights_ext, zeroshot_weights_nsclc_ext_conch_path)
    else:
        raise NotImplementedError

    print("zershot weights shape: ", zeroshot_weights.shape)
    print("zershot weights_ext shape: ", zeroshot_weights_ext.shape)
    
    if args.pretrain == 'conch':
        data_dir = os.path.join('data/nsclc', 'merge_features_conch')
    else:
        raise NotImplementedError

    dataset = Generic_MIL_Dataset(csv_path='dataset_csv/nsclc.csv',
                                data_dir=data_dir,
                                shuffle=False,
                                seed=1,
                                print_info=True,
                                label_dict= {'LUAD':0, 'LUSC':1},
                                patient_strat=False,
                                ignore=[])
    dataset.load_from_h5(True)
    dataset.load_full_path(True)

    train_dataset, val_dataset, test_dataset = dataset.return_splits(from_id=False, csv_path=f'splits/nsclc_fewshot/{args.shot}shots/splits_{args.fold}.csv', repeat_num=int(args.shot)*2)
    train_dataset.load_full_path(True)
    val_dataset.load_full_path(True)
    test_dataset.load_full_path(True)
    train_dataset.load_from_h5(True)
    val_dataset.load_from_h5(True)
    test_dataset.load_from_h5(True)
    
elif args.dataset == 'rcc':
    args.n_classes = 3
    prompt_file = "models/prompts/rcc_prompts_all_per_class.json"
    label_map = {"KICH": 0, "KIRC": 1, "KIRP": 2}
    prompt_file_ext = "models/prompts/rcc_prompts_w4normal.json"
    label_map_ext = {"KICH": 0, "KIRC": 1, "KIRP": 2, "Stroma": 3, "Inflammation": 4, "Vascular": 5, "Necrosis": 6}

    idx_to_class = {v:k for k,v in label_map.items()}
    with open(prompt_file) as f:
        prompts = json.load(f)['0']
    classnames = prompts['classnames']
    templates = prompts['templates']
    n_classes = len(classnames)
    classnames_text = [classnames[str(idx_to_class[idx])] for idx in range(n_classes)]
    conch_temperature = 56.3477
    zeroshot_weights_rcc_conch_path = "models/classifier_weights/weights_rcc_conch.pt"
    if os.path.exists(zeroshot_weights_rcc_conch_path) and args.load_weight:
        zeroshot_weights = torch.load(zeroshot_weights_rcc_conch_path)
    else:
        zeroshot_weights = zero_shot_classifier(conch_model, classnames_text, templates, device=device)
        torch.save(zeroshot_weights, zeroshot_weights_rcc_conch_path)

    idx_to_class_ext = {v:k for k,v in label_map_ext.items()}
    with open(prompt_file_ext) as f:
        prompts_ext = json.load(f)['0']
    classnames_ext = prompts_ext['classnames']
    classnames_text_ext = [classnames_ext[str(idx_to_class_ext[idx])] for idx in range(len(classnames_ext))]
    zeroshot_weights_rcc_ext_conch_path = "models/classifier_weights/weights_rcc_ext_conch.pt"
    if os.path.exists(zeroshot_weights_rcc_ext_conch_path) and args.load_weight:
        zeroshot_weights_ext = torch.load(zeroshot_weights_rcc_ext_conch_path)
    else:
        zeroshot_weights_ext = zero_shot_classifier(conch_model, classnames_text_ext, templates, device=device)
        torch.save(zeroshot_weights_ext, zeroshot_weights_rcc_ext_conch_path)

    print("zershot weights shape: ", zeroshot_weights.shape)
    print("zershot weights_ext shape: ", zeroshot_weights_ext.shape)

    if args.pretrain == 'conch':
        data_dir = os.path.join('data/rcc', 'merge_features_conch')
    else:
        raise NotImplementedError
    
    dataset = Generic_MIL_Dataset(csv_path='dataset_csv/rcc.csv',
                                data_dir=data_dir,
                                shuffle=False,
                                seed=1,
                                print_info=True,
                                label_dict= {"KICH": 0, "KIRC": 1, "KIRP": 2},
                                patient_strat=False,
                                ignore=[])
    dataset.load_from_h5(True)
    dataset.load_full_path(True)

    train_dataset, val_dataset, test_dataset = dataset.return_splits(from_id=False, csv_path=f'splits/rcc_fewshot/{args.shot}shots/splits_{args.fold}.csv', repeat_num=int(args.shot)*3)
    train_dataset.load_full_path(True)
    val_dataset.load_full_path(True)
    test_dataset.load_full_path(True)
    train_dataset.load_from_h5(True)
    val_dataset.load_from_h5(True)
    test_dataset.load_from_h5(True)

# feats, lbl, coords, full_path = dataset[0]
loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=1)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)

# <<<<<<<<<<<<<<<<<< data preparation <<<<<<<<<<<<<<<<<<<<<<<<


# >>>>>>>>>>>>>>>>>>> model initialization >>>>>>>>>>>>>>>>>>>
class senet(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(senet, self).__init__()
        self.hidden_dim = 64
        self.model = nn.Sequential(
            nn.Linear(in_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, out_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.model(x)
        return out
    
# if args.ablation_study != 'none':
model = senet(512, 4).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

# <<<<<<<<<<<<<<<<<< model initialization <<<<<<<<<<<<<<<<<<<<

# >>>>>>>>>>>>>>>>>>>> train & eval >>>>>>>>>>>>>>>>>>>>>>>>>>

def slide_process(feat, zeroshot_weights, zeroshot_weights_ext, 
                  n_classes, topj=10, random_mask=False,
                  discard_classifiers=[]) -> dict:
    device = zeroshot_weights.device
    feat = feat.to(device)

    # mask select half of the patches
    if random_mask:
        mask = torch.rand(feat.size(0)) > 0.5
        feat = feat[mask]

    zeroshot_weights_ext = zeroshot_weights_ext.to(device)
    
    selected_index = set()
    logits = feat @ zeroshot_weights
    logits_ext = feat @ zeroshot_weights_ext
    # bottomk = topj*2
    topj = [topj]

    if "topk" not in discard_classifiers:
        indices1 = index_topj_classifier(logits, topj)
        selected_index.update(indices1.flatten().tolist())
    if "delta_softmax" not in discard_classifiers:
        indices2 = index_delta_softmax_classifier(logits, topj)
        selected_index.update(indices2.flatten().tolist())
    if "delta_diff" not in discard_classifiers:
        indices3 = index_delta_diff_classifier(logits, topj)
        selected_index.update(indices3.flatten().tolist())
    if "bottomk" not in discard_classifiers:
        indices4 = index_bottomk_irrel_classifier(logits_ext, topj, n_classes)
        selected_index.update(indices4.flatten().tolist())

    selected_index = sorted(list(selected_index))
    selected_feat = feat[selected_index]
    selected_logits = selected_feat @ zeroshot_weights
    selected_logits_ext = selected_feat @ zeroshot_weights_ext

    logits_top_classifier = selected_logits
    logits_delta_softmax_classifier = selected_logits.softmax(dim=1)
    top1_logits_row = torch.topk(selected_logits, 1, dim=1)[0][:, 0]
    top2_logits_row = torch.topk(selected_logits, 2, dim=1)[0][:, 1]
    diff_logits = torch.abs(top1_logits_row - top2_logits_row)
    logits_delta_diff_classifier = torch.stack([diff_logits] * selected_logits.size(1), dim=1)
    bg_logits = selected_logits_ext[:, n_classes:].max(dim=1)[0]
    logits_bottomk_irrel_classifier = torch.stack([bg_logits] * selected_logits.size(1), dim=1)

    return {
        "selected_index": selected_index,
        "selected_feat": selected_feat,
        "logits_top_classifier": logits_top_classifier,
        "logits_delta_softmax_classifier": logits_delta_softmax_classifier,
        "logits_delta_diff_classifier": logits_delta_diff_classifier,
        "logits_bottomk_irrel_classifier": logits_bottomk_irrel_classifier,
    }


def train(model, train_loader, optimizer, device, args):
    model.train()
    for idx, data in enumerate(tqdm(train_loader, bar_format="{l_bar}{bar:10}{r_bar}", disable=args.disable_tqdm)):
        feats, lbl, coords, full_path = data
        feats = feats.squeeze(0).to(device)
        lbl = lbl.to(device)
        coords = coords.squeeze(0)
        full_path = full_path[0]
        slide_results = slide_process(feats, zeroshot_weights, zeroshot_weights_ext, 
                                      n_classes=args.n_classes, topj=args.topj, 
                                      random_mask=True, discard_classifiers=args.discard_classifiers)

        weights = model(slide_results['selected_feat'])
        logits_topk = weights[:, 0].unsqueeze(1) * slide_results['logits_top_classifier']
        logits_delta_softmax = weights[:, 1].unsqueeze(1) * slide_results['logits_delta_softmax_classifier']
        logits_delta_diff = weights[:, 2].unsqueeze(1) * slide_results['logits_delta_diff_classifier']
        logits_bottomk = weights[:, 3].unsqueeze(1) * slide_results['logits_bottomk_irrel_classifier']
        final_logits = torch.zeros_like(logits_topk)
        if "topk" not in args.discard_classifiers:
            final_logits += logits_topk
        if "delta_softmax" not in args.discard_classifiers:
            final_logits += logits_delta_softmax
        if "delta_diff" not in args.discard_classifiers:
            final_logits += logits_delta_diff
        if "bottomk" not in args.discard_classifiers:
            final_logits += logits_bottomk

        logits = topj_pooling(final_logits, [args.topk])[1][args.topk]
        loss = F.cross_entropy(logits, lbl)
        # print("loss: ", loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def zs_evaluation(loader, device, args, pooling_func=topj_pooling):
    test_loss = 0
    correct = 0
    logits_all = []
    lbl_all = []
    with torch.no_grad():
        real_len = loader.dataset.real_len()
        set_len = loader.dataset.repeat_num
        loader.dataset.repeat_num = real_len
        for idx, data in enumerate(tqdm(loader, bar_format="{l_bar}{bar:10}{r_bar}", disable=args.disable_tqdm)):
            feats, lbl, coords, full_path = data
            feats = feats.squeeze(0).to(device)
            lbl = lbl.to(device)
            coords = coords.squeeze(0)
            full_path = full_path[0]
            final_logits = feats @ zeroshot_weights
            final_logits_ext = feats @ zeroshot_weights_ext
            if pooling_func in [topj_pooling, delta_softmax_classifier_pooling, delta_diff_classifier_pooling]:
                logits = pooling_func(final_logits, [args.topk])[1][args.topk]
            else:   # bottomk_irrel_classifier_pooling
                logits = pooling_func(final_logits_ext, [args.topk], coords_list=args.n_classes)[1][args.topk]
            test_loss += F.cross_entropy(logits, lbl).item()
            pred = logits.argmax(dim=1, keepdim=True)
            correct += pred.eq(lbl.view_as(pred)).sum().item()
            logits_all.append(logits)
            lbl_all.append(lbl)
        loader.dataset.repeat_num = set_len
    test_loss /= len(loader.dataset)
    logits_all = torch.cat(logits_all, dim=0)
    lbl_all = torch.cat(lbl_all, dim=0)
    if args.pretrain == 'conch':
        temperature = 56.3477
    else:
        raise NotImplementedError
    probs = F.softmax(logits_all*temperature, dim=1)
    n_classes = probs.shape[1]
    if n_classes == 2:
        class_probs = probs[:,1]
        roc_kwargs = {}
    else:
        class_probs = probs
        roc_kwargs = {'multi_class': 'ovo', 'average': 'macro'}
    auc = roc_auc_score(lbl_all.cpu().numpy(), class_probs.cpu().numpy(), **roc_kwargs)
    eval_dict = {
        "loss": test_loss,
        "acc": correct / real_len,
        "auc": auc
    }
    return eval_dict

def evaluation(model, loader, device, args):
    model.eval()
    test_loss = 0
    correct = 0
    logits_all = []
    lbl_all = []
    with torch.no_grad():
        real_len = loader.dataset.real_len()
        set_len = len(loader.dataset)
        loader.dataset.repeat_num = real_len
        for idx, data in enumerate(tqdm(loader, bar_format="{l_bar}{bar:10}{r_bar}", disable=args.disable_tqdm)):
            feats, lbl, coords, full_path = data
            feats = feats.squeeze(0).to(device)
            lbl = lbl.to(device)
            coords = coords.squeeze(0)
            full_path = full_path[0]
            slide_results = slide_process(feats, zeroshot_weights, zeroshot_weights_ext, 
                                          n_classes=args.n_classes, topj=args.topj,
                                          discard_classifiers=args.discard_classifiers)
            weights = model(slide_results['selected_feat'])
            logits_topk = weights[:, 0].unsqueeze(1) * slide_results['logits_top_classifier']
            logits_delta_softmax = weights[:, 1].unsqueeze(1) * slide_results['logits_delta_softmax_classifier']
            logits_delta_diff = weights[:, 2].unsqueeze(1) * slide_results['logits_delta_diff_classifier']
            logits_bottomk = weights[:, 3].unsqueeze(1) * slide_results['logits_bottomk_irrel_classifier']
            final_logits = logits_topk
            if "delta_softmax" not in args.discard_classifiers:
                final_logits += logits_delta_softmax
            if "delta_diff" not in args.discard_classifiers:
                final_logits += logits_delta_diff
            if "delta_bottomk" not in args.discard_classifiers:
                final_logits += logits_bottomk
            logits = topj_pooling(final_logits, [args.topk])[1][args.topk]
            test_loss += F.cross_entropy(logits, lbl).item()
            pred = logits.argmax(dim=1, keepdim=True)
            correct += pred.eq(lbl.view_as(pred)).sum().item()
            logits_all.append(logits)
            lbl_all.append(lbl)
        loader.dataset.repeat_num = set_len
            
    test_loss /= len(loader.dataset)
    logits_all = torch.cat(logits_all, dim=0)
    lbl_all = torch.cat(lbl_all, dim=0)
    if args.pretrain == 'conch':
        temperature = 56.3477
    probs = F.softmax(logits_all*temperature, dim=1)
    n_classes = probs.shape[1]
    if n_classes == 2:
        class_probs = probs[:,1]
        roc_kwargs = {}
    else:
        class_probs = probs
        roc_kwargs = {'multi_class': 'ovo', 'average': 'macro'}
    auc = roc_auc_score(lbl_all.cpu().numpy(), class_probs.cpu().numpy(), **roc_kwargs)
    eval_dict = {
        "loss": test_loss,
        "acc": correct / real_len,
        "auc": auc
    }
    return eval_dict


def ablation_evaluation(loader, device, args):
    test_loss = 0
    correct = 0
    logits_all = []
    lbl_all = []
    real_len = loader.dataset.real_len()
    set_len = len(loader.dataset)
    loader.dataset.repeat_num = real_len
    for idx, data in enumerate(tqdm(loader, bar_format="{l_bar}{bar:10}{r_bar}", disable=args.disable_tqdm)):
        feats, lbl, coords, full_path = data
        feats = feats.squeeze(0).to(device)
        lbl = lbl.to(device)
        coords = coords.squeeze(0)
        full_path = full_path[0]
        slide_results = slide_process(feats, zeroshot_weights, zeroshot_weights_ext, n_classes=args.n_classes, topj=args.topj)
        if args.ablation_study == 'avg':
            final_logits = 0.25 * slide_results['logits_top_classifier'] + \
                            0.25 * slide_results['logits_delta_softmax_classifier'] + \
                            0.25 * slide_results['logits_delta_diff_classifier'] + \
                            0.25 * slide_results['logits_bottomk_irrel_classifier']
        elif args.ablation_study == 'sum':
            final_logits = slide_results['logits_top_classifier'] + \
                            slide_results['logits_delta_softmax_classifier'] + \
                            slide_results['logits_delta_diff_classifier'] + \
                            slide_results['logits_bottomk_irrel_classifier']
        elif args.ablation_study == 'max':
            max_logits = torch.stack([slide_results['logits_top_classifier'],
                                      slide_results['logits_delta_softmax_classifier'],
                                      slide_results['logits_delta_diff_classifier'],
                                      slide_results['logits_bottomk_irrel_classifier']], dim=0)
            final_logits = max_logits.max(dim=0)[0]

        logits = topj_pooling(final_logits, [args.topk])[1][args.topk]
        test_loss += F.cross_entropy(logits, lbl).item()
        pred = logits.argmax(dim=1, keepdim=True)
        correct += pred.eq(lbl.view_as(pred)).sum().item()
        logits_all.append(logits)
        lbl_all.append(lbl)
    loader.dataset.repeat_num = set_len

    test_loss /= len(loader.dataset)
    logits_all = torch.cat(logits_all, dim=0)
    lbl_all = torch.cat(lbl_all, dim=0)
    if args.pretrain == 'conch':
        temperature = 56.3477
    probs = F.softmax(logits_all*temperature, dim=1)
    n_classes = probs.shape[1]
    if n_classes == 2:
        class_probs = probs[:,1]
        roc_kwargs = {}
    else:
        class_probs = probs
        roc_kwargs = {'multi_class': 'ovo', 'average': 'macro'}
    auc = roc_auc_score(lbl_all.cpu().numpy(), class_probs.cpu().numpy(), **roc_kwargs)
    eval_dict = {
        "loss": test_loss,
        "acc": correct / real_len,
        "auc": auc
    }
    return eval_dict

# <<<<<<<<<<<<<<<<<<< train & eval <<<<<<<<<<<<<<<<<<<<<<<<<

def main(args):
    if args.ablation_study != 'none':
        ablation_eval_dict = ablation_evaluation(test_loader, device, args)
        print(f"Ablation Study: {args.ablation_study}, Test: {ablation_eval_dict}")
        with open(os.path.join(args.result_dir, f"ablation_results_{args.ablation_study}_shot_{args.shot}_fold_{args.fold}.json"), 'w') as f:
            json.dump(ablation_eval_dict, f, indent=4)
        return

    zs_train, zs_val, zs_test = -1, -1, -1
    if args.check_zeroshot:
        zs_train = zs_evaluation(train_loader, device, args)
        zs_val = zs_evaluation(val_loader, device, args)
        zs_test = zs_evaluation(test_loader, device, args)
        print(f"Zero-shot Train: {zs_train}, Val: {zs_val}, Test: {zs_test}")
        with open(os.path.join(args.result_dir, f"zs_results_shot_{args.shot}_fold_{args.fold}.json"), 'w') as f:
            json.dump({
                "zs_train": zs_train,
                "zs_val": zs_val,
                "zs_test": zs_test
            }, f, indent=4)

    best_val = 0
    test_at_best_val = 0
    test_acc_at_best_val = 0
    best_epoch = 0
    num_epoch = 25
    for epoch in range(num_epoch):
        print("Epoch: ", epoch)
        train(model, train_loader, optimizer, device, args)
        train_eval = evaluation(model, train_loader, device, args)
        val_eval = evaluation(model, val_loader, device, args)
        #REVIEW - temporary trade off for speed up
        if val_eval['auc'] > best_val:
            test_eval = evaluation(model, test_loader, device, args)
            print(f"Epoch: {epoch}, Train: {train_eval}, Val: {val_eval}, Test: {test_eval}")
        else:
            print(f"Epoch: {epoch}, Train: {train_eval}, Val: {val_eval}")
        if val_eval['auc'] > best_val:
            best_val = val_eval['auc']
            test_at_best_val = test_eval['auc']
            test_acc_at_best_val = test_eval['acc']
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(args.result_dir, f"best_model_shot_{args.shot}_fold_{args.fold}.pt"))
    print(f"Zero-shot Train: {zs_train}, Val: {zs_val}, Test: {zs_test}")
    print(f"Best Val: {best_val}, Test at Best Val: {test_at_best_val}, Test acc: {test_acc_at_best_val}, Best Epoch: {best_epoch}")
    results = {
        "zero_shot_train": zs_train,
        "zero_shot_val": zs_val,
        "zero_shot_test": zs_test,
        "best_val": best_val,
        "test_at_best_val": test_at_best_val,
        "test_acc_at_best_val": test_acc_at_best_val,
        "best_epoch": best_epoch,
        "best_model_path": os.path.join(args.result_dir, f"best_model_shot_{args.shot}_fold_{args.fold}.pt")
    }
    with open(os.path.join(args.result_dir, f"best_results_shot_{args.shot}_fold_{args.fold}.json"), 'w') as f:
        json.dump(results, f, indent=4)
    
    print("\nEnd training.")

if __name__ == "__main__":
    main(args)

# python main_moc.py --summary --summary_dir /home/txiang/pathology/CLAM/results/moc_eval_summary

# CUDA_VISIBLE_DEVICES=1 python main_moc.py --fold 0 --shot 8 --topj 400 --topk 10 --dataset nsclc
# CUDA_VISIBLE_DEVICES=7 python main_moc.py --fold 0 --shot 8 --topj 400 --topk 10 --dataset rcc
# python main_moc.py --summary --summary_dir results/moc_train/nsclc
