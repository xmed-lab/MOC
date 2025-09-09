import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.metrics import roc_auc_score
import copy
from torch.cuda import amp
from tqdm import tqdm, trange

from utils.utils import *
from utils.zeroshot_utils import zero_shot_classifier
from datasets.dataset_generic import save_splits
from models.model_conch import conch_coca
from models.model_conch import conch_coca, conch_lora, Conch_LoRA


def topj_pooling(logits, topj=10):
        """
        logits: N x C logit for each patch
        topj: tuple of the top number of patches to use for pooling
        """
        # Sums logits across topj patches for each class, to get class prediction for each topj
        maxj = min(topj, logits.size(0))
        values, _ = logits.topk(maxj, 0, True, True)
        pooled_logits = values[:min(topj, maxj)].mean(dim=0, keepdim=True)
        return pooled_logits

def update_sorted_queue(queue, new_item, max_length):
    queue.append(new_item)
    # queue.sort(key=lambda x: x[1][0][0], reverse=True)
    queue.sort(key=lambda x: x[1].max(), reverse=True)
    if len(queue) > max_length:
        queue.pop(-1)
    return queue

def train_wsi_lora(datasets, cur, args):
    print('\nTraining Fold {}!'.format(cur))
    writer_dir = os.path.join(args.results_dir, str(cur))
    if not os.path.isdir(writer_dir):
        os.mkdir(writer_dir)
    if args.log_data:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(writer_dir, flush_secs=15)
    else:
        writer = None
    print('\nInit train/val/test splits...', end=' ')
    train_split, val_split, test_split = datasets
    # save_splits(datasets, ['train', 'val', 'test'], os.path.join(args.results_dir, 'splits_{}.csv'.format(cur)))
    print('Done!')
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))
    print("Testing on {} samples".format(len(test_split)))

    print('\nInit loss function...', end=' ')
    loss_fn = nn.CrossEntropyLoss()
    print('Done!')

    print("\nInit zeroshot classifier weights...", end=' ')
    # check if zeroshot weights are available
    classifier_weights_path = "/home/txiang/pathology/CLAM/models/classifier_weights"
    checkpoint_path = "/home/txiang/pathology/CLAM/models/conch_checkpoint.bin"
    conch_temperature = 56.3477
    if args.task == 'task_1_tumor_vs_normal':
        zeroshot_weights_path = os.path.join(classifier_weights_path, 'camelyon_tumor_vs_normal.pt')
    else:
        zeroshot_weights_path = os.path.join(classifier_weights_path, 'nsclc_luad_lusc.pt')
        # zeroshot_weights_path = os.path.join(classifier_weights_path, 'nsclc_luad_lusc_better.pt')
    # if os.path.exists(zeroshot_weights_path) and not args.force_weights_update:
    if os.path.exists(zeroshot_weights_path):
        print("exist weights, loading...")
        zeroshot_weights = torch.load(zeroshot_weights_path)
    # if not availabel, generate zeroshot weights
    else:
        print("not exist weights, generating...")
        conch_model = conch_coca(checkpoint_path=checkpoint_path)
        conch_model.eval()
        conch_model.to(device)
        if args.task == 'task_1_tumor_vs_normal':
            label_map = {'NORMAL': 0, 'TUMOR': 1}
            idx_to_class = {v:k for k,v in label_map.items()}
            prompt_file = '/home/txiang/pathology/CLAM/models/prompts/camelyon_prompts_all_per_class.json'
            save_path = os.path.join(classifier_weights_path, 'camelyon_tumor_vs_normal.pt')
        else:
            label_map = {'LUAD': 0, 'LUSC': 1}
            idx_to_class = {v:k for k,v in label_map.items()}
            # prompt_file = '/home/txiang/pathology/CLAM/models/prompts/nsclc_prompts_all_per_class_worse.json'
            prompt_file = '/home/txiang/pathology/CLAM/models/prompts/nsclc_prompts_all_per_class.json'
            save_path = os.path.join(classifier_weights_path, 'nsclc_luad_lusc_better.pt')
        with open(prompt_file) as f:
            prompts = json.load(f)['0']
        classnames = prompts['classnames']
        templates = prompts['templates']
        n_classes = len(classnames)
        classnames_text = [classnames[str(idx_to_class[idx])] for idx in range(n_classes)]
        zeroshot_weights = zero_shot_classifier(conch_model, classnames_text, templates, device=device)
        torch.save(zeroshot_weights, save_path)
        conch_model.to("cpu")
        del conch_model
        torch.cuda.empty_cache()
    print("Done!")

    print('\nInit Model...', end=' ')
    model = Conch_LoRA(checkpoint_path=checkpoint_path, r=4, lora_cnt=None)
    model.to(device)
    print('Done!')
    # print_network(model)

    print('\nInit optimizer...', end=' ')
    optimizer = get_optim(model, args)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 20)
    print('Done!')

    print('\nInit dataloaders...', end=' ')
    train_loader = DataLoader(train_split, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_split, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_split, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    print('Done!')

    @torch.no_grad()
    def val_fn(model, loader, require_patient=False):
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        logits_all, targets_all = [], []
        if require_patient:
            slide_ids = loader.dataset.slide_data['slide_id'] #FIXME
            patient_results = {}
        
        for batch_idx, (data, target) in enumerate(tqdm(loader, desc="eval", bar_format="{l_bar}{bar:10}{r_bar}")):
            # if batch_idx == 2:
            #     break
            data = data.squeeze()
            target = torch.LongTensor(target).to(device)

            logits_queue = []
            max_queue_length = 10
                        
            minibatch_size = 8
            # for i in tqdm(range(0, len(data), minibatch_size), desc="hi", bar_format="{desc}{l_bar}{bar:10}{r_bar}"):
            for i in range(0, len(data), minibatch_size):
                datai = data[i:i+minibatch_size]
                datai = datai.to(device)
                feati = model(datai)
                feati = feati / feati.norm(dim=-1, keepdim=True)
                logiti = feati @ zeroshot_weights
                logiti = F.softmax(logiti, dim=1)
                for j in range(len(datai)):
                    update_sorted_queue(logits_queue, (i+j, logiti[j].unsqueeze(0)), max_queue_length)
            pooled_logits = torch.cat([logit[1] for logit in logits_queue], dim=0).mean(dim=0, keepdim=True)

            loss = loss_fn(pooled_logits, target)
            preds = pooled_logits.argmax(dim=1)
            acc = (preds == target).float().mean()
            logits_all.append(pooled_logits.cpu().numpy())
            targets_all.append(target.cpu().numpy())
            val_loss += loss.item()
            val_acc += acc.item()
            if require_patient:
                slide_id = slide_ids.iloc[batch_idx]
                patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': F.softmax(pooled_logits*conch_temperature, dim=1).cpu().numpy(), 'label': target.cpu().numpy()}})
        val_loss /= len(loader)
        val_acc /= len(loader)
        # val_loss /= 2
        # val_acc /= 2
        probs_all = F.softmax(torch.from_numpy(np.concatenate(logits_all, axis=0)) * conch_temperature, dim=1).numpy()
        targets_all = np.concatenate(targets_all, axis=0)
        val_auc = roc_auc_score(targets_all, probs_all[:,1])
        return val_loss, val_acc, val_auc
    
    # print("\nCheck with MI-Zero...")
    # val_results, _ = run_mizero_simple(model, zeroshot_weights, val_loader, device, dump_results=True, metrics=['acc', 'bacc', 'roc_auc'], topj=[10])
    # test_results, _ = run_mizero_simple(model, zeroshot_weights, test_loader, device, dump_results=True, metrics=['acc', 'bacc', 'roc_auc'], topj=[10])
    # val_acc, val_auc = val_results['acc'][10], val_results['roc_auc'][10]
    # test_acc, test_auc = test_results['acc'][10], test_results['roc_auc'][10]
    # print('Val AUC: {:.6f}\tVal Acc: {:.6f}'.format(val_auc, val_acc))
    # print('Test AUC: {:.6f}\tTest Acc: {:.6f}'.format(test_auc, test_acc))
    # print("Done!")
    
    # print("\nConch zero-shot results...")
    # val_loss, val_acc, val_auc = val_fn(model, val_loader)
    # test_loss, test_acc, test_auc = val_fn(model, test_loader)
    # print('Val AUC: {:.6f}\tVal Acc: {:.6f}\tVal Loss: {:.6f}'.format(val_auc, val_acc, val_loss))
    # print('Test AUC: {:.6f}\tTest Acc: {:.6f}\tTest Loss: {:.6f}'.format(test_auc, test_acc, test_loss))
    # zs_val_auc, zs_val_acc, zs_test_auc, zs_test_acc = val_auc, val_acc, test_auc, test_acc
    # print("Done!")

    print("\nStart training...")
    best_epoch = 0
    best_val_auc = 0.0
    best_val_acc = 0.0
    best_test_auc = 0.0
    best_test_acc = 0.0

    print("Init results...")
    train_loss, train_acc, train_auc = val_fn(model, train_loader)
    print('Train AUC: {:.6f}\tTrain Acc: {:.6f}\tTrain Loss: {:.6f}'.format(train_auc, train_acc, train_loss))
    val_loss, val_acc, val_auc = val_fn(model, val_loader)
    print('Val AUC: {:.6f}\tVal Acc: {:.6f}\tVal Loss: {:.6f}'.format(val_auc, val_acc, val_loss))
    test_loss, test_acc, test_auc = val_fn(model, test_loader)
    print('Test AUC: {:.6f}\tTest Acc: {:.6f}\tTest Loss: {:.6f}'.format(test_auc, test_acc, test_loss))
    zs_val_auc, zs_val_acc, zs_test_auc, zs_test_acc = val_auc, val_acc, test_auc, test_acc
    # zs_val_auc, zs_val_acc, zs_test_auc, zs_test_acc = 0,0,0,0

    amp_scaler = amp.GradScaler()
    for epoch in range(args.max_epochs):
        model.train()
        print('\nEpoch {}!'.format(epoch))
        train_loss = 0.0
        train_acc = 0.0
        for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc="train", bar_format="{l_bar}{bar:10}{r_bar}")):
            # data (N,3,224,224) target (1,)
            optimizer.zero_grad()

            # with amp.autocast():  # fp16 seems give worse results
            if True:
                data = data.squeeze()
                target = torch.LongTensor(target).to(device)

                logits_queue = []
                max_queue_length = 20                      
                minibatch_size = 8
                # for i, datai in enumerate(tqdm(data, desc="hi", bar_format="{desc}{l_bar}{bar:10}{r_bar}")):
                # for i in tqdm(range(0, len(data), minibatch_size), desc="hi", bar_format="{desc}{l_bar}{bar:10}{r_bar}"):
                for i in range(0, len(data), minibatch_size):
                    datai = data[i:i+minibatch_size]
                    datai = datai.to(device)
                    # datai = datai.unsqueeze(0).to(device)
                    feati = model(datai)
                    feati = feati / feati.norm(dim=-1, keepdim=True)
                    logiti = feati @ zeroshot_weights
                    for j in range(len(datai)):
                        update_sorted_queue(logits_queue, (i+j, logiti[j].unsqueeze(0)), max_queue_length)
                    # update_queue(logits_queue, (i, logiti), max_queue_length)
                pooled_logits = torch.cat([logit[1] for logit in logits_queue], dim=0).mean(dim=0, keepdim=True)
                loss = loss_fn(pooled_logits, target)
            # amp_scaler.scale(loss).backward()
            # amp_scaler.step(optimizer)
            # amp_scaler.update()
            loss.backward()
            optimizer.step()
            preds = pooled_logits.argmax(dim=1)
            acc = (preds == target).float().mean()
            train_loss += loss.item()
            train_acc += acc.item()

            if (batch_idx + 1) % 20 == 0:
                print('Train Epoch: {}\t[{:.0f}%]\tLoss: {:.6f}'.format(
                    epoch, 100. * batch_idx / len(train_loader), loss.item()))
                if writer is not None:
                    writer.add_scalar('train/loss', loss.item(), epoch * len(train_loader) + batch_idx)
                    writer.add_scalar('train/acc', acc.item(), epoch * len(train_loader) + batch_idx)

        train_loss /= len(train_loader)
        train_acc /= len(train_loader)
        print('Train Epoch: {} \tLoss: {:.6f}\tAcc: {:.6f}'.format(epoch, train_loss, train_acc))

        val_loss, val_acc, val_auc = val_fn(model, val_loader)
        print('Val AUC: {:.6f}\tVal Acc: {:.6f}\tVal Loss: {:.6f}'.format(val_auc, val_acc, val_loss))
        test_loss, test_acc, test_auc = val_fn(model, test_loader)
        print('Test AUC: {:.6f}\tTest Acc: {:.6f}\tTest Loss: {:.6f}'.format(test_auc, test_acc, test_loss))
        # test_auc, test_acc = 0.0, 0.0

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            best_val_acc = val_acc
            best_test_auc = test_auc
            best_test_acc = test_acc
            print('Saving best model...')
            torch.save(model.state_dict(), os.path.join(args.results_dir, 'best_model.pth'))
    
    print('Best Val AUC: {:.6f}\tBest Val Acc: {:.6f}\tBest Test AUC: {:.6f}\tBest Test Acc: {:.6f}'.format(\
        best_val_auc, best_val_acc, best_test_auc, best_test_acc))
    return (best_val_auc, best_val_acc, best_test_auc, best_test_acc), (zs_val_auc, zs_val_acc, zs_test_auc, zs_test_acc) 
    # return best_val_auc, best_val_acc, best_test_auc, best_test_acc
