import torch
import torch.nn as nn
import numpy as np
from utils.utils import *
import os
import os.path as osp
from PIL import Image
import openslide
# from .model_clam import Attn_Net_Gated
from .model_conch import conch_lora

def init_adapter_weight(fewshot_ds):
    aux_features = []
    aux_labels = []
    with torch.no_grad():
        for i, (data, target) in enumerate(fewshot_ds):
            data = data.squeeze()
            data -= data.mean(dim=-1, keepdim=True)
            data /= data.norm(dim=-1, keepdim=True)
            target = torch.tensor(target, dtype=torch.long)
            aux_features.append(data)
            aux_labels.append(target)
        aux_features = torch.cat(aux_features, dim=0).mean(dim=0)
        aux_features /= aux_features.norm(dim=-1, keepdim=True)
        aux_labels = torch.tensor(aux_labels)
    return aux_features, aux_labels


def init_adapter_weight_c16wGT(fewshot_ds):
    # check with camelyon GT
    features = []
    labels = []
    fewshot_ds.load_from_h5(True)
    fewshot_ds.load_full_path(True)
    gt_path = "/home/txiang/pathology/CLAM/data/camelyon/camelyon16_mask"
    wsi_path = "/home/txiang/pathology/CLAM/data/camelyon_raw/CAMELYON16/images"
    with torch.no_grad():
        for i, (data, target, coords, full_path) in enumerate(fewshot_ds):
            patient_idx = osp.basename(full_path).split('.')[0]
            slide = openslide.OpenSlide(osp.join(wsi_path, patient_idx + '.tif'))
            wsi_size = slide.dimensions
            # print(patient_idx[:3])
            if patient_idx[:3] == "nor":
                features.append(data)
                labels.append(target)
                continue
            gt_mask_path = osp.join(gt_path, patient_idx + '.png')
            gt_mask = Image.open(gt_mask_path).convert('L')
            gt_arr = np.array(gt_mask).transpose(1, 0)
            # for easy handle
            gt_arr[gt_arr == 0] = 1
            gt_arr[gt_arr == 255] = 0
            data = data.squeeze()
            target = torch.tensor(target, dtype=torch.long)
            fg_feat = []
            for feat, coord in zip(data, coords):
                x, y = coord
                x_, y_ = x + 224, y + 224
                x, y = int(x / wsi_size[0] * gt_arr.shape[0]), int(y / wsi_size[1] * gt_arr.shape[1])
                x_, y_ = int(x_ / wsi_size[0] * gt_arr.shape[0]), int(y_ / wsi_size[1] * gt_arr.shape[1])
                patch_mask = gt_arr[x:x_, y:y_]
                if patch_mask.sum() > 0:
                    fg_feat.append(feat)
            fg_feat = np.stack(fg_feat)
            fg_feat = torch.from_numpy(fg_feat)
            fg_feat -= fg_feat.mean(dim=-1, keepdim=True)
            fg_feat /= fg_feat.norm(dim=-1, keepdim=True)
            features.append(fg_feat)
            labels.append(target)
        features = torch.cat(features, dim=0).mean(dim=0)
        features /= features.norm(dim=-1, keepdim=True)
        labels = torch.tensor(labels)
    fewshot_ds.load_from_h5(False)
    fewshot_ds.load_full_path(False)
    return features, labels

class Linear_Adapter(nn.Module):
    def __init__(self, feat_dim, class_num, sample_features=None):
        super().__init__()
        self.fc = nn.Linear(feat_dim, class_num, bias=False)
        # init
        if sample_features is not None:
            print('init adapter weight by training samples...')
            aux_features, aux_labels = sample_features[0], sample_features[1]
            aux_features = (aux_features - aux_features.mean()) / aux_features.std()
            init_weight = torch.zeros(feat_dim, class_num, device=aux_features.device) 
            for i in range(len(aux_labels)):
                init_weight[:, aux_labels[i]] = aux_features[i] + init_weight[:, aux_labels[i]]
            feat_per_class = len(aux_labels) / class_num
            init_weight = init_weight / feat_per_class
            self.fc.weight.data = init_weight.t()
        else:
            print('init adapter weight by random...')
            nn.init.kaiming_normal_(self.fc.weight, a=np.sqrt(5))
        
    def forward(self, feat):
        return self.fc(feat)


def uncertainty(logits, type, power):
    softmax_fun = nn.Softmax(dim=-1) # sofemax-norm to get probability distribution
    logits = softmax_fun(logits)
    detect_nan(logits, 'uncertainty input logits')
    if type == 'entropy':
        entropy = -torch.sum(logits * torch.log2(logits), dim=-1, keepdim=True) / torch.log2(torch.tensor(logits.shape[-1]).float())
        entropy =  (entropy * power).exp() 
        return entropy
    elif type == 'energy':
        max_values = logits.max(dim=-1, keepdim=True).values
        logits = logits - max_values
        tau = 2
        energy = tau * (torch.log(torch.sum(torch.exp(logits / tau), dim=-1, keepdim=True)) + max_values)
        return 1.0 / (energy ** power)
    elif type == 'max':
        max_values = logits.max(dim=-1, keepdim=True).values
        return 1.0 / (max_values) ** power
    elif type == 'max-min':
        diff = logits.max(dim=-1, keepdim=True).values - logits.min(dim=-1, keepdim=True).values
        return 1.0 / diff ** power 
    elif type == 'var':
        variance = torch.std(logits, dim=-1, keepdim=True)
        return variance
    elif type == 'top5':
        top2 = logits.topk(5, dim=-1).values
        confidence = (top2[:, 0] - top2[:, -1]).unsqueeze(-1)
        return 1.0 / (confidence) ** power
    elif type == 'moment':
        mu = torch.mean(logits, dim=-1, keepdim=True)
        sigma = torch.std(logits, dim=-1, keepdim=True)
        detect_nan(mu, 'mu')
        detect_nan(sigma, 'sigma')
        normalized_logits = (logits - mu) / sigma
        detect_nan(normalized_logits, 'normalized_logits')
        moment_4 = torch.mean(normalized_logits ** 4, dim=-1, keepdim=True)
        momnan = detect_nan(moment_4, 'moment_4')
        if momnan:
            print("hi")
        return 1 / ((moment_4 / 250) ** power)
        #return 1.5 - 0.12 * moment_4
        #return filp(moment_4)
        #return (- moment_4 * power).exp() 
    elif type == 'none':
        return torch.tensor(1.0)
    else:
        raise RuntimeError('Invalid uncertainty type.')


class Conch_CLIP_Ada(nn.Module):
    def __init__(self, c_in=512, reduction=4, num_classes=2, 
                 classifier_tensor=None, clip_ratio=0.1, topj=10):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True),
        )
        self.topj = topj
        # self.adapter = Linear_Adapter(c_in, num_classes, None)
        nn.init.kaiming_normal_(self.adapter[0].weight, a=np.sqrt(5))
        nn.init.kaiming_normal_(self.adapter[2].weight, a=np.sqrt(5))
        self.classifier = classifier_tensor
        self.num_classes = num_classes
        self.clip_ratio = clip_ratio
        # size = [512, 512, 384]
        # self.attention_net = nn.Sequential(
        #     nn.Linear(size[0], size[1]), 
        #     nn.ReLU(),
        #     nn.Dropout(0.25),
        #     Attn_Net_Gated(size[1], size[2], dropout=True, n_classes=1)
        # )
    
    def topj_pooling(self, logits, topj=10):
        """
        logits: N x C logit for each patch
        topj: tuple of the top number of patches to use for pooling
        """
        # Sums logits across topj patches for each class, to get class prediction for each topj
        # logits = F.softmax(logits, dim=-1)
        maxj = min(topj, logits.size(0))
        values, _ = logits.topk(maxj, 0, True, True)
        pooled_logits = values[:min(topj, maxj)].mean(dim=0, keepdim=True)
        return pooled_logits

    def forward(self, feat):
        adapted_features = self.adapter(feat)
        image_features = adapted_features * self.clip_ratio + feat * (1 - self.clip_ratio)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # logit_scale = 56.3477   # conch_coca.logit_scale.exp().item()
        logits_all = image_features @ self.classifier
        pooled_logits = self.topj_pooling(logits_all, topj=self.topj)
        return pooled_logits

        # abmil
        # A, h = self.attention_net(adapted_features)
        # A = torch.transpose(A, 0, 1)
        # A = F.softmax(A, dim=1)
        # M = torch.mm(A, h)
        # pooled_logits = M @ self.classifier
        # return pooled_logits
    
        # feat /= feat.norm(dim=-1, keepdim=True)
        # clip_logits = feat @ self.classifier
        # adapted_logits = self.adapter(feat)
        # logits_all = adapted_logits * self.clip_ratio + clip_logits * (1 - self.clip_ratio)
        # pooled_logits = self.topj_pooling(logits_all, topj=10)
        # return pooled_logits
    
    def forward_disable_ada(self, feat):
        image_features = feat / feat.norm(dim=-1, keepdim=True)
        logit_scale = 56.3477
        logits_all = image_features @ self.classifier
        pooled_logits = self.topj_pooling(logits_all, topj=self.topj)
        return pooled_logits
    

class Conch_TIP_Ada(nn.Module):
    def __init__(self, c_in=512, num_classes=2, classifier_tensor=None, sample_features=None, clip_ratio=0.1):
        super().__init__()
        self.adapter = Linear_Adapter(c_in, num_classes, sample_features)
        self.classifier = classifier_tensor
        self.num_classes = num_classes
        self.clip_ratio = clip_ratio
        
    def topj_pooling(self, logits, topj=10):
        """
        logits: N x C logit for each patch
        topj: tuple of the top number of patches to use for pooling
        """
        # Sums logits across topj patches for each class, to get class prediction for each topj
        maxj = min(topj, logits.size(0))
        values, _ = logits.topk(maxj, 0, True, True)
        pooled_logits = values[:min(topj, maxj)].mean(dim=0, keepdim=True)
        return pooled_logits

    def forward(self, feat):
        feat /= feat.norm(dim=-1, keepdim=True)
        clip_logits = feat @ self.classifier
        adapted_logits = self.adapter(feat)
        logits_all = adapted_logits * self.clip_ratio + clip_logits * (1 - self.clip_ratio)
        pooled_logits = self.topj_pooling(logits_all, topj=10)
        return pooled_logits
    
    def forward_disable_ada(self, feat):
        image_features = feat / feat.norm(dim=-1, keepdim=True)
        logit_scale = 56.3477
        logits_all = image_features @ self.classifier
        pooled_logits = self.topj_pooling(logits_all, topj=10)
        return pooled_logits


def load_balancing_loss_func(router_probs: torch.Tensor, expert_indices: torch.Tensor) -> float:
    r"""
    Computes auxiliary load balancing loss as in Switch Transformer - implemented in Pytorch.

    See Switch Transformer (https://arxiv.org/abs/2101.03961) for more details. This function implements the loss
    function presented in equations (4) - (6) of the paper. It aims at penalizing cases where the routing between
    experts is too unbalanced.

    Args:
        router_probs (`torch.Tensor`):
            Probability assigned to each expert per token. Shape: [batch_size, seqeunce_length, num_experts].
        expert_indices (`torch.Tensor`):
            Indices tensor of shape [batch_size, seqeunce_length] identifying the selected expert for a given token.

    Returns:
        The auxiliary loss.
    """
    num_experts = router_probs.shape[-1]

    # cast the expert indices to int64, otherwise one-hot encoding will fail
    if expert_indices.dtype != torch.int64:
        expert_indices = expert_indices.to(torch.int64)

    if len(expert_indices.shape) == 2:
        expert_indices = expert_indices.unsqueeze(2)

    expert_mask = torch.nn.functional.one_hot(expert_indices, num_experts)

    # For a given token, determine if it was routed to a given expert.
    expert_mask = torch.max(expert_mask, axis=-2).values

    # cast to float32 otherwise mean will fail
    expert_mask = expert_mask.to(torch.float32)
    tokens_per_group_and_expert = torch.mean(expert_mask, axis=-2)

    router_prob_per_group_and_expert = torch.mean(router_probs, axis=-2)
    return torch.mean(tokens_per_group_and_expert * router_prob_per_group_and_expert) * (num_experts**2)


class SwitchGate(nn.Module):
    def __init__(self, c_in=512, num_experts=3, 
                 use_switch_gate=False, use_balance_loss=False,
                 init_tensor=None, router_trainable=True):
        super().__init__()
        self.dim = c_in
        self.num_experts = num_experts
        self.gate = nn.Linear(c_in, num_experts, bias=False)
        self.use_switch_gate = use_switch_gate
        self.use_balance_loss = use_balance_loss
        self.init_tensor = init_tensor
        self.router_trainable = router_trainable
        if self.init_tensor is not None:
            self.gate.weight.data = self.init_tensor.transpose(0, 1)
        else:
            nn.init.kaiming_normal_(self.gate.weight, a=np.sqrt(5))
        if not self.router_trainable:
            self.gate.weight.requires_grad = False
        
    def forward(self, x):
        gate_scores = F.softmax(self.gate(x), dim=-1)

        if not self.use_switch_gate:
            return gate_scores, None
        
        top_k_scores, top_k_indices = gate_scores.topk(1, dim=-1)
        mask = torch.zeros_like(gate_scores).scatter_(-1, top_k_indices, 1)
        masked_gate_scores = gate_scores * mask
        # denominators = torch.sum(masked_gate_scores, dim=1, keepdim=True) + 1e-6
        # gate_scores = masked_gate_scores / denominators
        gate_scores = masked_gate_scores
        if not self.use_balance_loss:
            return gate_scores, None
        
        balance_loss = load_balancing_loss_func(gate_scores.unsqueeze(0), top_k_indices.squeeze().unsqueeze(0))
        return gate_scores, balance_loss


class Conch_MOE_CLIP_Ada(nn.Module):
    def __init__(self, c_in=512, reduction=4, ada_num=5, topj=10, 
                 classifier_tensor=None, clip_ratio=0.1, 
                 use_switch_gate=False, use_balance_loss=False,
                 router_tensor=None, router_trainable=True):
        super().__init__()
        assert ada_num > 1
        self.ada_num = ada_num
        self.topj = topj
        self.init_router = router_tensor
        self.use_switch_gate = use_switch_gate
        self.use_balance_loss = use_balance_loss
        self.router_trainable = router_trainable
        for i in range(ada_num):
            setattr(self, f'adapter_{i}', nn.Sequential(
                nn.Linear(c_in, c_in // reduction, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(c_in // reduction, c_in, bias=False),
                nn.ReLU(inplace=True),
            ))
        if self.init_router is None:
            self.ada_router = SwitchGate(c_in, ada_num, use_switch_gate, use_balance_loss, None, router_trainable)
        else:
            assert self.use_balance_loss == False
            assert self.use_switch_gate == False
            self.ada_router = SwitchGate(c_in, ada_num, False, False, self.init_router, router_trainable)
        self.classifier = classifier_tensor
        self.clip_ratio = clip_ratio / self.ada_num
        self.reset_adapter_weight()

    def reset_adapter_weight(self):
        for i in range(self.ada_num):
            nn.init.kaiming_normal_(getattr(self, f'adapter_{i}')[0].weight, a=np.sqrt(5))
            nn.init.kaiming_normal_(getattr(self, f'adapter_{i}')[2].weight, a=np.sqrt(5))
        # if self.ada_router is not None:   # init is already done in SwitchGate
        #     nn.init.kaiming_normal_(self.ada_router.gate.weight, a=np.sqrt(5))
    
    def topj_pooling(self, logits, topj=10):
        """
        logits: N x C logit for each patch
        topj: tuple of the top number of patches to use for pooling
        """
        # Sums logits across topj patches for each class, to get class prediction for each topj
        maxj = min(topj, logits.size(0))
        values, _ = logits.topk(maxj, 0, True, True)
        pooled_logits = values[:min(topj, maxj)].mean(dim=0, keepdim=True)
        return pooled_logits

    def forward(self, feat):
        feat = feat / feat.norm(dim=-1, keepdim=True)
        # if self.ada_router is not None:
        #     router_weight, balance_loss = self.ada_router(feat)
        # elif self.init_router is not None:
        #     router_weight = feat @ self.init_router
        #     router_weight = F.softmax(router_weight, dim=-1)
        router_weight, balance_loss = self.ada_router(feat) # balance_loss is None if not use_balance_loss
        router_weight = router_weight.unsqueeze(-2)  # (N,1,expert_num)
        expert_features_list = [getattr(self, f'adapter_{i}')(feat) for i in range(self.ada_num)]
        stacked_expert_features = torch.stack(expert_features_list, dim=-1)  # (N,C,expert_num)
        expert_features = torch.sum(stacked_expert_features * router_weight, dim=-1) # (N,C)
        expert_features = expert_features / expert_features.norm(dim=-1, keepdim=True)

        image_features = expert_features * self.clip_ratio + feat * (1 - self.clip_ratio)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logits_all = image_features @ self.classifier
        pooled_logits = self.topj_pooling(logits_all, topj=self.topj)

        if self.use_balance_loss:
            return pooled_logits, balance_loss
        return pooled_logits
    
    def forward_disable_ada(self, feat):
        image_features = feat / feat.norm(dim=-1, keepdim=True)
        logits_all = image_features @ self.classifier
        pooled_logits = self.topj_pooling(logits_all, topj=self.topj)
        return pooled_logits
    

class Conch_AMUVanilla_Ada(nn.Module):
    def __init__(self, c_in=512, c_in_aux=1024, reduction=4, 
                 num_classes=2, classifier_tensor=None, 
                 clip_ratio=0.1, aux_ratio=0.1,
                 uncertainty_type='none', uncertainty_power=1.0):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True),
        )
        # self.aux_adapter = nn.Sequential(
        #     nn.Linear(c_in_aux, c_in_aux // reduction, bias=False),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(c_in_aux // reduction, c_in, bias=False),
        #     nn.ReLU(inplace=True),
        # )
        # self.adapter = Linear_Adapter(c_in, num_classes, None)
        self.aux_adapter = Linear_Adapter(c_in_aux, num_classes, None)
        self.classifier = classifier_tensor
        self.num_classes = num_classes
        self.clip_ratio = clip_ratio
        self.aux_ratio = aux_ratio
        self.uncertainty_type = uncertainty_type
        self.uncertainty_power = uncertainty_power
        nn.init.kaiming_normal_(self.adapter[0].weight, a=np.sqrt(5))
        nn.init.kaiming_normal_(self.adapter[2].weight, a=np.sqrt(5))
        # nn.init.kaiming_normal_(self.aux_adapter[0].weight, a=np.sqrt(5))
        # nn.init.kaiming_normal_(self.aux_adapter[2].weight, a=np.sqrt(5))
    
    def topj_pooling(self, logits, topj=10):
        """
        logits: N x C logit for each patch
        topj: tuple of the top number of patches to use for pooling
        """
        # Sums logits across topj patches for each class, to get class prediction for each topj
        maxj = min(topj, logits.size(0))
        values, _ = logits.topk(maxj, 0, True, True)
        pooled_logits = values[:min(topj, maxj)].mean(dim=0, keepdim=True)
        return pooled_logits

    def forward(self, feat, aux_feat):
        feat = feat / feat.norm(dim=-1, keepdim=True)
        clip_logits = feat @ self.classifier

        adapted_features = self.adapter(feat).squeeze()
        adapted_features = adapted_features / adapted_features.norm(dim=-1, keepdim=True)
        adapted_logts = adapted_features @ self.classifier

        aux_feat = aux_feat / aux_feat.norm(dim=-1, keepdim=True)
        aux_adapted_logits = self.aux_adapter(aux_feat).squeeze()
        factor = uncertainty(clip_logits.float(), self.uncertainty_type, self.uncertainty_power)

        logits_all = adapted_logts * self.clip_ratio + aux_adapted_logits * self.aux_ratio * factor + clip_logits * (1 - self.clip_ratio - self.aux_ratio)
        pooled_logits = self.topj_pooling(logits_all, topj=10)
        pooled_logits_aux = self.topj_pooling(aux_adapted_logits, topj=10)
        return pooled_logits, pooled_logits_aux
        
        # feat /= feat.norm(dim=-1, keepdim=True)
        # aux_feat /= aux_feat.norm(dim=-1, keepdim=True)
        # adapted_features = self.adapter(feat).squeeze()
        # aux_adapted_features = self.aux_adapter(aux_feat).squeeze()
        # adapted_features = adapted_features / adapted_features.norm(dim=-1, keepdim=True)
        # aux_adapted_features = aux_adapted_features / aux_adapted_features.norm(dim=-1, keepdim=True)

        # clip_logits = feat @ self.classifier
        # adapted_logts = adapted_features @ self.classifier
        # aux_adapted_logts = aux_adapted_features @ self.classifier
        # factor = uncertainty(clip_logits.float(), self.uncertainty_type, self.uncertainty_power)
        # detect_nan(factor, 'factor')

        # logits_all = adapted_logts * self.clip_ratio + aux_adapted_logts * self.aux_ratio * factor + clip_logits * (1 - self.clip_ratio - self.aux_ratio)
        # pooled_logits = self.topj_pooling(logits_all, topj=10)
        # pooled_logits_aux = self.topj_pooling(aux_adapted_logts, topj=10)
        # return pooled_logits, pooled_logits_aux

        # image_features = adapted_features * self.clip_ratio + aux_adapted_features * self.aux_ratio + feat * (1 - self.clip_ratio - self.aux_ratio)
        # image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        # logit_scale = 56.3477   # conch_coca.logit_scale.exp().item()
        # logits_all = image_features @ self.classifier
        # pooled_logits = self.topj_pooling(logits_all, topj=10)
        # return pooled_logits
    
    def forward_disable_ada(self, feat, aux_feat):
        image_features = feat / feat.norm(dim=-1, keepdim=True)
        logit_scale = 56.3477
        logits_all = image_features @ self.classifier
        pooled_logits = self.topj_pooling(logits_all, topj=10)
        return pooled_logits


class Conch_AMUTip_Ada(nn.Module):
    def __init__(self, 
                 c_in=512, 
                 c_in_aux=1024, 
                 num_classes=2, 
                 classifier_tensor=None, 
                 sample_features=None, 
                 aux_sample_features=None, 
                 clip_ratio=0.1, 
                 aux_ratio=0.1):
        super().__init__()
        self.adapter = Linear_Adapter(c_in, num_classes, sample_features)
        self.aux_adapter = Linear_Adapter(c_in_aux, num_classes, aux_sample_features)
        self.classifier = classifier_tensor
        self.num_classes = num_classes
        self.clip_ratio = clip_ratio
        self.aux_ratio = aux_ratio
    
    def topj_pooling(self, logits, topj=10):
        """
        logits: N x C logit for each patch
        topj: tuple of the top number of patches to use for pooling
        """
        # Sums logits across topj patches for each class, to get class prediction for each topj
        maxj = min(topj, logits.size(0))
        values, _ = logits.topk(maxj, 0, True, True)
        pooled_logits = values[:min(topj, maxj)].mean(dim=0, keepdim=True)
        return pooled_logits

    def forward(self, feat, aux_feat):
        feat /= feat.norm(dim=-1, keepdim=True)
        aux_feat /= aux_feat.norm(dim=-1, keepdim=True)
        clip_logits = feat @ self.classifier
        adapted_logits = self.adapter(feat)
        aux_adapted_logits = self.aux_adapter(aux_feat)
        logits_all = adapted_logits * self.clip_ratio + aux_adapted_logits * self.aux_ratio + clip_logits * (1 - self.clip_ratio - self.aux_ratio)
        pooled_logits = self.topj_pooling(logits_all, topj=10)
        return pooled_logits
    
    def forward_disable_ada(self, feat, aux_feat):
        image_features = feat / feat.norm(dim=-1, keepdim=True)
        logit_scale = 56.3477
        logits_all = image_features @ self.classifier
        pooled_logits = self.topj_pooling(logits_all, topj=10)
        return pooled_logits

