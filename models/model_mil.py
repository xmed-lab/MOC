import torch
import torch.nn as nn
import torch.nn.functional as F
# from utils.utils import initialize_weights
import numpy as np
from nystrom_attention import NystromAttention

def initialize_weights(model):
    pass

class MIL_fc(nn.Module):
    def __init__(self, gate = True, size_arg="benchmark", dropout = False, n_classes = 2, top_k=1):
        super(MIL_fc, self).__init__()
        assert n_classes == 2
        self.size_dict = {"small": [1024, 512], "benchmark": [384, 512]}
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))

        fc.append(nn.Linear(size[1], n_classes))
        self.classifier= nn.Sequential(*fc)
        initialize_weights(self)
        self.top_k=top_k

    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classifier.to(device)

    def forward(self, h, return_features=False):
        if return_features:
            h = self.classifier.module[:3](h)
            logits = self.classifier.module[3](h)
        else:
            logits  = self.classifier(h) # K x 1

        # change top-k to mean
        y_probs = F.softmax(logits, dim = 1)

        top_instance_idx = torch.topk(y_probs[:, 1], self.top_k, dim=0)[1].view(1,)
        # top_instance_idx = torch.topk(y_probs[:, 1], self.top_k, dim=0)[1]
        top_instance = torch.index_select(logits, dim=0, index=top_instance_idx)

        Y_hat = torch.topk(top_instance, 1, dim = 1)[1]
        Y_prob = F.softmax(top_instance, dim = 1) 
        results_dict = {}

        if return_features:
            top_features = torch.index_select(h, dim=0, index=top_instance_idx)
            results_dict.update({'features': top_features})
        return top_instance, Y_prob, Y_hat, y_probs, results_dict


class MIL_fc_mc(nn.Module):
    def __init__(self, gate = True, size_arg = "small", dropout = False, n_classes = 2, top_k=1):
        super(MIL_fc_mc, self).__init__()
        assert n_classes > 2
        self.size_dict = {"small": [1024, 512]}
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))
        self.fc = nn.Sequential(*fc)

        self.classifiers = nn.ModuleList([nn.Linear(size[1], 1) for i in range(n_classes)])
        initialize_weights(self)
        self.top_k=top_k
        self.n_classes = n_classes
        assert self.top_k == 1

    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fc = self.fc.to(device)
        self.classifiers = self.classifiers.to(device)
    
    def forward(self, h, return_features=False):
        device = h.device
       
        h = self.fc(h)
        logits = torch.empty(h.size(0), self.n_classes).float().to(device)

        for c in range(self.n_classes):
            if isinstance(self.classifiers, nn.DataParallel):
                logits[:, c] = self.classifiers.module[c](h).squeeze(1)
            else:
                logits[:, c] = self.classifiers[c](h).squeeze(1)        

        y_probs = F.softmax(logits, dim = 1)
        m = y_probs.view(1, -1).argmax(1)
        top_indices = torch.cat(((m // self.n_classes).view(-1, 1), (m % self.n_classes).view(-1, 1)), dim=1).view(-1, 1)
        top_instance = logits[top_indices[0]]

        Y_hat = top_indices[1]
        Y_prob = y_probs[top_indices[0]]
        
        results_dict = {}

        if return_features:
            top_features = torch.index_select(h, dim=0, index=top_indices[0])
            results_dict.update({'features': top_features})
        return top_instance, Y_prob, Y_hat, y_probs, results_dict


        
class TransLayer(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim = dim,
            dim_head = dim//8,
            heads = 8,
            num_landmarks = dim//2,    # number of landmarks
            pinv_iterations = 6,    # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual = True,         # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=0.1
        )

    def forward(self, x):
        x = x + self.attn(self.norm(x))

        return x


class PPEG(nn.Module):
    def __init__(self, dim=512):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7//2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5//2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3//2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat)+cnn_feat+self.proj1(cnn_feat)+self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x


class TransMIL(nn.Module):
    def __init__(self, n_classes, size_arg="small", **kwargs):
        super(TransMIL, self).__init__()
        # self.size_dict = {"small": 1024, "big": 1024, "benchmark": 384, "conch": 512}
        self.size_dict = {"small": 1024, "big": 1024, "benchmark": 384, 
                          "conch": 512, "gigapath": 1536, "virchow": 2560}
        size = self.size_dict[size_arg]
        self.pos_layer = PPEG(dim=512)
        self._fc1 = nn.Sequential(nn.Linear(size, 512), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, 512))
        self.n_classes = n_classes
        self.layer1 = TransLayer(dim=512)
        self.layer2 = TransLayer(dim=512)
        self.norm = nn.LayerNorm(512)
        self._fc2 = nn.Linear(512, self.n_classes)

    # def relocate(self):
    #     device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #     self.pos_layer = self.pos_layer.to(device)
    #     self._fc1 = self._fc1.to(device)
    #     self.cls_token = self.cls_token.to(device)
    #     self.layer1 = self.layer1.to(device)
    #     self.layer2 = self.layer2.to(device)
    #     self.norm = self.norm.to(device)
    #     self._fc2 = self._fc2.to(device)

    # def forward_patch_level(self, h):
    #     patch_logits = self.classifiers(h)
    #     return patch_logits

    def forward_patch_level(self, data):
        # if len(data.shape) == 2:
        #     data = data.unsqueeze(0)
        # h = data.float()
        # h = self._fc1(h) #[B, n, 512]
        # # h = h.squeeze(0)
        # logits = self._fc2(h).squeeze(0) #[n, n_classes]
        # return logits
        # h = kwargs['data'].float() #[B, n, 1024]
        if len(data.shape) == 2:
            data = data.unsqueeze(0)
        h = data.float()
        h = self._fc1(h) #[B, n, 512]
        # print("fc1", h.shape)
        
        #---->pad
        H = h.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        # print("H", H, _H, _W)
        add_length = _H * _W - H
        # print("add_length", add_length)
        h = torch.cat([h, h[:,:add_length,:]],dim = 1) #[B, N, 512]
        # print("pad", h.shape)

        #---->cls_token
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).cuda()
        h = torch.cat((cls_tokens, h), dim=1)
        # print("cls_token", h.shape)

        #---->Translayer x1
        h = self.layer1(h) #[B, N, 512]
        # print("translayerx1", h.shape)

        #---->PPEG
        h = self.pos_layer(h, _H, _W) #[B, N, 512]
        # print("PPEG", h.shape)
        
        #---->Translayer x2
        h = self.layer2(h) #[B, N, 512]
        # print("translayerx2", h.shape)

        #---->cls_token
        # h = self.norm(h)[:,0]
        # print("norm", h.shape)

        #---->predict
        logits = self._fc2(h) #[B, n_classes]
        logits = logits.squeeze(0)[1:data.shape[1]+1]
        # print(logits.shape)
        # Y_hat = torch.argmax(logits, dim=1)
        # Y_prob = F.softmax(logits, dim = 1)
        # results_dict = {'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat}
        # return results_dict
        return logits

    def forward(self, data, **kwargs):
        # h = kwargs['data'].float() #[B, n, 1024]
        if len(data.shape) == 2:
            data = data.unsqueeze(0)
        h = data.float()
        h = self._fc1(h) #[B, n, 512]
        # print("fc1", h.shape)
        
        #---->pad
        H = h.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        # print("H", H, _H, _W)
        add_length = _H * _W - H
        # print("add_length", add_length)
        h = torch.cat([h, h[:,:add_length,:]],dim = 1) #[B, N, 512]
        # print("pad", h.shape)

        #---->cls_token
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).cuda()
        h = torch.cat((cls_tokens, h), dim=1)
        # print("cls_token", h.shape)

        #---->Translayer x1
        h = self.layer1(h) #[B, N, 512]
        # print("translayerx1", h.shape)

        #---->PPEG
        h = self.pos_layer(h, _H, _W) #[B, N, 512]
        # print("PPEG", h.shape)
        
        #---->Translayer x2
        h = self.layer2(h) #[B, N, 512]
        # print("translayerx2", h.shape)

        #---->cls_token
        h = self.norm(h)[:,0]
        # print("norm", h.shape)

        #---->predict
        logits = self._fc2(h) #[B, n_classes]
        Y_hat = torch.argmax(logits, dim=1)
        Y_prob = F.softmax(logits, dim = 1)
        # results_dict = {'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat}
        # return results_dict
        return logits, Y_prob, Y_hat, None, None


if __name__ == "__main__":
    model = TransMIL(2).cuda()
    print("HI")
    data = torch.randn(10, 384).cuda()
    # logits, Y_prob, Y_hat, _, _ = model(data)
    logits = model.forward_patch_level(data)
    print(logits)