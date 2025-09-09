import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModel
from torch import nn

class CustomSequential(nn.Module):
    def __init__(self, model, mlp):
        super(CustomSequential, self).__init__()
        self.model = model
        self.mlp = mlp

    def forward(self, *args, **kwargs):
        x = self.model.encode_slide_from_patch_features(*args, **kwargs)
        x = self.mlp(x)
        return x
    
class TITAN(nn.Module):
    def __init__(self, num_classes, only_train_mlp=False):
        super(TITAN, self).__init__()
        self.titan = AutoModel.from_pretrained("MahmoodLab/TITAN", trust_remote_code=True)

        self.mlp = nn.Linear(768, num_classes)
        self.mlp.weight.data.normal_(mean=0.0, std=0.01)
        self.mlp.bias.data.zero_()
        # self.model = CustomSequential(self.titan, mlp)
        if only_train_mlp:
            for param in self.titan.parameters():
                param.requires_grad = False
        
    def forward(self, x):
        feats, coords = x
        if len(feats.shape) == 2:
            feats = feats.unsqueeze(0)
            coords = coords.unsqueeze(0)
        # logits = self.model.encode_slide_from_patch_features(feats, coords, 512)
        # logits = self.model(patch_features=feats, patch_coords=coords, patch_size_lv0=512)
        # logits = self.model(feats, coords, 512)
        logits = self.titan.encode_slide_from_patch_features(feats, coords, 512)
        logits = self.mlp(logits)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        Y_prob = torch.softmax(logits, dim=1)
        return logits, Y_prob, Y_hat, None, None

    def forward_patch_level(self, x):
        feats, coords = x
        if len(feats.shape) == 2:
            feats = feats.unsqueeze(0)
            coords = coords.unsqueeze(0)
        # logits = self.model.encode_slide_from_patch_features(feats, coords, 512)
        # logits = self.model.encode_slide_from_patch_features(feats, coords, 512)
        # logits = self.model(feats, coords, 512)
        # logits = self.titan.encode_slide_from_patch_features(feats, coords, 512)
        logits = self.mlp(feats)
        return logits
    
    def relocate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.titan = self.titan.to(device)
        self.mlp = self.mlp.to(device)


    # def relocate(self):
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #     self.attention_net = self.attention_net.to(device)
    #     self.classifiers = self.classifiers.to(device)
    #     self.instance_classifiers = self.instance_classifiers.to(device)
    