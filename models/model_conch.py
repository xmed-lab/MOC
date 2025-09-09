import torch
from .conch.open_clip_custom import create_model_from_pretrained, get_tokenizer, tokenize
from .lora import LoRA
from torch import nn

def conch_coca(checkpoint_path="conch_checkpoint.bin"):
    model_cfg = 'conch_ViT-B-16'
    model, preprocess = create_model_from_pretrained(model_cfg, checkpoint_path)
    return model


def conch_lora(checkpoint_path="conch_checkpoint.bin", r=8, lora_cnt=None):
    model_cfg = 'conch_ViT-B-16'
    model, preprocess = create_model_from_pretrained(model_cfg, checkpoint_path)
    model.visual.trunk = LoRA(model.visual.trunk, r=r, lora_cnt=lora_cnt)
    return model

class Conch_LoRA(nn.Module):
    def __init__(self, checkpoint_path="conch_checkpoint.bin", r=8, lora_cnt=None):
        super(Conch_LoRA, self).__init__()
        self.r = r
        self.lora_cnt = lora_cnt
        self.model = conch_lora(checkpoint_path=checkpoint_path, r=self.r, lora_cnt=self.lora_cnt)
        # self.model = conch_coca(checkpoint_path=checkpoint_path)
    
    def forward(self, x):
        return self.model.encode_image(x)
    
# model = conch_coca()
# print(model.logit_scale)
# print(model.text.ln_final.weight.dtype)
# # m = torch.rand(1,3,448,448)
# # emb = model.encode_image(m)
# # print(emb.shape)

# # print(model.ln_final.weight.shape)
# print(model.text.ln_final.weight.shape)
# tokenizer = get_tokenizer()
# text = "A cat is sitting on the window"
# print(tokenizer.encode(text))