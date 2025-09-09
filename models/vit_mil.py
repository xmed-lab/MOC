import torch
from timm.models.vision_transformer import VisionTransformer
import os
from huggingface_hub import hf_hub_download
# from transformers import login

def get_pretrained_url(key):
    URL_PREFIX = "https://github.com/lunit-io/benchmark-ssl-pathology/releases/download/pretrained-weights"
    model_zoo_registry = {
        "DINO_p16": "dino_vit_small_patch16_ep200.torch",
        "DINO_p8": "dino_vit_small_patch8_ep200.torch",
    }
    pretrained_url = f"{URL_PREFIX}/{model_zoo_registry.get(key)}"
    return pretrained_url


uni_pretrain_path = "/home/txiang/pathology/CLAM/models/uni.bin"
def get_uni_pretrain():
    if not os.path.exists("pytorch_model.bin") and not os.path.exists(uni_pretrain_path):
        hf_hub_download("MahmoodLab/UNI", filename="pytorch_model.bin", local_dir="./", force_download=True)


def vit_small(pretrained, progress, key, **kwargs):
    patch_size = kwargs.get("patch_size", 16)
    model = VisionTransformer(
        img_size=224, patch_size=patch_size, embed_dim=384, num_heads=6, num_classes=0
    )
    if pretrained:
        pretrained_url = get_pretrained_url(key)
        verbose = model.load_state_dict(
            torch.hub.load_state_dict_from_url(pretrained_url, progress=progress)
        )
        print(verbose)
    return model

def vit_large(pretrained=True, **kwargs):
    model = VisionTransformer(
        img_size=224, patch_size=16, embed_dim=1024, num_heads=16, num_classes=0, depth=24, init_values=1e-5
    )
    if pretrained:
        # get_uni_pretrain()
        # verbose = model.load_state_dict(torch.load("pytorch_model.bin"))
        verbose = model.load_state_dict(torch.load("/home/txiang/pathology/CLAM/models/uni.bin"))
        print(verbose)
    return model

def vit_large_decur(pretrained=True, **kwargs):
    model = VisionTransformer(
        img_size=224, patch_size=16, embed_dim=1024, num_heads=16, num_classes=0, depth=24, init_values=1e-5
    )
    if pretrained:
        decur_ckpt = torch.load("/home/txiang/pathology/CLAM/models/decur7168.pth")
        state_dict = decur_ckpt['model']
        state_dict = {k: v for k, v in state_dict.items() if k.startswith("backbone_1")}
        state_dict = {k.replace("backbone_1.", ""): v for k,v in state_dict.items()}
        
        verbose = model.load_state_dict(state_dict)
        print(verbose)
    return model


if __name__ == "__main__":
    model = vit_large_decur(pretrained=True)
    print("Done")